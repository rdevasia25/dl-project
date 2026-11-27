"""Always-on constraint extraction and dataset replay."""
import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
from io import StringIO
from itertools import islice
import json
from pathlib import Path
import subprocess
import sys

from sc_common import (
    Constraint, DeepInfraRuntime, ExtractionError, ExtractionRequest,
    ExtractionResult, ExtractionStatus, MockRuntime, ModelResponse, ModelRuntime,
    OllamaRuntime, RegistryEntry, extract_turn, render_registry, validate_request,
)

# Originally sc_baseline/__init__.py
# Preserve the public imports used by Jev and existing callers.
__all__ = ['extract_turn', 'render_registry', 'Constraint', 'ExtractionRequest',
           'ExtractionResult', 'ExtractionStatus', 'ExtractionError', 'RegistryEntry',
           'ModelResponse', 'ModelRuntime']


# Originally sc_baseline/dataset.py
# Map released inputs and replay turns with independent registries.
@dataclass(frozen=True)
class DatasetTurn:
    request: ExtractionRequest
    source_id: str
    source_conversation_id: str
    turn_index: int
    variant_id: str | None = None
    injection_turn: bool = False

    def source_dict(self):
        return {'source_id': self.source_id,
                'conversation_id': self.source_conversation_id,
                'turn_index': self.turn_index, 'variant_id': self.variant_id,
                'injection_turn': self.injection_turn}


def _identity(row, id_field='id'):
    if not isinstance(row, dict):
        raise ValueError('Dataset rows must be objects.')
    for field in (id_field, 'conversation_id'):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f'Missing or invalid {field}.')
    if type(row.get('turn_index')) is not int or row['turn_index'] < 0:
        raise ValueError('turn_index must be a nonnegative integer.')
    req = ExtractionRequest(row['conversation_id'], row[id_field], row.get('user_message'),
                            row.get('previous_assistant_message'))
    validate_request(req)
    return req


def _namespace(kind, value):
    return json.dumps([kind, value], ensure_ascii=False, separators=(',', ':'))


def _ordinary(rows, mode):
    seen, previous, turns = set(), {}, []
    for row in rows:
        req = _identity(row)
        if req.message_id in seen:
            raise ValueError('Duplicate dataset id.')
        seen.add(req.message_id)
        if mode == 'stream':
            index = row['turn_index']
            if index <= previous.get(req.conversation_id, -1):
                raise ValueError('Stream turn indices must increase within each conversation.')
            previous[req.conversation_id] = index
        else:
            req = ExtractionRequest(_namespace('example', req.message_id), req.message_id,
                                    req.user_message, req.previous_assistant_message)
        turns.append(DatasetTurn(req, row['id'], row['conversation_id'], row['turn_index']))
    return turns


def prepare_turns(rows, mode, *, clean_rows=None):
    """Validate all rows before inference; injection mode replays full variants.

    No gold fields are read. The supplied injected user text replaces exactly one
    clean user turn; subsequent observed assistant context stays as released.
    """
    if mode not in ('examples', 'stream', 'injections'):
        raise ValueError('Unknown dataset mode.')
    if mode != 'injections':
        if clean_rows is not None:
            raise ValueError('clean_rows is only valid for injections.')
        return _ordinary(rows, mode)
    if clean_rows is None:
        raise ValueError('Injection replay requires clean stream rows.')
    clean = _ordinary(clean_rows, 'stream')
    conversations, locations = {}, {}
    for turn in clean:
        conversations.setdefault(turn.source_conversation_id, []).append(turn)
        locations[(turn.source_conversation_id, turn.turn_index)] = turn
    seen, result = set(), []
    for row in rows:
        injected = _identity(row, 'conpact_id')
        variant = row['conpact_id']
        if variant in seen:
            raise ValueError('Duplicate conpact_id.')
        seen.add(variant)
        target = locations.get((injected.conversation_id, row['turn_index']))
        if target is None or injected.previous_assistant_message != target.request.previous_assistant_message:
            raise ValueError('Injection location/context does not match clean stream.')
        for turn in conversations[injected.conversation_id]:
            changed = turn.turn_index == row['turn_index']
            req = ExtractionRequest(_namespace('variant', variant), turn.request.message_id,
                injected.user_message if changed else turn.request.user_message,
                turn.request.previous_assistant_message)
            result.append(DatasetTurn(req, turn.source_id, turn.source_conversation_id,
                                      turn.turn_index, variant, changed))
    return result


def run_turns(turns, runtime):
    """Run each planned turn exactly once, with isolated per-execution state."""
    registries = {}
    for turn in turns:
        key = turn.request.conversation_id
        result = extract_turn(turn.request, registries.get(key, []), runtime)
        registries[key] = result.registry
        yield {'source': turn.source_dict(), 'extraction': result.to_dict()}
        if result.status != ExtractionStatus.SUCCESS:
            return


# Originally sc_baseline/runner.py
# Read JSONL and run the generic always-on baseline.
def run_records(records, runtime, *, limit=None):
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer.")
    registries, seen = {}, set()
    selected = records if limit is None else islice(records, limit)
    for record in selected:
        if not isinstance(record, dict):
            raise ValueError('Each record must be an object.')
        try:
            request = ExtractionRequest(**record)
        except TypeError:
            raise ValueError('Unexpected or missing request fields.') from None
        validate_request(request)
        key = (request.conversation_id, request.message_id)
        if key in seen:
            raise ValueError('Duplicate message ID within conversation.')
        seen.add(key)
        result = extract_turn(request, registries.get(request.conversation_id, []), runtime)
        registries[request.conversation_id] = result.registry
        yield result
        if result.status != ExtractionStatus.SUCCESS:
            return


def read_jsonl(stream):
    for number, line in enumerate(stream, 1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f'Invalid JSON on line {number}.') from None


def generic_main(argv=None):
    parser = argparse.ArgumentParser(description='Run sequential JSONL user turns.')
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--output', required=True, help='New output JSONL file (never overwritten)')
    parser.add_argument('--mode', required=True, choices=('mock', 'live', 'local'))
    parser.add_argument('--mock-responses', help='JSONL file of raw response strings, one per call')
    parser.add_argument('--limit', type=int, help='Maximum user messages')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if (args.mode == 'mock') != bool(args.mock_responses):
        parser.error('--mock-responses is required only in mock mode')
    with ExitStack() as stack:
        try:
            source = stack.enter_context(open(args.input, encoding='utf-8'))
            if args.mode == 'mock':
                mock_file = stack.enter_context(open(args.mock_responses, encoding='utf-8'))
                responses = list(read_jsonl(mock_file))
                if not all(isinstance(r, str) for r in responses):
                    raise ValueError('Mock responses must be JSON strings.')
                runtime = MockRuntime(responses)
            else:
                runtime = OllamaRuntime() if args.mode == 'local' else DeepInfraRuntime()
            # Open exclusively before any inference, protecting input and existing results.
            target = stack.enter_context(open(args.output, 'x', encoding='utf-8'))
            for result in run_records(read_jsonl(source), runtime, limit=args.limit):
                target.write(json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False) + '\n')
                target.flush()
                if result.status != ExtractionStatus.SUCCESS:
                    return 1
            return 0
        except (OSError, ValueError):
            print('Runner failed: invalid input/configuration or inaccessible/existing file. '
                  'Any completed results remain in the output; no retries were made.', file=sys.stderr)
            return 2


# Originally sc_baseline/dataset_runner.py
# Run released dataset modes and write outputs and manifests.
def _load(path):
    data = Path(path).read_bytes()
    # StringIO iterates physical lines, preserving U+2028 inside JSON strings.
    rows = list(read_jsonl(StringIO(data.decode('utf-8'))))
    return rows, {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}


def _code_identity():
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    digest = hashlib.sha256()
    for name in ('sc_baseline.py', 'sc_common.py'):
        path = Path(__file__).with_name(name)
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return {'revision': revision, 'package_sha256': digest.hexdigest()}


def dataset_main(argv=None):
    parser = argparse.ArgumentParser(description='Replay released dataset examples, clean streams or injection variants.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--dataset-mode', required=True, choices=('examples','stream','injections'))
    parser.add_argument('--clean-stream', help='Required for full injection replays')
    parser.add_argument('--dataset-revision', required=True, help='Source dataset Git commit')
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', required=True, choices=('mock','live','local'))
    parser.add_argument('--mock-responses')
    parser.add_argument('--limit', type=int, help='First N extraction turns; incomplete smoke run')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if (args.mode == 'mock') != bool(args.mock_responses):
        parser.error('--mock-responses is required only in mock mode')
    if (args.dataset_mode == 'injections') != bool(args.clean_stream):
        parser.error('--clean-stream is required only for injections')
    if not args.dataset_revision.strip():
        parser.error('--dataset-revision must be nonempty')
    manifest_path = Path(str(args.output) + '.manifest.json')
    manifest = None
    try:
        rows, source = _load(args.input)
        clean, clean_source = _load(args.clean_stream) if args.clean_stream else (None, None)
        turns = prepare_turns(rows, args.dataset_mode, clean_rows=clean)
        selected_count = min(args.limit, len(turns)) if args.limit else len(turns)
        mock_source = None
        if args.mode == 'mock':
            responses, mock_source = _load(args.mock_responses)
            if not all(isinstance(x, str) for x in responses):
                raise ValueError('Mock responses must be JSON strings.')
            if len(responses) < selected_count:
                raise ValueError('Not enough mock responses for selected turns.')
            runtime = MockRuntime(responses)
        else:
            runtime = OllamaRuntime() if args.mode == 'local' else DeepInfraRuntime()
        manifest = {'schema_version': 1, 'dataset_revision': args.dataset_revision,
            'input': source, 'clean_stream': clean_source, 'mock_responses': mock_source,
            'code': _code_identity(), 'dataset_mode': args.dataset_mode, 'mode': args.mode,
            'provider': runtime.provider, 'model': runtime.model, 'config_id': runtime.config_id,
            'source_rows': len(rows), 'planned_turns': len(turns), 'selected_turns': selected_count,
            'processed_turns': 0, 'request_count': 0, 'known_cost_usd': 0.0,
            'unknown_cost_results': 0, 'unknown_usage_results': 0, 'total_cost_usd': None,
            'status': 'running'}
        # Both exclusive handles must exist before any generation. No overwrites.
        with ExitStack() as stack:
            target = stack.enter_context(open(args.output, 'x', encoding='utf-8'))
            audit = stack.enter_context(open(manifest_path, 'x', encoding='utf-8'))
            def save():
                audit.seek(0)
                json.dump(manifest, audit, indent=2, ensure_ascii=False, allow_nan=False)
                audit.write('\n'); audit.truncate(); audit.flush()
            save()
            try:
                selected = islice(turns, selected_count)
                for record in run_turns(selected, runtime):
                    target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                    target.flush()
                    result = record['extraction']
                    manifest['processed_turns'] += 1
                    manifest['request_count'] += result['request_count']
                    if result['cost_usd'] is None:
                        manifest['unknown_cost_results'] += 1
                    else:
                        manifest['known_cost_usd'] += result['cost_usd']
                    manifest['unknown_usage_results'] += int(result['usage'] is None)
                    if result['status'] != 'success':
                        manifest['status'] = 'failed'
                        break
                else:
                    manifest['status'] = 'limited' if selected_count < len(turns) else 'complete'
            except BaseException:
                manifest['status'] = 'failed'
                raise
            finally:
                if not manifest['unknown_cost_results'] and manifest['processed_turns']:
                    manifest['total_cost_usd'] = manifest['known_cost_usd']
                save()
        return 1 if manifest['status'] == 'failed' else 0
    except (OSError, ValueError):
        print('Dataset run failed: invalid data/configuration or inaccessible/existing artifact. '
              'Completed results are preserved; no retries were made.', file=sys.stderr)
        return 2



# Originally sc_baseline/__main__.py
# Run the generic CLI by default and dataset replay with the dataset subcommand.
def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == 'dataset':
        return dataset_main(args[1:])
    return generic_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
