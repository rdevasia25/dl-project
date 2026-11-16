"""Dataset CLI: examples, clean stream, or full independent injection replays."""
import argparse
from contextlib import ExitStack
import hashlib
from io import StringIO
from itertools import islice
import json
from pathlib import Path
import subprocess
import sys
from .dataset import prepare_turns, run_turns
from .runner import read_jsonl
from .runtime import DeepInfraRuntime, MockRuntime


def _load(path):
    data = Path(path).read_bytes()
    # StringIO iterates physical lines, preserving U+2028 inside JSON strings.
    rows = list(read_jsonl(StringIO(data.decode('utf-8'))))
    return rows, {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}


def _code_identity():
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent.parent, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return {'revision': revision, 'package_sha256': digest.hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--dataset-mode', required=True, choices=('examples','stream','injections'))
    parser.add_argument('--clean-stream', help='Required for full injection replays')
    parser.add_argument('--dataset-revision', required=True, help='Source dataset Git commit')
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', required=True, choices=('mock','live'))
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
            runtime = DeepInfraRuntime()
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


if __name__ == '__main__':
    raise SystemExit(main())
