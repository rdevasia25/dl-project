"""Sequential JSONL driver. Each nonblank input line is one genuine user turn."""
import argparse
from contextlib import ExitStack
import json
from itertools import islice
import sys
from .core import extract_turn, validate_request
from .runtime import DeepInfraRuntime, MockRuntime
from .types import ExtractionRequest, ExtractionStatus


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--output', required=True, help='New output JSONL file (never overwritten)')
    parser.add_argument('--mode', required=True, choices=('mock', 'live'))
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
                runtime = DeepInfraRuntime()
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
