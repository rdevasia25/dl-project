import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from urllib.error import HTTPError, URLError
from unittest.mock import patch

from sc_baseline import *
from sc_baseline.parsing import parse_extractor_output, parse_with_errors
from sc_baseline.prompts import build_sc_extraction_prompts
from sc_baseline.runner import run_records
from sc_baseline.runtime import DeepInfraRuntime, MockRuntime, RuntimeFailure, MODEL, SETTINGS

FIXTURES = Path(__file__).parent / 'fixtures'


def reference(name):
    spec = importlib.util.spec_from_file_location(name, FIXTURES / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def request(text='Hello', message='1'):
    return ExtractionRequest('a', message, text)


def raw(*texts):
    return json.dumps({'scs': [{'text': t, 'evidence': 'e'} for t in texts]})


class ContractTests(unittest.TestCase):
    def test_snapshot_roundtrip(self):
        rt = MockRuntime([ModelResponse(raw('A'), 'scripted-mock', {'nested': []})])
        result = extract_turn(request(), [], rt)
        snap = result.to_dict()
        snap['usage']['nested'].append(1)
        snap['registry'][0]['text'] = 'changed'
        self.assertEqual(result.usage, {'nested': []})
        self.assertEqual(result.registry[0].text, 'A')
        self.assertEqual(json.loads(json.dumps(snap))['status'], 'success')

    def test_optional_unknown_defaults(self):
        result = extract_turn(request(''), [], MockRuntime([raw()]))
        self.assertIsNone(result.usage)
        self.assertIsNone(result.cost_usd)
        self.assertIsNone(request().previous_assistant_message)
        self.assertEqual(result.errors, [])
        result.errors.append(ExtractionError('x', 'x'))
        self.assertEqual(extract_turn(request(), [], MockRuntime([raw()])).errors, [])


class CoreTests(unittest.TestCase):
    def test_prompt_exact_parity(self):
        upstream = reference('upstream_prompt_reference')
        for text in ('', 'Hello', 'Always be brief.', '  a\nb\n  c'):
            for existing in (None, [], ['A'], ['A', 'B\nC']):
                for previous in (None, '', 'Use bullets?\nYes.'):
                    with self.subTest(text=text, existing=existing, previous=previous):
                        self.assertEqual(build_sc_extraction_prompts(text, existing, previous),
                            upstream.build_sc_extraction_prompts(text, existing, previous))

    def test_parser_acceptance_parity(self):
        upstream = reference('upstream_parser_reference')
        cases = ['', 'bad', 'null', '[]', '{}', '{"scs":false}', raw(), raw('A','A'),
                 '```json\n'+raw('A')+'\n```', 'prose '+raw(),
                 'prose '+raw('A'), '{"scs":[{"text":" A ","evidence":null},3,{},'
                 '{"text":"B","evidence":[1]}]}',
                 '```json\n{}\n```'+raw('B'), '{"scs":[{"text":"x"}]} trailing']
        for value in cases:
            with self.subTest(value=value):
                self.assertEqual(parse_extractor_output(value), upstream.parse_extractor_output(value))

    def test_every_turn_and_context(self):
        rt = MockRuntime([raw(), raw('Keep short.'), raw(), raw()])
        state = []
        for i, text in enumerate(['Hi', 'Always keep short.', 'Use bullets this once.', '']):
            state = extract_turn(request(text, str(i)), state, rt).registry
        self.assertEqual(len(rt.calls), 4)
        self.assertIn('Keep short.', rt.calls[-1][1]['content'])
        self.assertEqual(len(state), 1)
        context_runtime = MockRuntime([raw()])
        extract_turn(ExtractionRequest('a','x','Do that going forward.','Use metric?'), [], context_runtime)
        self.assertIn('Use metric?', context_runtime.calls[0][1]['content'])

    def test_append_only_modifications_revocations(self):
        state = []
        rt = MockRuntime([raw('Be brief.'), raw('Be detailed.'), raw()])
        for i, text in enumerate(['Always be brief.', 'Now always be detailed.', 'Revoke earlier style rules.']):
            old = list(state)
            result = extract_turn(request(text,str(i)), state, rt)
            self.assertEqual(state, old)
            state = result.registry
        self.assertEqual([e.text for e in state], ['Be brief.', 'Be detailed.'])

    def test_exact_dedup_and_evidence_disabled(self):
        rt = MockRuntime([raw(' A ','A','a','A!')])
        result = extract_turn(request(), [], rt)
        self.assertEqual([e.text for e in result.registry], ['A','a','A!'])
        self.assertEqual(result.registry[0].source_message_id, '1')
        self.assertEqual(render_registry(result.registry), 'Constraints from earlier in this session:\n1. A\n2. a\n3. A!')
        self.assertIsNone(render_registry([]))

    def test_malformed_and_partial(self):
        for value in ('', 'bad', '[]', '{}', '{"scs":3}', '{"scs":[{}]}'):
            with self.subTest(value=value):
                result = extract_turn(request(), [], MockRuntime([value]))
                self.assertEqual(result.status, 'response_error')
                self.assertEqual(result.request_count, 1)
                self.assertEqual(result.raw_output, value)
        result = extract_turn(request(), [], MockRuntime(['{"scs":[{"text":"A"},3]}']))
        self.assertEqual(result.status, 'response_error')
        self.assertEqual(len(result.registry), 1)

    def test_truncation_and_model_mismatch(self):
        result = extract_turn(request(), [], MockRuntime([ModelResponse(raw('A'),'scripted-mock',finish_reason='length')]))
        self.assertEqual(result.status, 'response_error')
        self.assertEqual(len(result.registry), 1)
        result = extract_turn(request(), [], MockRuntime([ModelResponse(raw('A'),'wrong')]))
        self.assertEqual(result.status, 'response_error')
        self.assertEqual(result.registry, [])

    def test_failures_no_retry_or_secret(self):
        for exc in (RuntimeFailure('timeout','Provider request timed out.'), RuntimeError('secret-key')):
            rt = MockRuntime([exc])
            result = extract_turn(request(), [RegistryEntry('A','e','0')], rt)
            self.assertEqual(result.status, 'provider_error')
            self.assertEqual(len(result.registry), 1)
            self.assertEqual(len(rt.calls), 1)
            self.assertNotIn('secret-key', json.dumps(result.to_dict()))

    def test_invalid_input_no_invocation(self):
        rt = MockRuntime([])
        for req in (ExtractionRequest('', 'm',''), ExtractionRequest('c','', ''), request(None),
                    ExtractionRequest('c','m','x',123)):
            with self.assertRaises(ValueError): extract_turn(req, [], rt)
        with self.assertRaises(ValueError): extract_turn(request(), ['bad'], rt)
        self.assertEqual(rt.calls, [])

    def test_invalid_runtime_response(self):
        class Bad:
            provider=model=config_id='bad'
            def generate(self, messages): return 42
        self.assertEqual(extract_turn(request(), [], Bad()).status, 'response_error')


class Opener:
    def __init__(self, data=None, error=None):
        self.data, self.error, self.calls = data, error, []
    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if self.error: raise self.error
        return io.BytesIO(self.data)


def envelope(**updates):
    data = {'model': MODEL, 'choices': [{'message': {'content': raw()}, 'finish_reason': 'stop'}]}
    data.update(updates)
    return json.dumps(data).encode()


class RuntimeTests(unittest.TestCase):
    def test_request_and_usage(self):
        opener = Opener(envelope(usage={'prompt_tokens':10, 'completion_tokens':2, 'estimated_cost':0.0001}))
        rt = DeepInfraRuntime(api_key='fake',opener=opener)
        result = extract_turn(request(), [], rt)
        self.assertEqual(result.status, 'success')
        self.assertEqual(result.cost_usd, 0.0001)
        self.assertEqual(result.cost_source, 'provider:usage.estimated_cost')
        req, timeout = opener.calls[0]
        payload = json.loads(req.data)
        self.assertEqual(payload, {'model': MODEL, 'messages':opener_payload_messages(), **SETTINGS})
        self.assertEqual(payload['reasoning_effort'], 'none')
        self.assertEqual(payload['chat_template_kwargs'], {'enable_thinking': False})
        self.assertEqual(payload['max_tokens'], 4096)
        self.assertEqual(req.full_url,'https://api.deepinfra.com/v1/openai/chat/completions')
        self.assertEqual(req.get_header('Authorization'), 'Bearer fake')
        self.assertGreaterEqual(result.latency_ms, 0)
        self.assertEqual(len(opener.calls),1)

    def test_missing_key(self):
        opener = Opener()
        result = extract_turn(request(), [], DeepInfraRuntime(api_key='',opener=opener))
        self.assertEqual(result.status, 'not_invoked')
        self.assertEqual(result.request_count, 0)
        self.assertIsNone(result.latency_ms)
        self.assertEqual(opener.calls, [])

    def test_transport_errors(self):
        for exc in (TimeoutError(), URLError('secret'), HTTPError('url',429,'secret',{},None)):
            opener = Opener(error=exc)
            result = extract_turn(request(), [], DeepInfraRuntime(api_key='fake',opener=opener))
            self.assertEqual(result.status, 'provider_error')
            self.assertIsNone(result.usage)
            self.assertIsNone(result.cost_usd)
            self.assertEqual(len(opener.calls),1)
            self.assertNotIn('secret',json.dumps(result.to_dict()))

    def test_provider_envelope_errors(self):
        for body in (b'bad',b'null',b'{}',b'{"choices":[]}',b'{"error":"secret"}',
                     envelope(usage=[]), envelope(choices=[{'message':{'content':None}}])):
            result = extract_turn(request(), [], DeepInfraRuntime(api_key='fake',opener=Opener(body)))
            self.assertEqual(result.status, 'provider_error')

    def test_unknown_cost_and_usage(self):
        for usage in (None, {}, {'estimated_cost':-1}, {'estimated_cost':True}):
            result = extract_turn(request(), [], DeepInfraRuntime(api_key='fake',opener=Opener(envelope(usage=usage))))
            self.assertEqual(result.status,'success')
            self.assertIsNone(result.cost_usd)

    def test_thinking_violation(self):
        body=envelope(choices=[{'message':{'content':raw(),'reasoning_content':'thinking'}}])
        result=extract_turn(request(),[],DeepInfraRuntime(api_key='fake',opener=Opener(body)))
        self.assertEqual(result.errors[0].code,'thinking_enabled')

    def test_config_identity(self):
        a=DeepInfraRuntime(api_key='a'); b=DeepInfraRuntime(api_key='b')
        self.assertEqual(a.config_id,b.config_id)
        self.assertNotEqual(a.config_id,MockRuntime([]).config_id)
        with patch.dict(SETTINGS, reasoning_effort='low'):
            self.assertNotEqual(a.config_id, DeepInfraRuntime(api_key='a').config_id)
        for timeout in (0,-1,float('inf'),True):
            with self.assertRaises(ValueError): DeepInfraRuntime(timeout=timeout)


def opener_payload_messages():
    system,user=build_sc_extraction_prompts('Hello',[],None)
    return [{'role':'system','content':system},{'role':'user','content':user}]


class RunnerTests(unittest.TestCase):
    def records(self):
        return [json.loads(x) for x in (FIXTURES/'turns.jsonl').read_text().splitlines()]

    def test_conversations_and_every_turn(self):
        rt=MockRuntime([raw('A'),raw(),raw()])
        rows=list(run_records(self.records(),rt))
        self.assertEqual(len(rt.calls),3)
        self.assertEqual(rows[1].registry,[])
        self.assertEqual(rows[2].registry,rows[0].registry)

    def test_duplicate_ids(self):
        rt=MockRuntime([raw()])
        with self.assertRaises(ValueError): list(run_records([self.records()[0]]*2,rt))
        self.assertEqual(len(rt.calls),1)

    def test_invalid_records(self):
        for record in ([],{}, {'role':'assistant'}, {'conversation_id':'a','message_id':'1','user_message':3}):
            with self.assertRaises(ValueError): list(run_records([record],MockRuntime([])))

    def test_limit(self):
        rt=MockRuntime([raw()])
        self.assertEqual(len(list(run_records(self.records(),rt,limit=1))),1)
        self.assertEqual(len(rt.calls),1)

    def test_limit_does_not_consume_next_record(self):
        def records():
            yield self.records()[0]
            raise ValueError('Must not read beyond limit.')
        self.assertEqual(len(list(run_records(records(), MockRuntime([raw()]), limit=1))), 1)

    def test_redirects_disabled(self):
        from sc_baseline.runtime import _NoRedirect
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, '', {}, 'https://example.com'))

    def test_error_stops(self):
        rt=MockRuntime(['bad',raw()])
        rows=list(run_records(self.records(),rt))
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0].status,'response_error')
        self.assertEqual(len(rt.calls),1)

    def test_mock_exhaustion(self):
        result=extract_turn(request(),[],MockRuntime([]))
        self.assertEqual(result.status,'not_invoked')

    def test_cli_fixture_and_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'results.jsonl'
            cmd=[sys.executable,'-m','sc_baseline','--mode','mock','--input',str(FIXTURES/'turns.jsonl'),
                 '--mock-responses',str(FIXTURES/'responses.jsonl'),'--output',str(output)]
            proc=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            rows=[json.loads(x) for x in output.read_text().splitlines()]
            self.assertEqual(sum(r['request_count'] for r in rows),3)
            before=output.read_bytes()
            self.assertEqual(subprocess.run(cmd,capture_output=True).returncode,2)
            self.assertEqual(output.read_bytes(),before)

    def test_cli_bad_json_and_failure_record(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'in'; mock=Path(d)/'mock'; out=Path(d)/'out'
            source.write_text('bad\n'); mock.write_text(json.dumps(raw())+'\n')
            cmd=[sys.executable,'-m','sc_baseline','--mode','mock','--input',str(source),
                 '--mock-responses',str(mock),'--output',str(out)]
            self.assertEqual(subprocess.run(cmd,capture_output=True).returncode,2)
            out.unlink(); source.write_text(json.dumps(self.records()[0])+'\n')
            mock.write_text(json.dumps('bad')+'\n')
            self.assertEqual(subprocess.run(cmd,capture_output=True).returncode,1)
            self.assertEqual(json.loads(out.read_text())['status'],'response_error')


if __name__ == '__main__':
    unittest.main()
