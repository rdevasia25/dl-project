"""Synthetic dataset integration tests; no annotation or model evaluation."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from sc_baseline.dataset import prepare_turns, run_turns
from sc_baseline.dataset_runner import main
from sc_baseline.runner import read_jsonl
from sc_baseline.runtime import MockRuntime, RuntimeFailure
from sc_baseline import ModelResponse


def row(id='r1', conversation='c', index=0, text='Hello', **extras):
    return dict(id=id, conversation_id=conversation, turn_index=index,
                user_message=text, previous_assistant_message=None, **extras)


def injection(id='v1', index=0, text='Always be brief.'):
    return dict(conpact_id=id, conversation_id='c', turn_index=index,
                user_message=text, previous_assistant_message=None,
                gold_constraint='SECRET GOLD')


def raw(text=None):
    return json.dumps({'scs': [] if text is None else [{'text':text,'evidence':'e'}]})


class MappingTests(unittest.TestCase):
    def test_examples_independent_and_no_gold_leak(self):
        records = [row(text='Always brief.',gold_constraint='SECRET',is_persistent_constraint=True),
                   row(id='r2',text='Hello')]
        turns = prepare_turns(records,'examples')
        rt=MockRuntime([raw('Be brief.'),raw()])
        results=list(run_turns(turns,rt))
        self.assertNotEqual(turns[0].request.conversation_id,turns[1].request.conversation_id)
        self.assertEqual(results[1]['extraction']['registry'],[])
        self.assertNotIn('Be brief.',rt.calls[1][1]['content'])
        self.assertNotIn('SECRET',json.dumps(rt.calls))
        self.assertEqual(results[0]['source']['source_id'],'r1')
        self.assertEqual(results[0]['source']['conversation_id'],'c')

    def test_stream_interleaving_gaps_and_context(self):
        records=[row(text='Always brief.'),row('b1','b',0),
                 {**row('r2','c',5),'previous_assistant_message':'previous full context'}]
        rt=MockRuntime([raw('Be brief.'),raw(),raw()])
        results=list(run_turns(prepare_turns(records,'stream'),rt))
        self.assertEqual(results[1]['extraction']['registry'],[])
        self.assertEqual(len(results[2]['extraction']['registry']),1)
        self.assertIn('Be brief.',rt.calls[2][1]['content'])
        self.assertIn('previous full context',rt.calls[2][1]['content'])
        self.assertEqual(len(rt.calls),3)

    def test_every_message_intact(self):
        texts=['', '你好', 'x'*13000]
        records=[{**row(str(i),'c',i,text),'previous_assistant_message':'y'*12000}
                 for i,text in enumerate(texts)]
        rt=MockRuntime([raw()]*3)
        results=list(run_turns(prepare_turns(records,'stream'),rt))
        for i,text in enumerate(texts):
            self.assertIn(text,rt.calls[i][1]['content'])
            self.assertIn('y'*12000,rt.calls[i][1]['content'])
        self.assertEqual(sum(r['extraction']['request_count'] for r in results),3)

    def test_full_injected_replays_fresh_state(self):
        clean=[row(),row('r2','c',2),row('b1','b',0)]
        turns=prepare_turns([injection('v1'),injection('v2',text='Always detailed.')],
                            'injections',clean_rows=clean)
        self.assertEqual(len(turns),4)
        self.assertEqual([t.injection_turn for t in turns],[True,False,True,False])
        rt=MockRuntime([raw('Be brief.'),raw(),raw('Be detailed.'),raw()])
        results=list(run_turns(turns,rt))
        self.assertIn('Be brief.',rt.calls[1][1]['content'])
        self.assertNotIn('Be brief.',rt.calls[2][1]['content'])
        self.assertIn('Be detailed.',rt.calls[3][1]['content'])
        self.assertEqual([r['source']['variant_id'] for r in results],['v1','v1','v2','v2'])
        self.assertEqual([r['source']['source_id'] for r in results],['r1','r2','r1','r2'])
        self.assertNotIn('SECRET GOLD',json.dumps(rt.calls))

    def test_noninitial_injection_replays_prefix(self):
        turns=prepare_turns([injection(index=2)],'injections',clean_rows=[row(text='prefix'),row('r2','c',2)])
        self.assertEqual([t.request.user_message for t in turns],['prefix','Always be brief.'])
        self.assertEqual([t.injection_turn for t in turns],[False,True])

    def test_injection_mismatch(self):
        for value in (injection(index=99),{**injection(),'previous_assistant_message':'mismatch'}):
            with self.assertRaises(ValueError): prepare_turns([value],'injections',clean_rows=[row()])
        with self.assertRaises(ValueError): prepare_turns([injection()],'injections')
        with self.assertRaises(ValueError): prepare_turns([injection()]*2,'injections',clean_rows=[row()])

    def test_invalid_schema(self):
        for value in (None,[],{},row(id=''),row(index=True),row(index=-1),row(text=None),
                      {**row(),'previous_assistant_message':1}):
            with self.subTest(value=value):
                with self.assertRaises(ValueError): prepare_turns([value],'examples')
        with self.assertRaises(ValueError): prepare_turns([row(),row()],'examples')
        with self.assertRaises(ValueError): prepare_turns([],'unknown')
        with self.assertRaises(ValueError): prepare_turns([],'stream',clean_rows=[])

    def test_stream_order(self):
        for indices in ((0,0),(2,1)):
            with self.assertRaises(ValueError):
                prepare_turns([row(str(i),index=n) for i,n in enumerate(indices)],'stream')

    def test_operational_errors_stop(self):
        turns=prepare_turns([row(),row('r2',index=1)],'stream')
        for response in ('bad',RuntimeFailure('timeout','timeout')):
            rt=MockRuntime([response,raw()]);results=list(run_turns(turns,rt))
            self.assertEqual(len(results),1)
            self.assertEqual(len(rt.calls),1)
            self.assertNotEqual(results[0]['extraction']['status'],'success')

    def test_unicode_physical_lines(self):
        value=json.dumps(row(text='a\u2028b\u0085c'),ensure_ascii=False)+'\n'
        self.assertEqual(len(list(read_jsonl(io.StringIO(value)))),1)

    def test_jev_callback_without_network(self):
        # Load only the teammate's exact callback definition; no requests dependency
        # or import-time provider setup, and no changes to the teammate file.
        source=Path(__file__).resolve().parent.parent/'src/jev_gate.py'
        if not source.exists():
            source=Path('/private/tmp/dl-main-inspection-5e8013b/src/jev_gate.py')
        if not source.exists(): self.skipTest('Teammate snapshot unavailable')
        tree=ast.parse(source.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gated_extract')
        namespace={'DEFAULT_THRESHOLD':0.5,'jev_score':lambda *a:{'score':1},
                   'should_extract':lambda score,threshold:score>=threshold}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
        from sc_baseline import extract_turn
        turn=prepare_turns([row()],'examples')[0]
        rt=MockRuntime([raw()])
        result=namespace['gated_extract']('Hello',None,lambda:extract_turn(turn.request,[],rt))
        self.assertTrue(result['extracted'])
        self.assertEqual(result['extraction'].status,'success')


class CLITests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source=self.root/'input.jsonl';self.mock=self.root/'responses.jsonl';self.output=self.root/'out.jsonl'
        self.source.write_text(json.dumps(row())+'\n');self.mock.write_text(json.dumps(raw())+'\n')

    def args(self,mode='examples'):
        return ['--input',str(self.source),'--dataset-mode',mode,'--dataset-revision','test-revision',
                '--mode','mock','--mock-responses',str(self.mock),'--output',str(self.output)]

    def manifest(self):
        return json.loads(Path(str(self.output)+'.manifest.json').read_text())

    def test_success_and_unknown_telemetry(self):
        self.assertEqual(main(self.args()),0)
        manifest=self.manifest()
        self.assertEqual(manifest['status'],'complete')
        self.assertEqual(manifest['request_count'],1)
        self.assertEqual(manifest['input']['sha256'],hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(manifest['unknown_cost_results'],1)
        self.assertIsNone(manifest['total_cost_usd'])
        self.assertTrue(manifest['code']['package_sha256'])

    def test_limit_incomplete_manifest(self):
        self.source.write_text('\n'.join(json.dumps(row(str(i),index=i)) for i in range(3))+'\n')
        self.assertEqual(main(self.args('stream')+['--limit','1']),0)
        m=self.manifest();self.assertEqual((m['planned_turns'],m['selected_turns'],m['processed_turns']),(3,1,1))
        self.assertEqual(m['status'],'limited')

    def test_invalid_later_row_preflight_zero_generation(self):
        self.source.write_text(json.dumps(row())+'\n{}\n')
        with patch('sc_baseline.dataset_runner.MockRuntime') as runtime:
            self.assertEqual(main(self.args()),2)
            runtime.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_overwrite_protection(self):
        self.output.write_text('keep')
        self.assertEqual(main(self.args()),2)
        self.assertEqual(self.output.read_text(),'keep')

    def test_manifest_overwrite_no_inference(self):
        manifest=Path(str(self.output)+'.manifest.json');manifest.write_text('keep')
        with patch('sc_baseline.dataset_runner.run_turns') as run:
            self.assertEqual(main(self.args()),2);run.assert_not_called()
        self.assertEqual(manifest.read_text(),'keep')

    def test_bad_response_failure_manifest(self):
        self.mock.write_text(json.dumps('bad')+'\n')
        self.assertEqual(main(self.args()),1)
        self.assertEqual(self.manifest()['status'],'failed')
        self.assertEqual(json.loads(self.output.read_text())['extraction']['status'],'response_error')

    def test_insufficient_mock_preflight(self):
        self.mock.write_text('')
        self.assertEqual(main(self.args()),2)
        self.assertFalse(self.output.exists())

    def test_injection_cli_full_replay(self):
        clean=self.root/'clean.jsonl'
        clean.write_text(json.dumps(row())+'\n'+json.dumps(row('r2',index=2))+'\n')
        self.source.write_text(json.dumps(injection())+'\n')
        self.mock.write_text(json.dumps(raw('A'))+'\n'+json.dumps(raw())+'\n')
        self.assertEqual(main(self.args('injections')+['--clean-stream',str(clean)]),0)
        self.assertEqual(self.manifest()['planned_turns'],2)
        results=[json.loads(s) for s in self.output.read_text().splitlines()]
        self.assertEqual(len(results[1]['extraction']['registry']),1)

    def test_known_cost_total(self):
        rt=MockRuntime([ModelResponse(raw(),'scripted-mock',{'prompt_tokens':1},0.01,'test')])
        with patch('sc_baseline.dataset_runner.MockRuntime',return_value=rt):
            self.assertEqual(main(self.args()),0)
        self.assertEqual(self.manifest()['total_cost_usd'],0.01)


if __name__=='__main__': unittest.main()
