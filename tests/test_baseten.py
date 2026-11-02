"""Offline tests: no live requests or credentials."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

from whatiff.baseten import BasetenClient, BasetenError, BasetenSparkPolicy, MODEL, load_baseten_key
from whatiff.contracts import Action
from whatiff.execution import audited_calls, projection
from whatiff.policy import POLICY_PROMPT
from tests import test_interface as fixtures
FakeAdversary, FakeSam = fixtures.FakeAdversary, fixtures.FakeSam


class BasetenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def client(self, answer=None, finish='stop'):
        client = BasetenClient('secret-test', self.root)
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            'choices': [{'finish_reason': finish, 'message': {'content': json.dumps(answer or {'action': 'PICK_TARGET', 'reason': 'Target visible.'})}}],
            'usage': {'total_tokens': 23}}).encode()
        opener = MagicMock()
        opener.open.return_value = response
        return client, opener

    def test_contract_and_isolated_prompt(self):
        client, opener = self.client()
        with patch('whatiff.baseten.build_opener', return_value=opener):
            decision = BasetenSparkPolicy(client).decide(Path('examples/tabletop-original.png').read_bytes(), 'Pick up the mug.')
        self.assertEqual(decision.action, Action.PICK_TARGET)
        request = json.loads(opener.open.call_args.args[0].data)
        self.assertEqual(request['messages'][0]['content'], POLICY_PROMPT)
        self.assertEqual(request['messages'][1]['content'][0]['text'], 'Pick up the mug.')
        self.assertEqual(len(request['messages']), 2)
        self.assertEqual(decision.raw['settings']['provider'], 'baseten')
        self.assertEqual(decision.raw['settings']['model'], MODEL)
        self.assertNotIn('secret-test', ''.join(p.read_text() for p in self.root.rglob('*.json')))
        self.assertEqual(audited_calls(self.root.parent), [])

    def test_invalid_contracts_fail(self):
        for answer, finish in [({'action':'FLY','reason':'No'},'stop'), ({'action':'ABSTAIN','reason':''},'stop'), ({'action':'ABSTAIN','reason':'No','expected':'ABSTAIN'},'stop'), ({'action':'ABSTAIN','reason':'No'},'length')]:
            client, opener = self.client(answer, finish)
            with patch('whatiff.baseten.build_opener', return_value=opener), self.assertRaisesRegex(BasetenError, 'Baseten: invalid policy response'):
                BasetenSparkPolicy(client).decide(Path('examples/tabletop-original.png').read_bytes(), 'Task')

    def test_network_failure_is_safe_and_recorded(self):
        client, opener = self.client()
        opener.open.side_effect = URLError('secret-test')
        with patch('whatiff.baseten.build_opener', return_value=opener), self.assertRaisesRegex(BasetenError, 'Baseten: network error'):
            client.request('/chat/completions', {'model': MODEL})
        evidence = ''.join(p.read_text() for p in self.root.rglob('*.json'))
        self.assertNotIn('secret-test', evidence)
        self.assertIn('network error', evidence)

    def test_http_failure_identifies_provider_without_response_secrets(self):
        client, opener = self.client()
        opener.open.side_effect = HTTPError('https://inference.baseten.co/v1/chat/completions', 401, 'secret-test', {}, None)
        with patch('whatiff.baseten.build_opener', return_value=opener), self.assertRaisesRegex(BasetenError, 'Baseten: HTTP 401'):
            client.request('/chat/completions', {'model': MODEL})
        evidence = ''.join(p.read_text() for p in self.root.rglob('*.json'))
        self.assertNotIn('secret-test', evidence)
        self.assertIn('HTTP 401', evidence)

    def test_key_is_loaded_from_project_env(self):
        env = self.root / '.env'
        env.write_text('MODEL_API_KEY=other\nBASETEN_API_KEY="baseten-test"\n')
        self.assertEqual(load_baseten_key(env), 'baseten-test')
        env.write_text('MODEL_API_KEY=other\n')
        with self.assertRaises(BasetenError):
            load_baseten_key(env)

    def test_panel_requires_actual_baseten_call(self):
        state={'mode':'live','status':'completed'}
        self.assertNotIn('Muse Spark Open', json.dumps(projection(state,self.root)))
        folder=self.root/'policy-api'/'call';folder.mkdir(parents=True)
        (folder/'request.json').write_text(json.dumps({'mode':'live','provider':'baseten','body':{'model':MODEL}}))
        (folder/'response.json').write_text(json.dumps({'choices':[{'finish_reason':'stop'}]}))
        calls=audited_calls(self.root)
        self.assertEqual(calls[0]['role'],'Policy')
        self.assertEqual(calls[0]['status'],'completed')
        self.assertIn('Muse Spark Open',json.dumps(projection(state,self.root)))


class BasetenWorkflowTests(unittest.TestCase):
    setUp = fixtures.InterfaceTests.setUp
    wait = fixtures.InterfaceTests.wait


    def test_default_failure_never_calls_hosted_policy(self):
        with patch('whatiff.interface.load_key',return_value='meta-test'), patch('whatiff.interface.load_baseten_key',return_value='base-test'), patch('whatiff.interface.SparkPolicy') as hosted, patch('whatiff.interface.BasetenSparkPolicy') as policy:
            policy.return_value.decide.side_effect=BasetenError('Baseten: HTTP 503')
            response=self.client.post('/api/runs',data={'task':'Pick up the white mug.','backend':'controlled'})
            self.wait()
        run=self.service.get(response.json()['id'])
        self.assertEqual(run['status'],'failed')
        self.assertEqual(run['error_provider'],'baseten')
        self.assertIn('Baseten',run['error'])
        hosted.assert_not_called()


    def test_controlled_default_calls_only_baseten_policy(self):
        from whatiff.contracts import Decision
        class Policy:
            def __init__(self,client): pass
            def decide(self,image,task):
                return Decision(Action.PICK_TARGET,'Visible.',{'settings':{'model':MODEL,'provider':'baseten'}})
        with patch('whatiff.interface.load_key',return_value='meta-test'), patch('whatiff.interface.load_baseten_key',return_value='base-test'), patch('whatiff.interface.SparkPolicy') as hosted, patch('whatiff.interface.BasetenSparkPolicy',Policy), patch('whatiff.interface.SparkAdversary',FakeAdversary), patch('whatiff.interface.SamSegmenter',FakeSam):
            response=self.client.post('/api/runs',data={'task':'Pick up the white mug.','backend':'controlled'})
            self.wait()
        run=self.service.get(response.json()['id'])
        self.assertEqual(run['status'],'completed',run.get('error'))
        self.assertEqual(run['cases'][0]['outcome'],'pass')
        report=self.client.get(f"/api/runs/{run['id']}/report").json()
        self.assertEqual(report['policy_provider'],'baseten')
        self.assertEqual(report['policy_model'],MODEL)
        hosted.assert_not_called()

    def test_muse_policy_failure_has_no_fallback(self):
        with patch('whatiff.interface.load_key', return_value='meta-test'), patch('whatiff.interface.load_baseten_key', return_value='base-test'), patch('whatiff.interface.SparkPolicy') as hosted, patch('whatiff.interface.BasetenSparkPolicy') as policy:
            policy.return_value.settings = {'model': MODEL, 'provider': 'baseten'}
            policy.return_value.decide.side_effect = BasetenError('Baseten: HTTP 503')
            response = self.client.post('/api/runs', data={'task': 'Pick up the sponge.', 'backend': 'muse'})
            self.wait()
        run = self.service.get(response.json()['id'])
        self.assertEqual(run['status'], 'failed')
        self.assertEqual(run['error_provider'], 'baseten')
        self.assertIn('Baseten', run['error'])
        hosted.assert_not_called()

    def test_explicit_hosted_policy_never_loads_baseten_key(self):
        with patch('whatiff.interface.load_key', return_value='meta-test'), patch('whatiff.interface.load_baseten_key', side_effect=AssertionError('Unexpected Baseten key access')), patch('whatiff.interface.BasetenSparkPolicy') as baseten, patch('whatiff.interface.SparkPolicy', fixtures.FakePolicy), patch('whatiff.interface.SparkAdversary', FakeAdversary), patch('whatiff.interface.SamSegmenter', FakeSam):
            response = self.client.post('/api/runs', data={'task': 'Pick up the white mug.', 'backend': 'controlled', 'policy_provider': 'meta'})
            self.wait()
        run = self.service.get(response.json()['id'])
        self.assertEqual(run['status'], 'completed', run.get('error'))
        self.assertEqual(run['report']['policy_provider'], 'meta')
        baseten.assert_not_called()
