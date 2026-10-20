# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Scripted-model tests of the real resolver, agent, dispatcher and summarizer.

Import-only ML adapters are isolated; no weights, GPU or model key is needed.
Run: python -m unittest discover -s tests -p test_web_search.py -v
"""
import importlib.util
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from llm4crs.web_search import (
    ParallelSearchClient, ParallelWebTool, SEARCH_MCP_URL, USER_AGENT,
    WebSearchError, new_search_session, resolve_web_tools,
)

SEARCH = {'objective': 'Find official game update notes', 'search_queries': ['game official update notes']}
FETCH = {'urls': ['https://example.com/game-update'], 'objective': 'Read update notes'}
EVIDENCE = 'Official update: feature released. Source: https://example.com/game-update'


class ScriptedModel:
    def __init__(self, **kwargs):
        self.plan, self.summaries = [], []

    def call(self, user_prompt, sys_prompt=None, **kwargs):
        if sys_prompt is not None:
            self.summaries.append(user_prompt)
            return user_prompt.split('Execution Result: \n ', 1)[1].split('\n\nPlease use', 1)[0]
        return 'Action: ToolExecutor\nAction Input: ' + json.dumps(self.plan)


@contextmanager
def scripted_token_usage():
    yield types.SimpleNamespace(get=lambda: {})


def load_runtime():
    # Only unrelated model/ML imports are replaced. The runtime file, prompts,
    # web client, resolver, parser and result consumption all run unchanged.
    adapters = {
        'langchain.agents': {'Tool': object},
        'loguru': {'logger': types.SimpleNamespace(debug=lambda *a, **k: None)},
        'llm4crs.critic': {'Critic': object},
        'llm4crs.demo.base': {'DemoSelector': object},
        'llm4crs.utils': {'OpenAICall': ScriptedModel, 'num_tokens_from_string': len, 'format_prompt': str},
        'llm4crs.utils.open_ai': {'get_openai_tokens': scripted_token_usage},
        'llm4crs.memory.memory': {'UserProfileMemory': object},
    }
    modules = {}
    for name, values in adapters.items():
        modules[name] = types.ModuleType(name)
        modules[name].__dict__.update(values)
    spec = importlib.util.spec_from_file_location(
        '_web_test_runtime', Path(__file__).resolve().parents[1] / 'llm4crs/agent_plan_first_openai.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


runtime = load_runtime()


class CatalogTool:
    def __init__(self, name):
        self.name, self.desc, self.calls = name, 'Catalog fixture', []

    def run(self, value):
        self.calls.append(value)
        return 'Catalog item result'


class Catalog:
    def info(self, query=None):
        return 'title, description'


class Buffer:
    track_info = ''
    def clear(self):
        pass
    def clear_tracks(self):
        pass


def make_agent(client=None, **kwargs):
    from llm4crs.prompt.tool import TOOL_NAMES
    tools = {key: CatalogTool(name.format(item='game', Item='Game', ITEM='GAME'))
             for key, name in TOOL_NAMES.items()}
    agent = runtime.CRSAgentPlanFirstOpenAI('game', tools, Buffer(), Catalog(),
        web_search_client=client, shorten_strategy='cut', **kwargs)
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'scripted-model-only'}, clear=True):
        agent.init_agent()
    return agent, tools


def plan(tool, value):
    return {'tool_name': tool, 'input': json.dumps(value)}


class Response(io.BytesIO):
    def __init__(self, data, content_type='application/json'):
        super().__init__(data)
        self.headers = {'Content-Type': content_type, 'Mcp-Session-Id': 'transport-only'}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()


class RecordingServer:
    def __init__(self, sse=False):
        self.requests, self.sse, self.error, self.tool_error = [], sse, None, False

    def open(self, request, timeout):
        payload = json.loads(request.data)
        self.requests.append((request, timeout, payload))
        method = payload['method']
        if method == 'notifications/initialized':
            return Response(b'')
        if method == 'initialize':
            result = {'protocolVersion': '2025-03-26', 'capabilities': {'tools': {}},
                      'serverInfo': {'name': 'fixture', 'version': '1'}}
        elif method == 'tools/list':
            result = {'tools': [{'name': n, 'inputSchema': {'type': 'object'}} for n in ('web_search', 'web_fetch')]}
        else:
            if self.error:
                raise self.error
            result = {'content': [{'type': 'text', 'text': EVIDENCE}], 'isError': self.tool_error}
        message = json.dumps({'jsonrpc': '2.0', 'id': payload['id'], 'result': result}).encode()
        if self.sse:
            return Response(b': keepalive\n\nevent: message\ndata: ' + message + b'\n\n', 'text/event-stream')
        return Response(message)


def fixture_client(sse=False):
    with patch.dict(os.environ, {}, clear=True):
        client = ParallelSearchClient(timeout=7)
    server = RecordingServer(sse)
    client._opener = server
    return client, server


def calls(server):
    return [p['params'] for _, _, p in server.requests if p['method'] == 'tools/call']


class WebRuntimeTests(unittest.TestCase):
    def test_fresh_no_key_default_uses_production_resolver(self):
        server = RecordingServer()
        with patch.dict(os.environ, {}, clear=True), patch('llm4crs.web_search.build_opener', return_value=server):
            agent, _ = make_agent()
            agent.agent.plan = [plan('Game Web Search Tool', SEARCH)]
            self.assertIn(EVIDENCE, agent.run({'input': 'current update'}))
        self.assertTrue(all(r.get_header('Authorization') is None for r, _, _ in server.requests))

    def test_default_agent_consumes_search_fetch_without_mapping(self):
        client, server = fixture_client()
        agent, tools = make_agent(client)
        agent.agent.plan = [plan('Game Web Search Tool', SEARCH), plan('Game Web Fetch Tool', FETCH)]
        self.assertIn(EVIDENCE, agent.run({'input': 'What changed in the game?'}))
        self.assertEqual(len(server.requests), 5)
        self.assertEqual(tools['MapTool'].calls, [])
        self.assertIn('source URLs', agent.agent.summaries[0])
        self.assertEqual([p['name'] for p in calls(server)], ['web_search', 'web_fetch'])
        self.assertEqual(len({p['arguments']['session_id'] for p in calls(server)}), 1)
        for request, timeout, payload in server.requests:
            self.assertEqual(request.full_url, SEARCH_MCP_URL)
            self.assertEqual(request.get_header('User-agent'), USER_AGENT)
            self.assertIsNone(request.get_header('Authorization'))
            self.assertEqual(timeout, 7)
            if payload['method'] != 'initialize':
                self.assertEqual(request.get_header('Mcp-session-id'), 'transport-only')
                self.assertEqual(request.get_header('Mcp-protocol-version'), '2025-03-26')

    def test_followup_concurrent_conversations_and_reset(self):
        client, server = fixture_client()
        first, _ = make_agent(client)
        second, _ = make_agent(client)
        for agent in (first, second):
            agent.agent.plan = [plan('Game Web Search Tool', SEARCH)]
        first.run({'input': 'update'})
        first.agent.plan = [plan('Game Web Fetch Tool', FETCH)]
        first.run({'input': 'read more'})
        first.agent.plan = [plan('Game Web Search Tool', SEARCH)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            answers = list(pool.map(lambda a: a.run({'input': 'followup'}), (first, second)))
        self.assertTrue(all(EVIDENCE in answer for answer in answers))
        identities = [p['arguments']['session_id'] for p in calls(server)]
        self.assertEqual(identities[0], identities[1])
        self.assertEqual(identities.count(first._search_session_id), 3)
        self.assertEqual(identities.count(second._search_session_id), 1)
        self.assertNotEqual(first._search_session_id, second._search_session_id)
        old = first._search_session_id
        first.clear()
        first.run({'input': 'new conversation'})
        self.assertNotEqual(old, first._search_session_id)
        self.assertEqual(calls(server)[-1]['arguments']['session_id'], first._search_session_id)

    def test_browser_conversation_context_is_explicit(self):
        client, server = fixture_client()
        agent, _ = make_agent(client)
        agent.agent.plan = [plan('Game Web Search Tool', SEARCH)]
        first, second = new_search_session(), new_search_session()
        for identity in (first, second, first):
            response, _ = agent.run_gr([['updates', None]], identity)
            self.assertIn(EVIDENCE, response[-1][1])
        self.assertEqual([p['arguments']['session_id'] for p in calls(server)], [first, second, first])

    def test_standalone_evaluation_runs_use_new_identity(self):
        client, server = fixture_client()
        agent, _ = make_agent(client)
        agent.agent.plan = [plan('Game Web Search Tool', SEARCH)]
        for _ in range(2):
            agent.run({'input': 'updates'}, chat_history='Independent evaluation case')
        self.assertNotEqual(calls(server)[0]['arguments']['session_id'], calls(server)[1]['arguments']['session_id'])

    def test_disabled_preserves_catalog_and_blocks_web(self):
        client, server = fixture_client()
        agent, tools = make_agent(client, web_search='disabled')
        self.assertEqual(len(agent._tools), 6)
        agent.agent.plan = [plan('Game Information Look Up Tool', 'SELECT title FROM games')]
        self.assertIn('Catalog item result', agent.run({'input': 'catalog item'}))
        agent.agent.plan = [plan('Game Web Search Tool', SEARCH)]
        self.assertIn('Something went wrong', agent.run({'input': 'web'}))
        self.assertEqual(server.requests, [])
        self.assertEqual(len(tools['LookUpTool'].calls), 1)

    def test_explicit_tools_and_unknown_provider(self):
        custom = CatalogTool('Custom Web Search Tool')
        supplied = {'WebSearchTool': custom}
        self.assertIs(resolve_web_tools(supplied, 'game')['WebSearchTool'], custom)
        self.assertEqual(set(supplied), {'WebSearchTool'})
        self.assertNotIn('WebSearchTool', resolve_web_tools(supplied, 'game', 'disabled'))
        with self.assertRaises(ValueError):
            resolve_web_tools({}, 'game', 'typo')

    def test_timeout_auth_quota_and_retry_preserve_identity(self):
        for error in (TimeoutError(), HTTPError(SEARCH_MCP_URL, 401, 'Denied', {}, None),
                      HTTPError(SEARCH_MCP_URL, 429, 'Quota', {}, None)):
            with self.subTest(error=type(error).__name__):
                client, server = fixture_client()
                agent, _ = make_agent(client)
                agent.agent.plan = [plan('Game Web Search Tool', SEARCH)]
                identity = agent._search_session_id
                server.error = error
                self.assertIn('Something went wrong', agent.run({'input': 'update'}))
                self.assertEqual(agent.agent.summaries, [])
                self.assertEqual(identity, agent._search_session_id)
                server.error = None
                self.assertIn(EVIDENCE, agent.run({'input': 'retry'}))
                self.assertEqual(calls(server)[-1]['arguments']['session_id'], identity)

    def test_cancellation_propagates(self):
        client, server = fixture_client()
        server.error = KeyboardInterrupt()
        agent, _ = make_agent(client)
        agent.agent.plan = [plan('Game Web Search Tool', SEARCH)]
        with self.assertRaises(KeyboardInterrupt):
            agent.run({'input': 'update'})
        identity = agent._search_session_id
        server.error = None
        self.assertIn(EVIDENCE, agent.run({'input': 'retry'}))
        self.assertEqual(calls(server)[-1]['arguments']['session_id'], identity)

    def test_sse_tool_error_and_explicit_credentials(self):
        client, server = fixture_client(sse=True)
        client.api_key = 'explicit-test-key'
        self.assertIn(EVIDENCE, ParallelWebTool('game', client).run(json.dumps(SEARCH), 'conversation'))
        self.assertTrue(all(r.get_header('Authorization') == 'Bearer explicit-test-key' for r, _, _ in server.requests))
        server.tool_error = True
        with self.assertRaises(WebSearchError):
            ParallelWebTool('game', client).run(json.dumps(SEARCH), 'conversation')

    def test_invalid_input_makes_no_requests(self):
        client, server = fixture_client()
        search, fetch = ParallelWebTool('game', client), ParallelWebTool('game', client, fetch=True)
        for tool, value in ((search, {'objective': 'x', 'search_queries': []}),
                            (search, {**SEARCH, 'session_id': 'model-controlled'}),
                            (fetch, {'urls': ['file:///private']}),
                            (fetch, {'urls': ['https://user:secret@example.com']}),
                            (fetch, {'urls': ['https://example.com'], 'objective': 'x' * 201})):
            with self.subTest(value=value), self.assertRaises(ValueError):
                tool.run(json.dumps(value), 'conversation')
        self.assertEqual(server.requests, [])

    def test_malformed_response_and_redirect(self):
        client = ParallelSearchClient(api_key='secret')
        class BrokenServer:
            def open(self, request, timeout):
                return Response(b'{bad json')
        client._opener = BrokenServer()
        with self.assertRaises(WebSearchError):
            client.call('web_search', SEARCH, 'conversation')
        from llm4crs.web_search import _NoRedirect
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, '', {}, 'https://other.test'))


if __name__ == '__main__':
    unittest.main()
