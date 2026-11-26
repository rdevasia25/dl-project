"""Dependency-free DeepInfra adapter and deterministic scripted mock runtime."""
import hashlib
import json
import math
import os
import socket
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPRedirectHandler
from .types import ModelResponse

MODEL = 'Qwen/Qwen3.5-9B'
ENDPOINT = 'https://api.deepinfra.com/v1/openai/chat/completions'
# Explicit hosted adaptation; provider enforcement remains a Stage 7 check.
SETTINGS = {'max_tokens': 4096, 'temperature': 0.0, 'top_p': 1.0,
            'stream': False, 'n': 1, 'reasoning_effort': 'none',
            'chat_template_kwargs': {'enable_thinking': False}}


def configuration_id(provider, model):
    sources = ''.join(Path(__file__).with_name(name).read_text()
                      for name in ('prompts.py', 'parsing.py', 'core.py'))
    config = json.dumps({'provider': provider, 'model': model, 'settings': SETTINGS,
                         'endpoint': ENDPOINT if provider == 'deepinfra' else None,
                         'sources': sources}, sort_keys=True)
    return hashlib.sha256(config.encode()).hexdigest()


class RuntimeFailure(Exception):
    def __init__(self, code, safe_message, *, attempted=True):
        super().__init__(safe_message)
        self.code, self.safe_message, self.attempted = code, safe_message, attempted


class MockRuntime:
    provider = 'mock'
    model = 'scripted-mock'

    def __init__(self, responses):
        self.responses = iter(responses)
        self.config_id = configuration_id(self.provider, self.model)
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        try:
            response = next(self.responses)
        except StopIteration:
            raise RuntimeFailure('mock_exhausted', 'No scripted response remains.', attempted=False) from None
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, ModelResponse) else ModelResponse(response, self.model)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


class DeepInfraRuntime:
    provider = 'deepinfra'
    model = MODEL

    def __init__(self, *, api_key=None, timeout=60.0, opener=None):
        if not _number(timeout) or timeout == 0:
            raise ValueError('timeout must be positive and finite.')
        self._api_key = api_key if api_key is not None else os.environ.get('DEEPINFRA_API_KEY')
        self.timeout = timeout
        self._opener = opener or build_opener(_NoRedirect())
        self.config_id = configuration_id(self.provider, self.model)

    def generate(self, messages):
        if not isinstance(self._api_key, str) or not self._api_key.strip():
            raise RuntimeFailure('missing_api_key', 'Set DEEPINFRA_API_KEY.', attempted=False)
        payload = {'model': self.model, 'messages': messages, **SETTINGS}
        try:
            request = Request(ENDPOINT, data=json.dumps(payload).encode(), headers={
                'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self._api_key}, method='POST')
        except (ValueError, UnicodeError):
            raise RuntimeFailure('invalid_configuration', 'Invalid request configuration.', attempted=False) from None
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                body = response.read()
        except HTTPError as exc:
            raise RuntimeFailure('http_error', f'Provider HTTP status {exc.code}.') from None
        except (TimeoutError, socket.timeout):
            raise RuntimeFailure('timeout', 'Provider request timed out.') from None
        except (URLError, OSError, ValueError):
            raise RuntimeFailure('transport_error', 'Provider transport failed.') from None
        try:
            data = json.loads(body)
            if not isinstance(data, dict) or 'error' in data:
                raise ValueError
            choice = data['choices'][0]
            text = choice['message']['content']
            model = data['model']
            if not isinstance(text, str) or not isinstance(model, str):
                raise ValueError
            # Detect an observed violation of the requested thinking switch.
            if choice['message'].get('reasoning_content') or '<think>' in text:
                raise RuntimeFailure('thinking_enabled', 'Provider returned reasoning despite disabled thinking.')
            usage = data.get('usage')
            if usage is not None and not isinstance(usage, dict):
                raise ValueError
            cost = usage.get('estimated_cost') if usage else None
            source = 'provider:usage.estimated_cost' if _number(cost) else None
            if source is None:
                cost = None
            # Do not invent usage or rely on changing list prices when absent.
            return ModelResponse(text, model, usage, cost, source, choice.get('finish_reason'))
        except (ValueError, KeyError, IndexError, TypeError, UnicodeError):
            raise RuntimeFailure('provider_schema', 'Provider returned an invalid response envelope.') from None
