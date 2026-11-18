from __future__ import annotations

import base64
import importlib
import json
import struct
import zlib
from collections.abc import AsyncIterator

import httpx
import pytest
from starlette.testclient import TestClient

from gptmock.app import create_app
from gptmock.core.settings import Settings


def png_bytes() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))

    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('!2I5B', 1, 1, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(b'\x00\xff\x00\x00'))
            + chunk(b'IEND', b''))


def upstream_image() -> dict:
    return {
        'created': 1795000000,
        'data': [{'b64_json': base64.b64encode(png_bytes()).decode(), 'generation_id': 'gen_test'}],
        'size': '1024x1024', 'quality': 'low', 'output_format': 'png',
        'usage': {'input_tokens': 21, 'output_tokens': 343, 'total_tokens': 364},
    }


@pytest.fixture
def image_client(monkeypatch):
    state = {'requests': [], 'auth_calls': 0, 'body': upstream_image(), 'status': 200}

    async def auth():
        state['auth_calls'] += 1
        return 'synthetic-access-token', 'synthetic-account'

    def transport(request: httpx.Request) -> httpx.Response:
        state['requests'].append(request)
        return httpx.Response(state['status'], json=state['body'])

    async def http_client() -> AsyncIterator[httpx.AsyncClient]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            yield client

    monkeypatch.setattr(importlib.import_module('gptmock.services.images'), 'get_effective_chatgpt_auth', auth)
    app = importlib.import_module('gptmock.app').create_app(Settings())
    dependency = importlib.import_module('gptmock.core.dependencies').get_http_client
    app.dependency_overrides[dependency] = http_client
    with TestClient(app) as client:
        yield client, state


def test_native_image_generation_preserves_usage_and_returns_actual_size(image_client):
    client, state = image_client
    result = client.post('/v1/images/generations', json={
        'model': 'gptmock-image', 'prompt': 'A red circle', 'n': 1,
        'quality': 'low', 'size': '1024x1024', 'response_format': 'b64_json',
        'output_format': 'png', 'stream': False,
    })
    assert result.status_code == 200, result.text
    data = result.json()
    assert data['size'] == '1x1'
    assert data['usage'] == state['body']['usage']
    assert data['data'][0] == state['body']['data'][0]
    assert len(state['requests']) == 1
    request = state['requests'][0]
    assert str(request.url) == 'https://chatgpt.com/backend-api/codex/images/generations'
    assert request.headers['authorization'] == 'Bearer synthetic-access-token'
    assert request.headers['chatgpt-account-id'] == 'synthetic-account'
    payload = json.loads(request.content)
    assert payload['model'] == 'gpt-image-2'
    assert payload['prompt'] == 'A red circle'
    assert payload['size'] == '1024x1024'
    assert 'response_format' not in payload
    assert 'output_format' not in payload
    assert 'stream' not in payload


@pytest.mark.parametrize('overrides', [
    {'model': 'gpt-image-2.6-flare'}, {'model': 'gpt-image-2.6-sunburst'},
    {'prompt': ' '}, {'n': 2}, {'n': True}, {'stream': True},
    {'response_format': 'url'}, {'output_format': 'jpeg'},
    {'quality': 'max'}, {'background': 'transparent'}, {'size': 'nonsense'},
    {'complexity': 'high'}, {'style': 'vivid'}, {'image': 'not-supported'},
])
def test_unsupported_image_options_fail_before_auth(image_client, overrides):
    client, state = image_client
    response = client.post('/v1/images/generations', json={
        'model': 'gptmock-image', 'prompt': 'A circle', **overrides,
    })
    assert response.status_code == 400, response.text
    assert state['auth_calls'] == 0
    assert state['requests'] == []


@pytest.mark.parametrize('body', [
    {'created': 1, 'data': []}, {'created': 1, 'data': [{'b64_json': '!bad!'}]},
    {'created': 1, 'data': [{'b64_json': base64.b64encode(b'not a PNG').decode()}]},
    {'created': 1, 'data': [{'url': 'https://example.com/image.png'}]},
    {'created': 1, 'data': upstream_image()['data'] * 2},
    {'data': upstream_image()['data']},
])
def test_invalid_image_success_payload_is_not_reported_as_success(image_client, body):
    client, state = image_client
    state['body'] = body
    response = client.post('/v1/images/generations', json={'model': 'gptmock-image', 'prompt': 'A circle'})
    assert response.status_code == 502


def test_image_upstream_failure_is_not_retried(image_client):
    client, state = image_client
    state.update(status=429, body={'error': {'message': 'Image limit reached', 'type': 'rate_limit_error'}})
    response = client.post('/v1/images/generations', json={'model': 'gptmock-image', 'prompt': 'A circle'})
    assert response.status_code == 429
    assert 'Image limit reached' in response.text
    assert len(state['requests']) == 1


def test_image_route_uses_existing_proxy_authentication():
    with TestClient(create_app(Settings(api_key='proxy-secret'))) as client:
        response = client.post('/v1/images/generations', json={'model': 'gptmock-image', 'prompt': 'A circle'})
    assert response.status_code == 401


def test_image_payload_cap_fails_without_returning_partial_data(image_client, monkeypatch):
    client, state = image_client
    monkeypatch.setattr(importlib.import_module('gptmock.services.images'), 'MAX_RESPONSE_BYTES', 16)
    response = client.post('/v1/images/generations', json={'model': 'gptmock-image', 'prompt': 'A circle'})
    assert response.status_code == 502
    assert 'b64_json' not in response.text
    assert len(state['requests']) == 1


@pytest.mark.parametrize('route, fields', [
    ('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'Hello'}]}),
    ('/v1/responses', {'input': 'Hello'}),
])
def test_image_alias_cannot_be_used_as_text_model(image_client, route, fields):
    client, state = image_client
    response = client.post(route, json={'model': 'gptmock-image', **fields})
    assert response.status_code == 400
    assert '/v1/images/generations' in response.text
    assert state['requests'] == []


def test_image_discovery_is_separate_from_text_models():
    with TestClient(create_app(Settings(expose_reasoning_models=True))) as client:
        models = client.get('/v1/models').json()['data']
        image = next(m for m in models if m['id'] == 'gptmock-image')
        assert image['mode'] == 'image_generation'
        assert image['model_selection'] == 'subscription_default'
        assert 'reasoning' not in image
        assert not any(m['id'].startswith('gptmock-image-') for m in models)
        assert not any(m['name'] == 'gptmock-image' for m in client.get('/api/tags').json()['models'])
        assert client.post('/api/show', json={'model': 'gptmock-image'}).status_code == 404
