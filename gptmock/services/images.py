"""Bounded image generation through the native ChatGPT Codex Images API."""
from __future__ import annotations

import base64
import binascii
import json
import re
import struct
import uuid
from typing import Any

import httpx

from gptmock.infra.auth import get_effective_chatgpt_auth
from gptmock.infra.limits import record_rate_limits_from_response
from gptmock.services.chat import ChatCompletionError

IMAGE_MODEL_ALIAS = "gptmock-image"
NATIVE_IMAGE_MODEL = "gpt-image-2"
IMAGE_GENERATIONS_URL = "https://chatgpt.com/backend-api/codex/images/generations"
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_RESPONSE_BYTES = ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 65536
_ALLOWED_FIELDS = {
    "model", "prompt", "n", "stream", "response_format", "quality", "size", "background", "output_format",
}


def _invalid(message: str, param: str) -> ChatCompletionError:
    return ChatCompletionError(
        message, status_code=400,
        error_data={"error": {"message": message, "type": "invalid_request_error", "param": param}},
    )


def _native_payload(payload: dict[str, Any]) -> dict[str, Any]:
    unknown = set(payload) - _ALLOWED_FIELDS
    if unknown:
        field = sorted(unknown)[0]
        raise _invalid(f"Unsupported image generation parameter: {field}", field)
    if payload.get("model") != IMAGE_MODEL_ALIAS:
        raise _invalid(f"Image generation requires model '{IMAGE_MODEL_ALIAS}'", "model")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32000:
        raise _invalid("Image prompt must be nonblank and at most 32000 characters", "prompt")
    n = payload.get("n", 1)
    if type(n) is not int or n != 1:
        raise _invalid("Only n=1 is supported", "n")
    if payload.get("stream", False) is not False:
        raise _invalid("Image generation streaming is not supported", "stream")
    if payload.get("response_format", "b64_json") != "b64_json":
        raise _invalid("Only response_format='b64_json' is supported", "response_format")
    if payload.get("output_format") not in (None, "png"):
        raise _invalid("Only PNG output is supported", "output_format")

    native: dict[str, Any] = {"model": NATIVE_IMAGE_MODEL, "prompt": prompt, "n": 1}
    for field, supported in (
        ("quality", ("auto", "low", "medium", "high")),
        ("background", ("auto", "opaque")),
    ):
        value = payload.get(field)
        if value is not None:
            if value not in supported:
                raise _invalid(f"Unsupported image {field}", field)
            native[field] = value
    size = payload.get("size")
    if size is not None:
        if not isinstance(size, str) or (size != "auto" and not re.fullmatch(r"[1-9][0-9]{0,4}x[1-9][0-9]{0,4}", size)):
            raise _invalid("Image size must be 'auto' or WIDTHxHEIGHT", "size")
        native["size"] = size
    # Codex omits output_format for native PNG generation. Do not send API-only options.
    return native


def _checked_image_response(data: Any) -> dict[str, Any]:
    def malformed() -> ChatCompletionError:
        return ChatCompletionError("Upstream returned an invalid PNG image response", status_code=502)

    if not isinstance(data, dict) or type(data.get("created")) is not int:
        raise malformed()
    images = data.get("data")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise malformed()
    encoded = images[0].get("b64_json")
    if not isinstance(encoded, str) or len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise malformed()
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise malformed() from None
    if (
        len(raw) < 45 or len(raw) > MAX_IMAGE_BYTES
        or raw[:8] != b"\x89PNG\r\n\x1a\n"
        or raw[8:16] != b"\x00\x00\x00\rIHDR"
        or raw[-12:] != b"\x00\x00\x00\x00IEND\xaeB`\x82"
    ):
        raise malformed()
    if binascii.crc32(raw[12:29]) & 0xFFFFFFFF != struct.unpack(">I", raw[29:33])[0]:
        raise malformed()
    width, height = struct.unpack(">II", raw[16:24])
    if not width or not height:
        raise malformed()
    # Preserve native metadata, including usage and generation IDs, without inventing image-model identity.
    result = dict(data)
    result["size"] = f"{width}x{height}"
    return result


async def process_image_generation(payload: dict[str, Any], http_client: httpx.AsyncClient) -> dict[str, Any]:
    """Validate one PNG request and preserve the native image response metadata."""
    native = _native_payload(payload)
    access_token, account_id = await get_effective_chatgpt_auth()
    if not access_token or not account_id:
        raise ChatCompletionError("Missing ChatGPT credentials. Run 'gptmock login' first.", status_code=401)
    headers = {
        "Authorization": f"Bearer {access_token}",
        "chatgpt-account-id": account_id,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-codex-image-turn-id": str(uuid.uuid4()),
    }
    try:
        request = http_client.build_request("POST", IMAGE_GENERATIONS_URL, headers=headers, json=native, timeout=600.0)
        response = await http_client.send(request, stream=True)
        try:
            record_rate_limits_from_response(response)
            maximum = MAX_RESPONSE_BYTES if response.is_success else 65536
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > maximum:
                    if not response.is_success:
                        break
                    raise ChatCompletionError("Upstream image response exceeds the supported size", status_code=502)
                body.extend(chunk)
            if not response.is_success:
                message = f"Upstream image generation failed (HTTP {response.status_code})"
                try:
                    error_body = json.loads(body)
                    error = error_body.get("error") if isinstance(error_body, dict) else None
                    detail = error.get("message") if isinstance(error, dict) else error
                    if isinstance(detail, str) and detail:
                        message = detail[:2000]
                except (ValueError, UnicodeDecodeError):
                    pass
                raise ChatCompletionError(message, status_code=response.status_code)
            try:
                decoded = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                raise ChatCompletionError("Upstream returned invalid image JSON", status_code=502) from None
            return _checked_image_response(decoded)
        finally:
            await response.aclose()
    except httpx.RequestError:
        raise ChatCompletionError("Upstream image generation request failed", status_code=502) from None
