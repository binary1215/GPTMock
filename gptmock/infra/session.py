from __future__ import annotations

import hashlib
import json
import threading
import uuid
from typing import Any

_LOCK = threading.Lock()
_FINGERPRINT_TO_UUID: dict[str, str] = {}
_ORDER: list[str] = []
_MAX_ENTRIES = 10000


def resolve_client_session_id(payload: dict[str, Any], client_supplied: str | None = None) -> str | None:
    """Prefer the transport session header, then an explicit body session ID."""
    if isinstance(client_supplied, str) and client_supplied.strip():
        return client_supplied.strip()
    metadata = payload.get("client_metadata")
    if isinstance(metadata, dict):
        session_id = metadata.get("session_id")
        if isinstance(session_id, str) and session_id.strip():
            return session_id.strip()
    return None


def resolve_prompt_cache_key(payload: dict[str, Any], session_id: str) -> str:
    """Keep a caller's cache key independent from routing/session identity."""
    key = payload.get("prompt_cache_key")
    if key is not None and not isinstance(key, str):
        raise ValueError("prompt_cache_key must be a string")
    return key if isinstance(key, str) and key.strip() else session_id


def _canonicalize_first_user_message(input_items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Extract the first stable user message from Responses input items. Good use for a fingerprint for prompt caching.
    """
    for item in input_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        role = item.get("role")
        if role != "user":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        norm_content = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "input_text":
                text = part.get("text") if isinstance(part.get("text"), str) else ""
                if text:
                    norm_content.append({"type": "input_text", "text": text})
            elif ptype == "input_image":
                url = part.get("image_url") if isinstance(part.get("image_url"), str) else None
                if url:
                    norm_content.append({"type": "input_image", "image_url": url})
        if norm_content:
            return {"type": "message", "role": "user", "content": norm_content}
    return None


def canonicalize_prefix(instructions: str | None, input_items: list[dict[str, Any]]) -> str:
    prefix: dict[str, Any] = {}
    if isinstance(instructions, str) and instructions.strip():
        prefix["instructions"] = instructions.strip()
    first_user = _canonicalize_first_user_message(input_items)
    if first_user is not None:
        prefix["first_user_message"] = first_user
    return json.dumps(prefix, sort_keys=True, separators=(",", ":"))


def _fingerprint(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _remember(fp: str, sid: str) -> None:
    if fp in _FINGERPRINT_TO_UUID:
        return
    _FINGERPRINT_TO_UUID[fp] = sid
    _ORDER.append(fp)
    if len(_ORDER) > _MAX_ENTRIES:
        oldest = _ORDER.pop(0)
        _FINGERPRINT_TO_UUID.pop(oldest, None)


def ensure_session_id(
    instructions: str | None,
    input_items: list[dict[str, Any]],
    client_supplied: str | None = None,
) -> str:
    if isinstance(client_supplied, str) and client_supplied.strip():
        return client_supplied.strip()

    canon = canonicalize_prefix(instructions, input_items)
    fp = _fingerprint(canon)
    with _LOCK:
        if fp in _FINGERPRINT_TO_UUID:
            return _FINGERPRINT_TO_UUID[fp]
        sid = str(uuid.uuid4())
        _remember(fp, sid)
        return sid
