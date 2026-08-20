"""Small HMAC capability-token primitives for sandbox services."""

from __future__ import annotations

import base64
import hashlib
import heapq
import hmac
import json
import threading
import time
from dataclasses import dataclass
from typing import Any

from open_deep_research.sandbox.wire import TaskTokenClaimsV1


def decode_root_key(encoded: str) -> bytes:
    """Decode and validate the administrator root signing key."""
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("sandbox_root_signing_key_invalid_base64") from exc
    if len(raw) < 32:
        raise ValueError("sandbox_root_signing_key_too_short")
    return raw


def _hkdf_expand(root: bytes, context: bytes, length: int = 32) -> bytes:
    """Derive a context-separated key with HKDF-SHA256."""
    prk = hmac.new(b"insightforge-sandbox-v1", root, hashlib.sha256).digest()
    output = b""
    previous = b""
    counter = 1
    while len(output) < length:
        previous = hmac.new(
            prk,
            previous + context + bytes([counter]),
            hashlib.sha256,
        ).digest()
        output += previous
        counter += 1
    return output[:length]


@dataclass(frozen=True, slots=True)
class SandboxDerivedKeys:
    """Purpose-separated sandbox service keys."""

    task_token: bytes
    policy_signature: bytes
    service_auth: bytes

    @classmethod
    def from_root(cls, encoded_root: str) -> SandboxDerivedKeys:
        """Derive purpose-separated keys from one validated root."""
        root = decode_root_key(encoded_root)
        return cls(
            task_token=_hkdf_expand(root, b"task-token"),
            policy_signature=_hkdf_expand(root, b"policy-signature"),
            service_auth=_hkdf_expand(root, b"service-auth"),
        )


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sign_payload(payload: dict[str, Any], key: bytes) -> str:
    """Return a URL-safe HMAC signature for canonical JSON."""
    return base64.urlsafe_b64encode(hmac.new(key, _canonical(payload), hashlib.sha256).digest()).decode().rstrip("=")


def verify_payload(payload: dict[str, Any], signature: str, key: bytes) -> bool:
    """Verify a canonical JSON signature without timing leakage."""
    return hmac.compare_digest(sign_payload(payload, key), signature)


def validate_timestamp(timestamp: float, *, max_skew_seconds: int = 30) -> None:
    """Reject stale or future RPC timestamps."""
    if abs(time.time() - float(timestamp)) > max_skew_seconds:
        raise ValueError("sandbox_request_timestamp_out_of_range")


def encode_task_token(claims: TaskTokenClaimsV1, key: bytes) -> str:
    """Encode signed, URL-safe task capability claims."""
    payload = claims.model_dump(mode="json")
    body = base64.urlsafe_b64encode(_canonical(payload)).decode().rstrip("=")
    return f"{body}.{sign_payload(payload, key)}"


def decode_task_token(token: str, key: bytes, *, now: float | None = None) -> TaskTokenClaimsV1:
    """Verify and decode one task capability token."""
    try:
        body, signature = token.split(".", 1)
        raw = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
        payload = json.loads(raw)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("sandbox_task_token_malformed") from exc
    if not isinstance(payload, dict) or not verify_payload(payload, signature, key):
        raise ValueError("sandbox_task_token_invalid_signature")
    claims = TaskTokenClaimsV1.model_validate(payload)
    current = time.time() if now is None else now
    if claims.expires_at <= current:
        raise ValueError("sandbox_task_token_expired")
    return claims


class NonceReplayCache:
    """In-memory TTL replay protection scoped by task-token jti."""

    def __init__(self) -> None:
        """Create an empty in-memory replay cache."""
        self._seen: dict[tuple[str, str], float] = {}
        self._expirations: list[tuple[float, tuple[str, str]]] = []
        self._lock = threading.Lock()

    def consume(self, jti: str, nonce: str, *, expires_at: float, now: float | None = None) -> None:
        """Consume a nonce once and retain it until the token expires."""
        if len(nonce) < 16 or len(nonce) > 256:
            raise ValueError("sandbox_nonce_invalid")
        current = time.time() if now is None else now
        with self._lock:
            while self._expirations and self._expirations[0][0] <= current:
                expiry, expired_key = heapq.heappop(self._expirations)
                if self._seen.get(expired_key) == expiry:
                    self._seen.pop(expired_key, None)
            key = (jti, nonce)
            if key in self._seen:
                raise ValueError("sandbox_nonce_replayed")
            self._seen[key] = expires_at
            heapq.heappush(self._expirations, (expires_at, key))
