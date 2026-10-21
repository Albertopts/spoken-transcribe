"""
Turn ElevenLabs / OpenAI failures into (kind, message) for plain-language help.
kind is one of: credits, permissions, key, busy, other.
"""

import openai
from elevenlabs.core.api_error import ApiError

ELEVENLABS_CREDIT_CODES = {"quota_exceeded", "insufficient_credits", "payment_required"}
ELEVENLABS_PERMISSION_CODES = {"missing_permissions", "insufficient_permissions"}
ELEVENLABS_KEY_CODES = {"invalid_api_key", "missing_api_key", "api_key_expired"}
ELEVENLABS_BUSY_CODES = {"rate_limit_exceeded", "concurrent_limit_exceeded",
                         "too_many_concurrent_requests", "system_busy"}


def classify_error(err: Exception) -> tuple[str, str]:
    if isinstance(err, ApiError):
        return _classify_elevenlabs(err)
    if isinstance(err, openai.APIStatusError):
        return _classify_openai(err)
    if isinstance(err, (openai.APITimeoutError, openai.APIConnectionError)):
        return "busy", str(err)
    return "other", str(err)


def _classify_elevenlabs(err: ApiError) -> tuple[str, str]:
    """Handles both shapes: {"detail": {"status"|"code", "message"}} and a flat body."""
    body = err.body
    detail = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(detail, dict):
        code = str(detail.get("code") or detail.get("status") or detail.get("error") or "").lower()
        message = str(detail.get("message") or "")
    else:
        code, message = "", str(detail or "")
    text = f"{code} {message}".lower()
    status = err.status_code

    if code in ELEVENLABS_CREDIT_CODES or status == 402 or "quota" in text or "credits remaining" in text:
        kind = "credits"
    elif code in ELEVENLABS_PERMISSION_CODES or "permission" in text:
        kind = "permissions"
    elif code in ELEVENLABS_KEY_CODES or (status == 401 and "key" in text):
        kind = "key"
    elif code in ELEVENLABS_BUSY_CODES or status == 429:
        kind = "busy"
    else:
        kind = "other"
    return kind, message or f"HTTP {status}: {body}"


def _classify_openai(err: "openai.APIStatusError") -> tuple[str, str]:
    """OpenAI body: {"error": {"message", "type", "code"}} (the SDK may pass the inner dict)."""
    body = err.body if isinstance(err.body, dict) else {}
    inner = body.get("error", body) if isinstance(body.get("error", body), dict) else body
    code = str(inner.get("code") or inner.get("type") or "").lower()
    message = str(inner.get("message") or err.message or "")
    text = f"{code} {message}".lower()
    status = err.status_code

    if "insufficient_quota" in text or "billing" in text or "quota" in text:
        kind = "credits"
    elif status == 401 or "invalid_api_key" in text:
        kind = "key"
    elif status in (403, 404) or "permission" in text or "model_not_found" in text:
        kind = "permissions"
    elif status == 429:
        kind = "busy"
    else:
        kind = "other"
    return kind, message
