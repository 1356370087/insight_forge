"""Low-level provider error inspection shared by tools and model routing."""

from __future__ import annotations


def is_token_limit_exceeded(
    exception: Exception,
    model_name: str | None = None,
) -> bool:
    """Return whether an exception indicates a provider context-limit error."""
    error_str = str(exception).lower()
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith("openai:"):
            provider = "openai"
        elif model_str.startswith("anthropic:"):
            provider = "anthropic"
        elif model_str.startswith(("gemini:", "google:")):
            provider = "gemini"

    if provider == "openai":
        return _check_openai_token_limit(exception, error_str)
    if provider == "anthropic":
        return _check_anthropic_token_limit(exception, error_str)
    if provider == "gemini":
        return _check_gemini_token_limit(exception, error_str)
    return (
        _check_openai_token_limit(exception, error_str)
        or _check_anthropic_token_limit(exception, error_str)
        or _check_gemini_token_limit(exception, error_str)
    )


def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, "__module__", "")
    is_openai_exception = (
        "openai" in exception_type.lower() or "openai" in module_name.lower()
    )
    if is_openai_exception and class_name in {"BadRequestError", "InvalidRequestError"}:
        if any(
            keyword in error_str
            for keyword in ("token", "context", "length", "maximum context", "reduce")
        ):
            return True
    return (
        getattr(exception, "code", "") == "context_length_exceeded"
        or getattr(exception, "type", "") == "invalid_request_error"
    )


def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, "__module__", "")
    is_anthropic_exception = (
        "anthropic" in exception_type.lower() or "anthropic" in module_name.lower()
    )
    return (
        is_anthropic_exception
        and class_name == "BadRequestError"
        and "prompt is too long" in error_str
    )


def _check_gemini_token_limit(exception: Exception, _error_str: str) -> bool:
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, "__module__", "")
    is_google_exception = (
        "google" in exception_type.lower() or "google" in module_name.lower()
    )
    return (
        is_google_exception
        and class_name in {"ResourceExhausted", "GoogleGenerativeAIFetchError"}
    ) or "google.api_core.exceptions.resourceexhausted" in exception_type.lower()
