"""Model context-limit lookup utilities."""

# Keep exact keys ahead of the longest-substring fallback in
# :func:`get_model_token_limit`; this table is intentionally maintained here.
MODEL_TOKEN_LIMITS: dict[str, int] = {
    "openai:deepseek-v4-flash[1m]": 1_000_000,
    "openai:deepseek-v4-pro[1m]": 1_000_000,
    "openai:deepseek-v4-flash": 200_000,
    "openai:deepseek-v4-pro": 200_000,
    "openai:gpt-4.1-mini": 1_047_576,
    "openai:gpt-4.1-nano": 1_047_576,
    "openai:gpt-4.1": 1_047_576,
    "openai:gpt-4o-mini": 128_000,
    "openai:gpt-4o": 128_000,
    "openai:o4-mini": 200_000,
    "openai:o3-mini": 200_000,
    "openai:o3": 200_000,
    "openai:o3-pro": 200_000,
    "openai:o1": 200_000,
    "openai:o1-pro": 200_000,
    "anthropic:claude-opus-4": 200_000,
    "anthropic:claude-sonnet-4": 200_000,
    "anthropic:claude-3-7-sonnet": 200_000,
    "anthropic:claude-3-5-sonnet": 200_000,
    "anthropic:claude-3-5-haiku": 200_000,
    "google:gemini-1.5-pro": 2_097_152,
    "google:gemini-1.5-flash": 1_048_576,
    "google:gemini-pro": 32_768,
    "cohere:command-r-plus": 128_000,
    "cohere:command-r": 128_000,
    "cohere:command-light": 4_096,
    "cohere:command": 4_096,
    "mistral:mistral-large": 32_768,
    "mistral:mistral-medium": 32_768,
    "mistral:mistral-small": 32_768,
    "mistral:mistral-7b-instruct": 32_768,
    "ollama:codellama": 16_384,
    "ollama:llama2:70b": 4_096,
    "ollama:llama2:13b": 4_096,
    "ollama:llama2": 4_096,
    "ollama:mistral": 32_768,
    "bedrock:us.amazon.nova-premier-v1:0": 1_000_000,
    "bedrock:us.amazon.nova-pro-v1:0": 300_000,
    "bedrock:us.amazon.nova-lite-v1:0": 300_000,
    "bedrock:us.amazon.nova-micro-v1:0": 128_000,
    "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200_000,
    "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0": 200_000,
    "bedrock:us.anthropic.claude-opus-4-20250514-v1:0": 200_000,
    "anthropic.claude-opus-4-1-20250805-v1:0": 200_000,
}


def get_model_token_limit(model_name: str) -> int | None:
    """Resolve an exact limit first, then the longest matching model key."""
    if model_name in MODEL_TOKEN_LIMITS:
        return MODEL_TOKEN_LIMITS[model_name]
    for model_key in sorted(MODEL_TOKEN_LIMITS, key=len, reverse=True):
        if model_key in model_name:
            return MODEL_TOKEN_LIMITS[model_key]
    return None

__all__ = ["MODEL_TOKEN_LIMITS", "get_model_token_limit"]
