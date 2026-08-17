"""Model-facing guidance for ``anthropic_web_search``."""

TOOL_NAME = "anthropic_web_search"
DESCRIPTION = "Search the web using Anthropic native web search."


def render_prompt(config) -> str:
    del config
    return (
        "Use `anthropic_web_search` in legacy or shadow mode for concise "
        "research queries. Verify important claims against its linked sources."
    )
