"""Model-facing guidance for ``tavily_search``."""

TOOL_NAME = "tavily_search"
DESCRIPTION = "Search the web with Tavily and return a multi-source research digest."


def render_prompt(config) -> str:
    del config
    return (
        "Use `tavily_search` in legacy or shadow mode for short, focused web "
        "queries. Start broad, inspect the returned sources, then refine only "
        "when the evidence leaves a concrete gap."
    )
