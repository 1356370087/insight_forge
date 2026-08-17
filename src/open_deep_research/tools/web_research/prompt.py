"""Model-facing guidance for ``web_research``."""

TOOL_NAME = "web_research"
DESCRIPTION = "Run the governed Search → Top-K Fetch → Evidence web pipeline."


def render_prompt(config) -> str:
    del config
    return (
        "Use `web_research` in enforced mode for the normal web workflow. Give "
        "it a complete objective plus 1-3 short queries. It performs discovery, "
        "ranking, bounded fetching, extraction, and evidence registration."
    )
