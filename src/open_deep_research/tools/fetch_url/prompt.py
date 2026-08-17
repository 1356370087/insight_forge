"""Model-facing guidance for ``fetch_url``."""

TOOL_NAME = "fetch_url"
DESCRIPTION = "Fetch and extract evidence from one specific known URL."


def render_prompt(config) -> str:
    del config
    return (
        "Use `fetch_url` only when the exact URL is already known or the task "
        "contract restricts research to that URL. Supply the evidence objective "
        "so extraction keeps only relevant claims."
    )
