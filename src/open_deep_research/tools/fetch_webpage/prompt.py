"""Model-facing guidance for ``fetch_webpage``."""

TOOL_NAME = "fetch_webpage"
DESCRIPTION = "Fetch a webpage in legacy or shadow web-pipeline mode."


def render_prompt(config) -> str:
    del config
    return (
        "Use `fetch_webpage` in legacy or shadow mode to retrieve one known "
        "HTTP(S) page. Its URL is subject to the run egress-domain policy."
    )
