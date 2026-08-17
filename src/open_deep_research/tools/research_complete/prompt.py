"""Model-facing guidance for researcher completion."""

TOOL_NAME = "ResearchComplete"
DESCRIPTION = "Signal that the delegated research task is complete."


def render_prompt(config) -> str:
    del config
    return (
        "Call `ResearchComplete` only after the assigned objective has enough "
        "supported evidence or the remaining evidence gaps have been recorded."
    )
