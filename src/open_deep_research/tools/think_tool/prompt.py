"""Model-facing guidance for ``think_tool``."""

TOOL_NAME = "think_tool"
DESCRIPTION = "Reflect on evidence and plan the next research action."


def render_prompt(config) -> str:
    del config
    return (
        "Call `think_tool` after a search or fetch has returned, by itself, to "
        "assess evidence quality and choose the next action. Do not call it in "
        "parallel with a search or another tool."
    )
