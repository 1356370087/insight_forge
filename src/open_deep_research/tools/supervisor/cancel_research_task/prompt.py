TOOL_NAME = "CancelResearchTask"
DESCRIPTION = "Cancel one or more unnecessary asynchronous research tasks."


def render_prompt(config):
    del config
    return "Use `CancelResearchTask` when a running task is redundant, out of scope, or no longer useful."
