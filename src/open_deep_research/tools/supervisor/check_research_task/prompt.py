TOOL_NAME = "CheckResearchTask"
DESCRIPTION = "Check status or retrieve results for asynchronous research tasks."


def render_prompt(config):
    del config
    return "Use `CheckResearchTask` to refresh one or more known task IDs and retrieve completed compressed findings."
