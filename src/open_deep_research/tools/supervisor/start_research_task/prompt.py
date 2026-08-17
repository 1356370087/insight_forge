TOOL_NAME = "StartResearchTask"
DESCRIPTION = "Launch an asynchronous background Researcher task."


def render_prompt(config):
    del config
    return "Use `StartResearchTask` for a complete standalone objective, record its task ID, and continue coordinating while it runs."
