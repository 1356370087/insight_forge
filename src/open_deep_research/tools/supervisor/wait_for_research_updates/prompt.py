TOOL_NAME = "WaitForResearchUpdates"
DESCRIPTION = "Wait briefly for durable asynchronous Researcher mailbox updates."


def render_prompt(config):
    del config
    return "Use `WaitForResearchUpdates` when background tasks are still running and no other coordination action is useful."
