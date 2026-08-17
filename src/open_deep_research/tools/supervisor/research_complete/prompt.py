TOOL_NAME = "ResearchComplete"
DESCRIPTION = "Signal that Supervisor research is complete."


def render_prompt(config):
    del config
    return "Call `ResearchComplete` only after all required coverage has accepted evidence or an explicit unresolved gap."
