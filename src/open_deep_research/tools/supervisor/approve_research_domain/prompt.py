TOOL_NAME = "ApproveResearchDomain"
DESCRIPTION = "Approve or deny a pending research-task egress domain."


def render_prompt(config):
    del config
    return "Use `ApproveResearchDomain` only for the exact task and host reported as waiting for confirmation."
