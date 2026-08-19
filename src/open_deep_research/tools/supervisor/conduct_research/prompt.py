TOOL_NAME = "ConductResearch"
DESCRIPTION = "Delegate one self-contained research objective to a Researcher."


def render_prompt(config):
    del config
    return (
        "Use `ConductResearch` for independent research directions and assign "
        "the factual coverage requirement IDs that the task owns. Process and "
        "deliverable-format requirements are not delegable; they are satisfied "
        "by the orchestration and the final report."
    )
