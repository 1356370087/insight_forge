TOOL_NAME = "ReadResearchArtifact"
DESCRIPTION = "Read a bounded SHA-verified section of a persisted Researcher artifact."


def render_prompt(config):
    del config
    return "Use `ReadResearchArtifact` only when compressed findings are insufficient; prefer `evidence_registry` and paginate with the returned offset."
