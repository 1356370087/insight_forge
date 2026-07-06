"""Unit tests for the report profile registry."""

from __future__ import annotations

from open_deep_research import prompts
from open_deep_research.configuration import Configuration
from open_deep_research.report.profiles import (
    REPORT_PROFILES,
    AssemblyMode,
    OutputFormat,
    ReferenceStyle,
    get_profile,
)


def test_get_profile_returns_default_for_none_and_unknown():
    default = REPORT_PROFILES["default"]
    assert get_profile(None) is default
    assert get_profile("") is default
    assert get_profile("does-not-exist") is default
    assert get_profile("default") is default


def test_default_profile_matches_original_behavior():
    profile = get_profile("default")
    assert profile.prompt_template == "final_report_generation_prompt"
    assert profile.assembly == AssemblyMode.ONE_SHOT
    assert profile.default_format == OutputFormat.MARKDOWN
    assert profile.reference_style == ReferenceStyle.NUMBERED


def test_every_profile_prompt_resolves_to_a_real_constant():
    """A profile must not reference a prompt constant that doesn't exist."""
    for key, profile in REPORT_PROFILES.items():
        assert hasattr(prompts, profile.prompt_template), (
            f"profile {key!r} references unknown prompt {profile.prompt_template!r}"
        )


def test_optional_report_controls_do_not_set_ui_defaults():
    properties = Configuration.model_json_schema()["properties"]

    assert properties["output_format"]["metadata"]["x_oap_ui_config"]["default"] is None
    assert properties["reference_style"]["metadata"]["x_oap_ui_config"]["default"] is None
