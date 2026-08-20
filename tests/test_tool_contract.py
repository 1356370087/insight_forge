"""Focused tests for the project-owned structural Tool Interface."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from open_deep_research.tools.base import (
    Tool,
    ToolContext,
    ToolEffect,
    ToolExecutionZone,
    ToolOrigin,
    ToolResult,
    build_tool,
    build_tool_registry,
    serialize_tool_output,
    tool_to_model_definition,
)


class EchoInput(BaseModel):
    """Input for contract probes."""

    text: str


def _tool(*, name: str = "echo"):
    async def call(input, context, on_progress=None):
        del context
        if on_progress is not None:
            emitted = on_progress(input.text)
            if emitted is not None:
                await emitted
        return ToolResult(output={"echo": input.text})

    return build_tool(
        name=name,
        description=lambda input: "Echo" if input is None else f"Echo {input.text}",
        input_schema=EchoInput,
        call=call,
        origin=ToolOrigin.SYSTEM,
    )


def test_plain_structural_object_satisfies_tool_protocol():
    @dataclass
    class PlainTool:
        name: str = "plain"
        input_schema: type[BaseModel] = EchoInput
        origin: ToolOrigin = ToolOrigin.SYSTEM
        retryable: bool = False
        effect: ToolEffect = ToolEffect.READ_ONLY
        execution_zone: ToolExecutionZone = ToolExecutionZone.SANDBOX_LOCAL
        concurrency_safe: bool = False
        max_output_chars: int | None = None

        def prompt(self, config):
            del config
            return None

        def is_enabled(self, config):
            del config
            return True

        def egress_urls(self, input):
            del input
            return []

        async def description(self, input=None):
            return "plain"

        async def call(self, input, context, on_progress=None):
            return ToolResult(output=input.text)

    assert isinstance(PlainTool(), Tool)


@pytest.mark.asyncio
async def test_dynamic_description_and_progress_callback():
    tool = _tool()
    input = EchoInput(text="hello")
    progress = []

    async def on_progress(value):
        progress.append(value)

    assert await tool.description(None) == "Echo"
    assert await tool.description(input) == "Echo hello"
    result = await tool.call(
        input,
        ToolContext(config={}, role="researcher", tool_call_id="1"),
        on_progress,
    )
    assert result.output == {"echo": "hello"}
    assert progress == ["hello"]


def test_registry_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate tool name"):
        build_tool_registry([_tool(), _tool()])


def test_builder_rejects_non_pydantic_schema():
    async def call(input, context, on_progress=None):
        return ToolResult(output=None)

    with pytest.raises(TypeError, match="Pydantic"):
        build_tool(
            name="bad",
            description="bad",
            input_schema=dict,
            call=call,
            origin=ToolOrigin.SYSTEM,
        )


@pytest.mark.asyncio
async def test_model_projection_contains_schema_without_execution_hook():
    definition = await tool_to_model_definition(_tool())
    assert set(definition) == {"name", "description", "parameters"}
    assert definition["parameters"]["required"] == ["text"]


def test_non_string_output_uses_stable_json_serialization():
    assert serialize_tool_output({"z": 1, "a": 2}) == '{"a":2,"z":1}'


def test_new_metadata_defaults_preserve_existing_behavior():
    tool = _tool()

    assert tool.effect is ToolEffect.READ_ONLY
    assert tool.concurrency_safe is False
    assert tool.prompt({}) is None
    assert tool.is_enabled({}) is True
    assert tool.egress_urls({"url": "https://example.com"}) == []
    assert tool.max_output_chars is None


def test_builder_accepts_declarative_prompt_availability_and_egress():
    async def call(input, context, on_progress=None):
        del context, on_progress
        return ToolResult(output=input.text)

    tool = build_tool(
        name="declared",
        description="Declared tool",
        input_schema=EchoInput,
        call=call,
        origin=ToolOrigin.SYSTEM,
        prompt=lambda config: f"mode={config['configurable']['mode']}",
        is_enabled=lambda config: config["configurable"]["mode"] == "on",
        egress_urls=lambda args: [args["url"]],
        max_output_chars=512,
    )
    config = {"configurable": {"mode": "on"}}

    assert tool.prompt(config) == "mode=on"
    assert tool.is_enabled(config) is True
    assert tool.egress_urls({"url": "https://example.com"}) == [
        "https://example.com"
    ]
    assert tool.max_output_chars == 512


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "100"])
def test_builder_rejects_invalid_output_budget(value):
    async def call(input, context, on_progress=None):
        return ToolResult(output=None)

    with pytest.raises(ValueError, match="positive integer"):
        build_tool(
            name="bad-budget",
            description="bad",
            input_schema=EchoInput,
            call=call,
            origin=ToolOrigin.SYSTEM,
            max_output_chars=value,
        )
