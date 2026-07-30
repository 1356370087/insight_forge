"""Budget-aware request compilation for Supervisor and Researcher loops."""

from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately

from open_deep_research.agents.model_recovery import (
    resolve_model_context_window,
)
from open_deep_research.agents.tool_protocol import validate_tool_transcript
from open_deep_research.tools.base import Tool, tools_to_model_definitions


@dataclass(frozen=True, slots=True)
class ContextEnvelope:
    """Input capacity after mandatory request reservations."""

    context_window_tokens: int
    reserved_output_tokens: int
    system_prompt_tokens: int
    tool_schema_tokens: int
    safety_margin_tokens: int

    @property
    def available_input_tokens(self) -> int:
        """Return remaining tokens available to conversation history."""
        return max(
            0,
            self.context_window_tokens
            - self.reserved_output_tokens
            - self.system_prompt_tokens
            - self.tool_schema_tokens
            - self.safety_margin_tokens,
        )


@dataclass(frozen=True, slots=True)
class ContextCompilation:
    """Compiled request view and its budget diagnostics."""

    messages: tuple[BaseMessage, ...]
    envelope: ContextEnvelope
    estimated_input_tokens: int
    compacted: bool
    within_budget: bool


class ContextCompiler:
    """Compile a protocol-valid request view within a real model envelope."""

    def __init__(
        self,
        *,
        model_context_window_overrides: dict[str, int] | None = None,
        unknown_model_context_window_tokens: int = 32_768,
        max_tool_result_chars: int = 50_000,
    ) -> None:
        """Initialize model capability overrides and compaction limits."""
        self._overrides = dict(model_context_window_overrides or {})
        self._unknown_default = unknown_model_context_window_tokens
        self._max_tool_result_chars = max_tool_result_chars

    async def compile(
        self,
        messages: list[BaseMessage],
        *,
        system_prompt: str | BaseMessage | None,
        tools: list[Tool],
        model_name: str,
        reserved_output_tokens: int,
        target_ratio: float = 1.0,
        keep_last_messages: int | None = None,
        max_tool_description_chars: int = 2_000,
    ) -> ContextCompilation:
        """Build a bounded request while retaining complete tool-call pairs."""
        normalized = self._trim_tool_results(messages)
        validate_tool_transcript(normalized, allow_pending_tail=True)
        system_message = self._system_message(system_prompt)
        system_tokens = (
            count_tokens_approximately([system_message])
            if system_message is not None
            else 0
        )
        definitions = await tools_to_model_definitions(
            tools,
            max_description_chars=max_tool_description_chars,
        )
        tool_schema_tokens = max(
            0,
            len(
                json.dumps(
                    definitions,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            )
            // 4,
        )
        context_window = resolve_model_context_window(
            model_name,
            overrides=self._overrides,
            unknown_default=self._unknown_default,
        )
        safety_margin = max(2_048, int(context_window * 0.05))
        envelope = ContextEnvelope(
            context_window_tokens=context_window,
            reserved_output_tokens=max(0, reserved_output_tokens),
            system_prompt_tokens=system_tokens,
            tool_schema_tokens=tool_schema_tokens,
            safety_margin_tokens=safety_margin,
        )
        target_tokens = max(
            0,
            int(envelope.available_input_tokens * max(0.05, target_ratio)),
        )
        projected = self._project(
            normalized,
            target_tokens=target_tokens,
            keep_last_messages=keep_last_messages,
        )
        validate_tool_transcript(projected, allow_pending_tail=True)
        request_messages = (
            [system_message, *projected]
            if system_message is not None
            else projected
        )
        return ContextCompilation(
            messages=tuple(request_messages),
            envelope=envelope,
            estimated_input_tokens=count_tokens_approximately(
                request_messages
            ),
            compacted=len(projected) < len(normalized),
            within_budget=(
                count_tokens_approximately(request_messages)
                + envelope.tool_schema_tokens
                + envelope.reserved_output_tokens
                + envelope.safety_margin_tokens
                <= envelope.context_window_tokens
            ),
        )

    def deterministic_compact(
        self,
        messages: list[BaseMessage],
        *,
        target_tokens: int,
    ) -> list[BaseMessage]:
        """Provide a deterministic fail-safe when model summarization fails."""
        if not messages:
            return []
        protected = self._protected_prefix(messages)
        recent = self._project(
            messages[len(protected):],
            target_tokens=max(1, target_tokens // 2),
            keep_last_messages=None,
        )
        omitted_count = max(0, len(messages) - len(protected) - len(recent))
        summary = SystemMessage(
            content=(
                "[Deterministic context recovery]\n"
                f"{omitted_count} older messages were omitted. "
                "The authoritative research brief and recent complete "
                "tool interactions are retained."
            )
        )
        compacted = [*protected, summary, *recent]
        validate_tool_transcript(compacted, allow_pending_tail=True)
        return compacted

    def _trim_tool_results(
        self,
        messages: list[BaseMessage],
    ) -> list[BaseMessage]:
        trimmed: list[BaseMessage] = []
        for message in messages:
            if (
                not isinstance(message, ToolMessage)
                or not isinstance(message.content, str)
                or len(message.content) <= self._max_tool_result_chars
            ):
                trimmed.append(message)
                continue
            copied = message.model_copy(deep=True)
            copied.content = (
                message.content[: self._max_tool_result_chars]
                + "\n\n[Tool result trimmed by ContextCompiler]"
            )
            trimmed.append(copied)
        return trimmed

    @staticmethod
    def _system_message(
        system_prompt: str | BaseMessage | None,
    ) -> BaseMessage | None:
        if system_prompt is None:
            return None
        if isinstance(system_prompt, BaseMessage):
            return system_prompt
        return SystemMessage(content=system_prompt)

    @staticmethod
    def _protected_prefix(messages: list[BaseMessage]) -> list[BaseMessage]:
        protected: list[BaseMessage] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                protected.append(message)
                continue
            if not isinstance(message, ToolMessage):
                protected.append(message)
            break
        return protected

    def _project(
        self,
        messages: list[BaseMessage],
        *,
        target_tokens: int,
        keep_last_messages: int | None,
    ) -> list[BaseMessage]:
        if not messages:
            return []
        if count_tokens_approximately(messages) <= target_tokens:
            return list(messages)

        blocks = self._protocol_blocks(messages)
        protected = self._protected_prefix(messages)
        protected_ids = {id(message) for message in protected}
        protected_blocks = [
            block
            for block in blocks
            if any(id(message) in protected_ids for message in block)
        ]
        protected_block_ids = {id(block) for block in protected_blocks}
        recent_blocks = [
            block for block in blocks if id(block) not in protected_block_ids
        ]
        if keep_last_messages is not None:
            retained: list[list[BaseMessage]] = []
            retained_messages = 0
            for block in reversed(recent_blocks):
                retained.insert(0, block)
                retained_messages += len(block)
                if retained_messages >= max(1, keep_last_messages):
                    break
            recent_blocks = retained

        selected_blocks: list[list[BaseMessage]] = []
        protected_messages = [
            message for block in protected_blocks for message in block
        ]
        remaining = max(
            0,
            target_tokens
            - count_tokens_approximately(protected_messages),
        )
        for block in reversed(recent_blocks):
            candidate = [
                message
                for candidate_block in [block, *selected_blocks]
                for message in candidate_block
            ]
            if count_tokens_approximately(candidate) > remaining:
                continue
            selected_blocks.insert(0, block)

        projected = [
            message
            for block in [*protected_blocks, *selected_blocks]
            for message in block
        ]
        validate_tool_transcript(projected, allow_pending_tail=True)
        return projected

    @staticmethod
    def _protocol_blocks(
        messages: list[BaseMessage],
    ) -> list[list[BaseMessage]]:
        """Group each assistant tool call with all contiguous tool results."""
        blocks: list[list[BaseMessage]] = []
        cursor = 0
        while cursor < len(messages):
            message = messages[cursor]
            if not isinstance(message, AIMessage) or not message.tool_calls:
                blocks.append([message])
                cursor += 1
                continue
            call_ids = {
                str(call.get("id", "")) for call in message.tool_calls
            }
            block: list[BaseMessage] = [message]
            cursor += 1
            while cursor < len(messages):
                result = messages[cursor]
                if (
                    not isinstance(result, ToolMessage)
                    or str(result.tool_call_id) not in call_ids
                ):
                    break
                block.append(result)
                cursor += 1
            blocks.append(block)
        return blocks
