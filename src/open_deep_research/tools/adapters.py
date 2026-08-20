"""Adapters that bring external tool implementations behind the Tool seam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from open_deep_research.tools.base import (
    ProgressCallback,
    ToolContext,
    ToolEffect,
    ToolExecutionZone,
    ToolOrigin,
    ToolResult,
)


@dataclass(frozen=True, slots=True)
class LangChainToolAdapter:
    """Adapt a LangChain BaseTool without making it the project contract.

    LangChain ``BaseTool`` does not expose a progress callback channel, so
    :meth:`call` intentionally cannot forward ``on_progress``.
    """

    adapted: BaseTool
    origin: ToolOrigin
    effect: ToolEffect = ToolEffect.READ_ONLY
    execution_zone: ToolExecutionZone = ToolExecutionZone.GATEWAY
    retryable: bool = False
    concurrency_safe: bool = False
    supports_idempotency: bool = False
    auth_satisfied: bool = False
    max_output_chars: Optional[int] = None
    _prompt: Callable[[RunnableConfig], Optional[str]] = lambda _: None
    _is_enabled: Callable[[RunnableConfig], bool] = lambda _: True
    _egress_urls: Callable[[dict[str, Any]], list[str]] = lambda _: []

    @property
    def name(self) -> str:
        """Return the adapted tool's registered name."""
        return self.adapted.name

    @property
    def input_schema(self) -> type[BaseModel]:
        """Return the Pydantic schema exposed by LangChain."""
        schema = self.adapted.input_schema
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise TypeError(f"LangChain tool '{self.name}' has no Pydantic input schema")
        return schema

    async def description(self, input: Optional[BaseModel] = None) -> str:
        """Return the external tool's static description."""
        del input
        return self.adapted.description or self.name

    def prompt(self, config: RunnableConfig) -> Optional[str]:
        """Return detailed model guidance declared by the adapter."""
        value = self._prompt(config)
        return None if value is None else str(value)

    def is_enabled(self, config: RunnableConfig) -> bool:
        """Return whether this adapted tool is enabled."""
        return bool(self._is_enabled(config))

    def egress_urls(self, input: dict[str, Any]) -> list[str]:
        """Return outbound URLs declared by the adapter."""
        urls = self._egress_urls(input)
        if not isinstance(urls, list) or any(not isinstance(url, str) for url in urls):
            raise TypeError(f"Tool '{self.name}' egress_urls() must return list[str]")
        return urls

    async def call(
        self,
        input: BaseModel,
        context: ToolContext,
        on_progress: Optional[ProgressCallback[Any]] = None,
    ) -> ToolResult[Any]:
        """Invoke the adapted LangChain tool with validated arguments."""
        del on_progress
        output = await self.adapted.ainvoke(input.model_dump(), context.config)
        return ToolResult(output=output)


def adapt_langchain_tool(
    tool: BaseTool,
    *,
    origin: ToolOrigin,
    effect: ToolEffect = ToolEffect.READ_ONLY,
    execution_zone: ToolExecutionZone = ToolExecutionZone.GATEWAY,
    retryable: bool = False,
    concurrency_safe: bool = False,
    supports_idempotency: bool = False,
    auth_satisfied: bool = False,
    prompt: Union[
        str, Callable[[RunnableConfig], Optional[str]], None
    ] = None,
    is_enabled: Optional[Callable[[RunnableConfig], bool]] = None,
    egress_urls: Optional[Callable[[dict[str, Any]], list[str]]] = None,
    max_output_chars: Optional[int] = None,
) -> LangChainToolAdapter:
    """Create a LangChain Adapter with explicit governance metadata."""
    prompt_fn: Callable[[RunnableConfig], Optional[str]]
    if isinstance(prompt, str):
        static_prompt = prompt

        def render_prompt(_: RunnableConfig) -> str:
            return static_prompt

        prompt_fn = render_prompt
    elif prompt is None:
        def no_prompt(_: RunnableConfig) -> None:
            return None

        prompt_fn = no_prompt
    elif callable(prompt):
        prompt_fn = prompt
    else:
        raise TypeError("prompt must be a string, callable, or None")
    if is_enabled is not None and not callable(is_enabled):
        raise TypeError("is_enabled must be callable or None")
    if egress_urls is not None and not callable(egress_urls):
        raise TypeError("egress_urls must be callable or None")
    if max_output_chars is not None and (
        isinstance(max_output_chars, bool)
        or not isinstance(max_output_chars, int)
        or max_output_chars <= 0
    ):
        raise ValueError("max_output_chars must be a positive integer or None")
    return LangChainToolAdapter(
        adapted=tool,
        origin=origin,
        effect=effect,
        execution_zone=execution_zone,
        retryable=retryable,
        concurrency_safe=concurrency_safe,
        supports_idempotency=supports_idempotency,
        auth_satisfied=auth_satisfied,
        max_output_chars=max_output_chars,
        _prompt=prompt_fn,
        _is_enabled=is_enabled or (lambda _: True),
        _egress_urls=egress_urls or (lambda _: []),
    )
