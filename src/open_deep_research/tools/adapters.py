"""Adapters that bring external tool implementations behind the Tool seam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from open_deep_research.tools.base import (
    ProgressCallback,
    ToolContext,
    ToolEffect,
    ToolOrigin,
    ToolResult,
)


@dataclass(frozen=True, slots=True)
class LangChainToolAdapter:
    """Adapt a LangChain BaseTool without making it the project contract."""

    adapted: BaseTool
    origin: ToolOrigin
    effect: ToolEffect = ToolEffect.READ_ONLY
    retryable: bool = False
    auth_satisfied: bool = False

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
    retryable: bool = False,
    auth_satisfied: bool = False,
) -> LangChainToolAdapter:
    """Create a LangChain Adapter with explicit governance metadata."""
    return LangChainToolAdapter(
        adapted=tool,
        origin=origin,
        effect=effect,
        retryable=retryable,
        auth_satisfied=auth_satisfied,
    )
