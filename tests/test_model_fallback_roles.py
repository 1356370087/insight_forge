"""Role-level model fallback policy regression tests."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from open_deep_research.agents.model_recovery import (
    build_model_candidate_chain,
    invoke_with_model_fallback,
)
from open_deep_research.configuration import Configuration
from open_deep_research.public_events import event_store_from_config


@pytest.mark.asyncio
async def test_shared_fallback_switches_and_sanitizes_provider_metadata(
    tmp_path,
) -> None:
    calls: list[tuple[str, list]] = []
    events: list[dict[str, str]] = []

    async def invoke(model_id: str, messages: list):
        calls.append((model_id, messages))
        if model_id == "openai:primary":
            raise RuntimeError("model unavailable")
        return AIMessage(content="ok")

    config = {
        "configurable": {"runs_dir": str(tmp_path)},
        "metadata": {"run_id": "fallback-event", "query_turn": 7},
    }
    response = await invoke_with_model_fallback(
        invoke,
        [
            AIMessage(
                content="prior",
                additional_kwargs={"signature": "provider-bound"},
                response_metadata={"reasoning": "provider-bound"},
            ),
            HumanMessage(content="continue"),
        ],
        primary_model="openai:primary",
        model_fallbacks={"compression": ["anthropic:fallback"]},
        role="compression",
        config=config,
        on_fallback=events.append,
    )

    assert response.content == "ok"
    assert [model_id for model_id, _messages in calls] == [
        "openai:primary",
        "anthropic:fallback",
    ]
    replayed = calls[1][1][0]
    assert "signature" not in replayed.additional_kwargs
    assert "reasoning" not in replayed.response_metadata
    assert events == [{
        "turn": 7,
        "from_model": "openai:primary",
        "to_model": "anthropic:fallback",
        "reason": "model_unavailable",
    }]
    public_events = event_store_from_config(config).read()
    fallback_event = next(
        event for event in public_events if event.type == "query.model_fallback"
    )
    assert fallback_event.payload == events[0]


@pytest.mark.asyncio
async def test_shared_fallback_does_not_cross_on_auth_errors() -> None:
    attempted: list[str] = []

    async def invoke(model_id: str, _messages: list):
        attempted.append(model_id)
        raise RuntimeError("invalid api key")

    with pytest.raises(RuntimeError, match="invalid api key"):
        await invoke_with_model_fallback(
            invoke,
            [HumanMessage(content="brief")],
            primary_model="openai:primary",
            model_fallbacks={"quality_evaluation": ["openai:fallback"]},
            role="quality_evaluation",
        )

    assert attempted == ["openai:primary"]


def test_candidate_chain_is_deduplicated_and_uses_role_config() -> None:
    template = object()
    candidates = build_model_candidate_chain(
        "openai:primary",
        ["openai:primary", "anthropic:fallback"],
        max_tokens=4096,
        config={},
        role="supervisor",
        model=template,
    )

    assert [candidate.model_id for candidate in candidates] == [
        "openai:primary",
        "anthropic:fallback",
    ]
    assert all(candidate.model is template for candidate in candidates)
    assert candidates[1].model_config["model"] == "anthropic:fallback"


def test_model_fallbacks_env_json_configures_role_chains(monkeypatch) -> None:
    monkeypatch.setenv(
        "MODEL_FALLBACKS",
        '{"researcher": ["anthropic:fallback", "openai:secondary"]}',
    )

    configurable = Configuration.from_runnable_config({})

    assert configurable.model_fallbacks == {
        "researcher": ["anthropic:fallback", "openai:secondary"],
    }
    candidates = build_model_candidate_chain(
        "openai:primary",
        configurable.model_fallbacks.get("researcher", []),
        max_tokens=4096,
        config={},
        role="researcher",
        model=object(),
    )
    assert [candidate.model_id for candidate in candidates] == [
        "openai:primary",
        "anthropic:fallback",
        "openai:secondary",
    ]


def test_model_fallbacks_env_empty_string_disables_fallback(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_FALLBACKS", "")

    configurable = Configuration.from_runnable_config({})

    assert configurable.model_fallbacks == {}


def test_model_fallbacks_env_rejects_unknown_roles(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_FALLBACKS", '{"lead": ["openai:fallback"]}')

    with pytest.raises(ValidationError, match="unknown model fallback roles"):
        Configuration.from_runnable_config({})
