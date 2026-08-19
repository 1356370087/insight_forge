"""Configuration and opt-in live tests for the DashScope quality evaluator."""

from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import pytest
from dotenv import dotenv_values
from openai import OpenAI
from pydantic import BaseModel

from open_deep_research.configuration import Configuration
from open_deep_research.evaluation import JudgeConfig, build_judge_model
from open_deep_research.models.capabilities import dashscope_qwen_enable_thinking
from open_deep_research.quality import (
    _build_quality_model,
    _content_text,
    evaluate_subagent_handoff,
    evaluate_tool_results,
)

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
LIVE_TEST_ENV = "RUN_DASHSCOPE_QUALITY_LIVE_TEST"
_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$", re.IGNORECASE)


class _LiveJudgeProbe(BaseModel):
    ok: bool


def _quality_env() -> dict[str, str]:
    """Read only the quality-evaluator settings without exporting secrets."""
    values = dotenv_values(ENV_PATH)
    return {
        key: str(os.getenv(key) or values.get(key) or "").strip()
        for key in (
            "DASHSCOPE_API_KEY",
            "QUALITY_EVALUATION_ENABLED",
            "QUALITY_EVALUATION_MODEL",
            "QUALITY_EVALUATION_BASE_URL",
            "QUALITY_EVALUATION_FAIL_OPEN",
            "QUALITY_EVALUATION_RIGOR",
            "QUALITY_EVALUATION_MIN_SCORE",
            "QUALITY_EVALUATION_MIN_SOURCES",
        )
    }


def _dashscope_test_env() -> dict[str, str]:
    """Return deterministic unit-test settings independent of the developer .env."""
    return {
        "DASHSCOPE_API_KEY": "sk-dashscope-test",
        "QUALITY_EVALUATION_ENABLED": "true",
        "QUALITY_EVALUATION_MODEL": "openai:qwen3.7-flash",
        "QUALITY_EVALUATION_BASE_URL": (
            "https://workspace.cn-beijing.maas.aliyuncs.com/"
            "compatible-mode/v1"
        ),
        "QUALITY_EVALUATION_FAIL_OPEN": "false",
        "QUALITY_EVALUATION_RIGOR": "strict",
        "QUALITY_EVALUATION_MIN_SCORE": "4",
        "QUALITY_EVALUATION_MIN_SOURCES": "2",
    }


def _assert_dashscope_base_url(base_url: str) -> None:
    """Validate the deployable form of a DashScope OpenAI-compatible URL."""
    assert base_url, "QUALITY_EVALUATION_BASE_URL is required"
    assert "{" not in base_url and "}" not in base_url, (
        "QUALITY_EVALUATION_BASE_URL still contains template braces; replace "
        "{WorkspaceId} with the raw workspace ID and remove the braces"
    )
    parsed = urlparse(base_url)
    assert parsed.scheme == "https", "QUALITY_EVALUATION_BASE_URL must use HTTPS"
    assert parsed.username is None and parsed.password is None
    assert parsed.query == "" and parsed.fragment == ""
    assert parsed.path.rstrip("/") == "/compatible-mode/v1"
    assert parsed.hostname, "QUALITY_EVALUATION_BASE_URL must contain a hostname"
    assert parsed.hostname.endswith(".cn-beijing.maas.aliyuncs.com")
    assert all(_DNS_LABEL_RE.fullmatch(label) for label in parsed.hostname.split("."))


class TestDashScopeQualityEvaluationConfiguration:
    """Exercise the same configuration seam used by the runtime quality gate."""

    def test_dashscope_fixture_is_a_valid_openai_compatible_configuration(
        self,
    ) -> None:
        values = _dashscope_test_env()

        assert values["DASHSCOPE_API_KEY"].startswith("sk-"), (
            "DASHSCOPE_API_KEY does not look like a DashScope API key"
        )
        provider, separator, model = values["QUALITY_EVALUATION_MODEL"].partition(":")
        assert separator and provider == "openai"
        assert model and model.strip() == model
        assert values["QUALITY_EVALUATION_ENABLED"].lower() in {"true", "false"}
        assert values["QUALITY_EVALUATION_FAIL_OPEN"].lower() in {"true", "false"}
        if values["QUALITY_EVALUATION_RIGOR"]:
            assert values["QUALITY_EVALUATION_RIGOR"] in {
                "very_relaxed",
                "relaxed",
                "balanced",
                "strict",
                "very_strict",
            }
        else:
            assert 1 <= int(values["QUALITY_EVALUATION_MIN_SCORE"]) <= 5
        assert int(values["QUALITY_EVALUATION_MIN_SOURCES"]) >= 0
        _assert_dashscope_base_url(values["QUALITY_EVALUATION_BASE_URL"])

    def test_configuration_and_model_factory_use_dashscope_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        values = _dashscope_test_env()
        captured: dict[str, object] = {}

        class FakeModel:
            def bind(self, **kwargs):
                captured["bind"] = kwargs
                return self

        def fake_init_chat_model(**kwargs):
            captured["init"] = kwargs
            return FakeModel()

        for key, value in values.items():
            monkeypatch.setenv(key, value)
        monkeypatch.delenv("QUALITY_EVALUATION_API_KEY", raising=False)
        monkeypatch.setattr(
            "open_deep_research.quality.init_chat_model",
            fake_init_chat_model,
        )

        configurable = Configuration.from_runnable_config({"configurable": {}})
        _build_quality_model(configurable, {"configurable": {}})

        model_spec = values["QUALITY_EVALUATION_MODEL"]
        assert configurable.quality_evaluation_model == model_spec
        assert (
            configurable.quality_evaluation_base_url
            == values["QUALITY_EVALUATION_BASE_URL"]
        )
        expected_init = {
            "model": model_spec,
            "max_tokens": configurable.quality_evaluation_model_max_tokens,
            "max_retries": 0,
            "api_key": values["DASHSCOPE_API_KEY"],
            "base_url": values["QUALITY_EVALUATION_BASE_URL"],
        }
        if model_spec.split(":", 1)[1].lower().startswith("qwen"):
            expected_init["extra_body"] = {
                "enable_thinking": dashscope_qwen_enable_thinking(model_spec)
            }
        assert captured["init"] == expected_init
        assert captured["bind"] == {"response_format": {"type": "json_object"}}

    @pytest.mark.skipif(
        os.getenv(LIVE_TEST_ENV) != "1",
        reason=f"Set {LIVE_TEST_ENV}=1 to call the configured DashScope endpoint",
    )
    def test_live_openai_compatible_json_request(self) -> None:
        values = _quality_env()
        _assert_dashscope_base_url(values["QUALITY_EVALUATION_BASE_URL"])
        model = values["QUALITY_EVALUATION_MODEL"].split(":", 1)[1]
        client = OpenAI(
            api_key=values["DASHSCOPE_API_KEY"],
            base_url=values["QUALITY_EVALUATION_BASE_URL"],
            timeout=30.0,
            max_retries=0,
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": 'Return exactly this JSON object: {"ok": true}',
            }],
            response_format={"type": "json_object"},
            extra_body={
                "enable_thinking": dashscope_qwen_enable_thinking(model)
            },
        )

        content = response.choices[0].message.content
        assert content is not None
        assert json.loads(content) == {"ok": True}

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        os.getenv(LIVE_TEST_ENV) != "1",
        reason=f"Set {LIVE_TEST_ENV}=1 to call the runtime quality gate",
    )
    async def test_live_runtime_quality_gate_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        values = _quality_env()
        _assert_dashscope_base_url(values["QUALITY_EVALUATION_BASE_URL"])
        for key, value in values.items():
            monkeypatch.setenv(key, value)

        result = await evaluate_tool_results(
            "Assess whether three official sources support a test claim.",
            [{
                "name": "tavily_search",
                "content": (
                    "Source A: https://example.com/official-a supports the claim. "
                    "Source B: https://example.org/official-b independently supports it. "
                    "Source C: https://example.net/official-c confirms it."
                ),
                "error": False,
            }],
            {
                "configurable": {},
                "metadata": {"run_id": "dashscope-quality-live-test"},
            },
        )

        assert result.evaluator_error is None
        assert result.deterministic_checks["passed"] is True
        assert result.deterministic_checks["source_count"] == 3
        assert 1 <= result.relevance <= 5
        assert 1 <= result.source_quality <= 5
        assert 1 <= result.evidence_coverage <= 5
        assert 1 <= result.corroboration <= 5

    @pytest.mark.skipif(
        os.getenv(LIVE_TEST_ENV) != "1",
        reason=f"Set {LIVE_TEST_ENV}=1 to call the configured Judge endpoint",
    )
    def test_live_offline_judge_reuses_runtime_qwen_configuration(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        values = _quality_env()
        for key, value in values.items():
            monkeypatch.setenv(key, value)
        for key in ("EVALUATION_MODEL", "EVALUATION_BASE_URL", "EVALUATION_API_KEY"):
            monkeypatch.delenv(key, raising=False)

        model = build_judge_model(JudgeConfig.from_env())
        structured = model.with_structured_output(
            _LiveJudgeProbe,
            method="function_calling",
        )
        response = structured.invoke(
            "Return ok=true to confirm this synthetic evaluation probe.",
        )

        assert response.ok is True


class TestQualityEvaluationProviderIsolation:
    """Ensure one provider's credentials and request options do not leak to another."""

    def test_openai_model_uses_openai_key_when_dashscope_key_is_also_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        class FakeModel:
            def bind(self, **kwargs):
                captured["bind"] = kwargs
                return self

        def fake_init_chat_model(**kwargs):
            captured["init"] = kwargs
            return FakeModel()

        monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        monkeypatch.setattr(
            "open_deep_research.quality.init_chat_model",
            fake_init_chat_model,
        )
        configurable = Configuration(
            quality_evaluation_model="openai:gpt-4.1-mini",
            quality_evaluation_base_url=None,
        )

        _build_quality_model(configurable, {"configurable": {}})

        assert captured["init"]["api_key"] == "openai-key"
        assert "extra_body" not in captured["init"]

    def test_native_anthropic_model_has_no_openai_only_options(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        class FakeModel:
            def bind(self, **kwargs):
                captured["bind"] = kwargs
                return self

        def fake_init_chat_model(**kwargs):
            captured["init"] = kwargs
            return FakeModel()

        monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
        monkeypatch.setattr(
            "open_deep_research.quality.init_chat_model",
            fake_init_chat_model,
        )
        configurable = Configuration(
            quality_evaluation_model="anthropic:claude-sonnet-4-5",
            quality_evaluation_base_url=None,
        )

        _build_quality_model(configurable, {"configurable": {}})

        assert captured["init"]["api_key"] == "anthropic-key"
        assert "extra_body" not in captured["init"]
        assert "bind" not in captured

    def test_explicit_quality_key_supports_other_openai_compatible_endpoints(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        class FakeModel:
            def bind(self, **_kwargs):
                return self

        monkeypatch.setenv("QUALITY_EVALUATION_API_KEY", "endpoint-key")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        monkeypatch.setattr(
            "open_deep_research.quality.init_chat_model",
            lambda **kwargs: captured.update(kwargs) or FakeModel(),
        )
        configurable = Configuration(
            quality_evaluation_model="openai:custom-model",
            quality_evaluation_base_url="https://models.example.test/v1",
        )

        _build_quality_model(configurable, {"configurable": {}})

        assert captured["api_key"] == "endpoint-key"

    def test_text_content_blocks_are_normalized_for_json_validation(self) -> None:
        assert _content_text([
            {"type": "text", "text": '{"decision":"continue"}'},
        ]) == '{"decision":"continue"}'

    def test_fenced_json_is_normalized_for_provider_compatibility(self) -> None:
        assert _content_text(
            '```json\n{"decision":"continue"}\n```',
        ) == '{"decision":"continue"}'

    @pytest.mark.asyncio
    async def test_handoff_judge_receives_authoritative_runtime_date(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        async def fake_evaluate_json(
            _schema,
            system_prompt,
            payload,
            _config,
            **_kwargs,
        ):
            captured["system_prompt"] = system_prompt
            captured["payload"] = payload
            return {
                "accepted": True,
                "relevance": 3,
                "source_quality": 3,
                "evidence_coverage": 3,
                "groundedness": 3,
                "missing_information": [],
                "unsupported_claims": [],
                "follow_up_tasks": [],
                "reason": "Synthetic acceptance.",
            }

        monkeypatch.setattr(
            "open_deep_research.quality._evaluate_json",
            fake_evaluate_json,
        )
        handoff = {
            "compressed_research": (
                "A sufficiently detailed synthetic handoff supported by "
                "https://example.com/source-a and https://example.org/source-b. "
            )
            * 3,
            "raw_notes": [],
        }

        result = await evaluate_subagent_handoff(
            "Evaluate a synthetic research handoff.",
            handoff,
            {"configurable": {}},
        )

        assert result.accepted is True
        assert captured["payload"]["runtime_current_date"] == date.today().isoformat()
        assert isinstance(captured["payload"]["evidence_registry"], list)
        assert "training cutoff" in captured["system_prompt"]

    @pytest.mark.asyncio
    async def test_tool_gate_receives_deduplicated_json_native_evidence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        async def fake_evaluate_json(
            _schema,
            _system_prompt,
            payload,
            _config,
            **_kwargs,
        ):
            captured["payload"] = payload
            return {
                "decision": "complete",
                "relevance": 4,
                "source_quality": 4,
                "evidence_coverage": 4,
                "corroboration": 4,
                "unresolved_conflicts": [],
                "missing_information": [],
                "suggested_queries": [],
                "reason": "The structured cumulative evidence is sufficient.",
            }

        monkeypatch.setattr(
            "open_deep_research.quality._evaluate_json",
            fake_evaluate_json,
        )
        evidence_registry = [
            {
                "evidence_id": "ev-79",
                "claim": "PEP 8 limits code lines to 79 characters.",
                "supporting_excerpt": "Limit all lines to a maximum of 79 characters.",
                "source_url": "https://peps.python.org/pep-0008/",
                "source_authority": 1.0,
                "security_status": "accepted",
            },
            {
                "evidence_id": "ev-79",
                "claim": "Duplicate of the same evidence.",
                "source_url": "https://peps.python.org/pep-0008/",
                "security_status": "accepted",
            },
            {
                "evidence_id": "ev-72",
                "claim": "PEP 8 limits comments and docstrings to 72 characters.",
                "supporting_excerpt": (
                    "For flowing long blocks of text (docstrings or comments), "
                    "the line length should be limited to 72 characters."
                ),
                "source_url": "https://peps.python.org/pep-0008/",
                "source_authority": 1.0,
                "security_status": "accepted",
            },
        ]

        result = await evaluate_tool_results(
            "Extract both the 79- and 72-character PEP 8 recommendations.",
            [{
                "name": "fetch_url",
                "content": "Structured fetch completed.",
                "error": False,
            }],
            {
                "configurable": {
                    "quality_evaluation_rigor": "balanced",
                    "quality_evaluation_min_sources": 1,
                }
            },
            evidence_registry=evidence_registry,
        )

        payload = captured["payload"]
        assert isinstance(payload, dict)
        cumulative = payload["cumulative_evidence"]
        assert isinstance(cumulative, list)
        assert [item["evidence_id"] for item in cumulative] == ["ev-79", "ev-72"]
        assert "72 characters" in cumulative[1]["supporting_excerpt"]
        assert payload["cumulative_evidence_stats"] == {
            "accepted_count": 3,
            "unique_count": 2,
            "included_count": 2,
            "truncated": False,
        }
        assert result.decision == "complete"
