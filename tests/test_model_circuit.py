from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from open_deep_research.configuration import (
    RUN_CONFIG_FROZEN_FIELDS,
    RUN_CONFIG_FROZEN_FIELDS_V4,
    RUN_CONFIG_SCHEMA_VERSION,
    Configuration,
    freeze_run_config,
)
from open_deep_research.events.public import sanitize_public_payload
from open_deep_research.events.task_activity import sanitize_task_activity_payload
from open_deep_research.models.circuit import (
    CircuitFailureKind,
    CircuitOpenError,
    ModelCircuitBreaker,
    ModelCircuitPolicy,
    ModelCircuitState,
    _reset_model_circuit_registry,
    get_model_circuit_registry,
    model_circuit_policy_from_configuration,
)
from open_deep_research.models.fallback import invoke_with_model_fallback
from open_deep_research.observability import core as observability_core
from open_deep_research.observability import invoke_model_with_retry_observability


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def policy(**overrides) -> ModelCircuitPolicy:
    values = {
        "failure_threshold": 2,
        "failure_window_seconds": 300.0,
        "open_cooldown_seconds": 60.0,
        "max_cooldown_seconds": 600.0,
        "slow_ratio_threshold": 0.5,
        "slow_min_samples": 4,
        "first_packet_probe": "shadow",
        "slow_first_packet_threshold_seconds": 8.0,
    }
    values.update(overrides)
    return ModelCircuitPolicy(**values)


@pytest.mark.asyncio
async def test_failure_threshold_opens_and_window_expiry_prunes() -> None:
    clock = FakeClock()
    breaker = ModelCircuitBreaker("openai:test", policy(), now=clock)

    first, _ = await breaker.before_call()
    assert await breaker.record_failure(
        first,
        failure_kind=CircuitFailureKind.RATE_LIMITED,
    ) is None
    assert (await breaker.snapshot()).state is ModelCircuitState.CLOSED

    clock.advance(301)
    second, _ = await breaker.before_call()
    assert await breaker.record_failure(
        second,
        failure_kind=CircuitFailureKind.TRANSIENT,
    ) is None
    snapshot = await breaker.snapshot()
    assert snapshot.state is ModelCircuitState.CLOSED
    assert snapshot.failure_count == 1

    third, _ = await breaker.before_call()
    transition = await breaker.record_failure(
        third,
        failure_kind=CircuitFailureKind.MODEL_UNAVAILABLE,
    )
    assert transition is not None
    assert transition.to_state is ModelCircuitState.OPEN
    assert transition.failure_count == 2


@pytest.mark.asyncio
async def test_open_rejects_then_allows_one_half_open_probe() -> None:
    clock = FakeClock()
    breaker = ModelCircuitBreaker(
        "openai:test",
        policy(failure_threshold=1),
        now=clock,
    )
    permit, _ = await breaker.before_call()
    await breaker.record_failure(
        permit,
        failure_kind=CircuitFailureKind.RATE_LIMITED,
    )

    with pytest.raises(CircuitOpenError) as rejected:
        await breaker.before_call()
    assert rejected.value.retry_after_seconds == 60.0

    clock.advance(60)

    async def attempt():
        try:
            return await breaker.before_call()
        except CircuitOpenError as exc:
            return exc

    results = await asyncio.gather(attempt(), attempt(), attempt())
    permits = [result for result in results if isinstance(result, tuple)]
    errors = [result for result in results if isinstance(result, CircuitOpenError)]
    assert len(permits) == 1
    assert permits[0][0].is_probe is True
    assert len(errors) == 2
    assert all(error.reason == "half_open_probe_in_flight" for error in errors)


@pytest.mark.asyncio
async def test_probe_success_closes_and_clears_windows() -> None:
    clock = FakeClock()
    breaker = ModelCircuitBreaker(
        "openai:test",
        policy(failure_threshold=1),
        now=clock,
    )
    permit, _ = await breaker.before_call()
    await breaker.record_failure(
        permit,
        failure_kind=CircuitFailureKind.TRANSIENT,
    )
    clock.advance(60)
    probe, transition = await breaker.before_call()
    assert transition is not None
    assert transition.to_state is ModelCircuitState.HALF_OPEN

    recovered = await breaker.record_success(probe, ttft_seconds=1.0)
    assert recovered is not None
    assert recovered.to_state is ModelCircuitState.CLOSED
    snapshot = await breaker.snapshot()
    assert snapshot.failure_count == 0
    assert snapshot.sample_count == 0
    assert snapshot.cooldown_seconds == 60.0


@pytest.mark.asyncio
async def test_probe_failures_back_off_to_cap() -> None:
    clock = FakeClock()
    breaker = ModelCircuitBreaker(
        "openai:test",
        policy(failure_threshold=1),
        now=clock,
    )
    permit, _ = await breaker.before_call()
    await breaker.record_failure(
        permit,
        failure_kind=CircuitFailureKind.TRANSIENT,
    )
    assert (await breaker.snapshot()).cooldown_seconds == 60.0

    expected = [120.0, 240.0, 480.0, 600.0, 600.0]
    for cooldown in expected:
        current = await breaker.snapshot()
        clock.advance(current.cooldown_seconds)
        probe, _ = await breaker.before_call()
        transition = await breaker.record_failure(
            probe,
            failure_kind=CircuitFailureKind.MODEL_UNAVAILABLE,
        )
        assert transition is not None
        assert transition.cooldown_seconds == cooldown


@pytest.mark.asyncio
async def test_shadow_slow_samples_do_not_open() -> None:
    clock = FakeClock()
    breaker = ModelCircuitBreaker("openai:test", policy(), now=clock)
    for _ in range(6):
        permit, _ = await breaker.before_call()
        assert await breaker.record_success(permit, ttft_seconds=9.0) is None
    snapshot = await breaker.snapshot()
    assert snapshot.state is ModelCircuitState.CLOSED
    assert snapshot.slow_count == 6


@pytest.mark.asyncio
async def test_enforced_slow_ratio_opens_at_minimum_samples() -> None:
    clock = FakeClock()
    breaker = ModelCircuitBreaker(
        "openai:test",
        policy(first_packet_probe="enforced"),
        now=clock,
    )
    ttfts = [9.0, 1.0, 9.0, 1.0]
    transition = None
    for ttft in ttfts:
        permit, _ = await breaker.before_call()
        transition = await breaker.record_success(permit, ttft_seconds=ttft)
    assert transition is not None
    assert transition.reason == "slow_first_packet_ratio"
    assert transition.to_state is ModelCircuitState.OPEN


@pytest.mark.asyncio
async def test_stale_generation_outcome_cannot_close_new_state() -> None:
    clock = FakeClock()
    breaker = ModelCircuitBreaker(
        "openai:test",
        policy(failure_threshold=1),
        now=clock,
    )
    first, _ = await breaker.before_call()
    stale, _ = await breaker.before_call()
    await breaker.record_failure(
        first,
        failure_kind=CircuitFailureKind.RATE_LIMITED,
    )
    assert await breaker.record_success(stale, ttft_seconds=1.0) is None
    assert (await breaker.snapshot()).state is ModelCircuitState.OPEN


@pytest.mark.asyncio
async def test_inconclusive_probe_releases_single_flight_without_backoff() -> None:
    clock = FakeClock()
    breaker = ModelCircuitBreaker(
        "openai:test",
        policy(failure_threshold=1),
        now=clock,
    )
    permit, _ = await breaker.before_call()
    await breaker.record_failure(
        permit,
        failure_kind=CircuitFailureKind.TRANSIENT,
    )
    clock.advance(60)
    probe, _ = await breaker.before_call()
    transition = await breaker.record_inconclusive(probe)
    assert transition is not None
    assert transition.reason == "probe_inconclusive"
    snapshot = await breaker.snapshot()
    assert snapshot.state is ModelCircuitState.OPEN
    assert snapshot.probe_in_flight is False
    assert snapshot.cooldown_seconds == 60.0


@pytest.mark.asyncio
async def test_registry_forces_oldest_open_candidate() -> None:
    _reset_model_circuit_registry()
    clock = FakeClock()
    registry = get_model_circuit_registry()
    circuit_policy = policy(failure_threshold=1)
    first = registry.get_or_create("openai:first", circuit_policy, now=clock)
    second = registry.get_or_create("openai:second", circuit_policy, now=clock)
    assert first is not None and second is not None

    permit, _ = await first.before_call()
    await first.record_failure(
        permit,
        failure_kind=CircuitFailureKind.TRANSIENT,
    )
    clock.advance(10)
    permit, _ = await second.before_call()
    await second.record_failure(
        permit,
        failure_kind=CircuitFailureKind.TRANSIENT,
    )

    index, transition = await registry.select_candidate_index(
        ["openai:first", "openai:second"],
        circuit_policy,
    )
    assert index == 0
    assert transition is not None
    assert transition.forced_probe is True
    assert (await first.snapshot()).state is ModelCircuitState.HALF_OPEN


def test_registry_policy_mismatch_fails_open_without_replacement(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _reset_model_circuit_registry()
    registry = get_model_circuit_registry()
    original = policy(failure_threshold=2)
    changed = policy(failure_threshold=3)
    first = registry.get_or_create("openai:test", original)

    assert first is not None
    with caplog.at_level("WARNING", logger="open_deep_research.models.circuit"):
        assert registry.get_or_create("openai:test", changed) is None
        assert registry.get_or_create("openai:test", changed) is None
    assert registry.get("openai:test") is first
    mismatch_warnings = [
        record
        for record in caplog.records
        if "Model circuit policy mismatch" in record.getMessage()
    ]
    assert len(mismatch_warnings) == 1


def test_registry_reset_removes_breakers() -> None:
    _reset_model_circuit_registry()
    registry = get_model_circuit_registry()
    assert registry.get_or_create("openai:test", policy()) is not None
    _reset_model_circuit_registry()
    assert get_model_circuit_registry().get("openai:test") is None


def test_circuit_configuration_defaults_remain_frozen_in_v7() -> None:
    configuration = Configuration()
    assert RUN_CONFIG_SCHEMA_VERSION == 7
    assert configuration.model_circuit_breaker_enabled is True
    assert configuration.model_first_packet_probe == "shadow"

    frozen = freeze_run_config({"configurable": {}, "metadata": {}})
    assert frozen["metadata"]["run_config_schema_version"] == 7
    for field_name in (
        "model_circuit_breaker_enabled",
        "model_circuit_failure_threshold",
        "model_circuit_failure_window_seconds",
        "model_circuit_open_cooldown_seconds",
        "model_circuit_max_cooldown_seconds",
        "model_circuit_slow_ratio_threshold",
        "model_circuit_slow_min_samples",
        "model_first_packet_probe",
        "model_first_packet_timeout_seconds",
        "model_slow_first_packet_threshold_seconds",
    ):
        assert field_name in RUN_CONFIG_FROZEN_FIELDS
        assert field_name not in RUN_CONFIG_FROZEN_FIELDS_V4
        assert field_name in frozen["configurable"]


@pytest.mark.parametrize(
    "values",
    [
        {
            "model_circuit_open_cooldown_seconds": 61,
            "model_circuit_max_cooldown_seconds": 60,
        },
        {
            "model_slow_first_packet_threshold_seconds": 15,
            "model_first_packet_timeout_seconds": 15,
        },
    ],
)
def test_circuit_configuration_rejects_invalid_threshold_relationships(
    values,
) -> None:
    with pytest.raises(ValidationError):
        Configuration(**values)


class CountingModel:
    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.calls = 0

    async def ainvoke(self, _messages, config=None):
        self.calls += 1
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def circuit_config() -> dict:
    return {
        "configurable": {
            "model_circuit_breaker_enabled": True,
            "model_circuit_failure_threshold": 1,
            "model_first_packet_probe": "off",
            "observability_enabled": False,
        }
    }


async def open_registered_circuit(model_id: str, config: dict) -> None:
    configuration = Configuration.from_runnable_config(config)
    breaker = get_model_circuit_registry().get_or_create(
        model_id,
        ModelCircuitPolicy(
            failure_threshold=configuration.model_circuit_failure_threshold,
            failure_window_seconds=(
                configuration.model_circuit_failure_window_seconds
            ),
            open_cooldown_seconds=(
                configuration.model_circuit_open_cooldown_seconds
            ),
            max_cooldown_seconds=(
                configuration.model_circuit_max_cooldown_seconds
            ),
            slow_ratio_threshold=(
                configuration.model_circuit_slow_ratio_threshold
            ),
            slow_min_samples=configuration.model_circuit_slow_min_samples,
            first_packet_probe=configuration.model_first_packet_probe,
            slow_first_packet_threshold_seconds=(
                configuration.model_slow_first_packet_threshold_seconds
            ),
        ),
    )
    assert breaker is not None
    permit, _ = await breaker.before_call()
    await breaker.record_failure(
        permit,
        failure_kind=CircuitFailureKind.RATE_LIMITED,
    )


@pytest.mark.asyncio
async def test_retry_fast_rejects_open_circuit_without_call_or_sleep() -> None:
    config = circuit_config()
    await open_registered_circuit("openai:primary", config)
    model = CountingModel(AIMessage(content="unused"))
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    with pytest.raises(CircuitOpenError):
        await invoke_model_with_retry_observability(
            model,
            [HumanMessage(content="question")],
            config,
            span_name="test.circuit.fast_fail",
            model_name="openai:primary",
            max_attempts=3,
            sleeper=fake_sleep,
        )

    assert model.calls == 0
    assert sleeps == []


@pytest.mark.asyncio
async def test_open_primary_falls_back_without_provider_call() -> None:
    config = circuit_config()
    await open_registered_circuit("openai:primary", config)
    primary = CountingModel(AIMessage(content="unused"))
    fallback = CountingModel(AIMessage(content="fallback"))

    async def invoke(model_id: str, messages: list):
        model = primary if model_id == "openai:primary" else fallback
        return await invoke_model_with_retry_observability(
            model,
            messages,
            config,
            span_name="test.circuit.fallback",
            model_name=model_id,
        )

    response = await invoke_with_model_fallback(
        invoke,
        [HumanMessage(content="question")],
        primary_model="openai:primary",
        model_fallbacks={"researcher": ["anthropic:fallback"]},
        role="researcher",
        config=config,
    )

    assert response.content == "fallback"
    assert primary.calls == 0
    assert fallback.calls == 1


class InvalidRequestError(RuntimeError):
    status_code = 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [RuntimeError("invalid api key"), InvalidRequestError("bad request")],
)
async def test_non_availability_errors_do_not_enter_failure_window(
    error: BaseException,
) -> None:
    config = circuit_config()
    model_id = f"openai:non-counted-{type(error).__name__}"
    with pytest.raises(type(error)):
        await invoke_model_with_retry_observability(
            CountingModel(error),
            [HumanMessage(content="question")],
            config,
            span_name="test.circuit.non_counted",
            model_name=model_id,
            max_attempts=1,
        )

    configuration = Configuration.from_runnable_config(config)
    breaker = get_model_circuit_registry().get(
        model_id,
        model_circuit_policy_from_configuration(configuration),
    )
    assert breaker is not None
    snapshot = await breaker.snapshot()
    assert snapshot.state is ModelCircuitState.CLOSED
    assert snapshot.failure_count == 0


class StreamModel:
    def __init__(self, chunks) -> None:
        self.chunks = chunks
        self.stream_calls = 0
        self.invoke_calls = 0

    async def astream(self, _messages, config=None):
        self.stream_calls += 1
        for chunk in self.chunks:
            yield chunk

    async def ainvoke(self, _messages, config=None):
        self.invoke_calls += 1
        return AIMessage(content="ainvoke")


def probe_config(mode: str, **overrides) -> dict:
    values = {
        "model_circuit_breaker_enabled": True,
        "model_first_packet_probe": mode,
        "observability_enabled": False,
    }
    values.update(overrides)
    return {"configurable": values}


@pytest.mark.asyncio
async def test_probe_off_uses_ainvoke_only() -> None:
    model = StreamModel([observability_core.AIMessageChunk(content="stream")])
    config = probe_config("off")
    result = await observability_core._ainvoke_model(
        model,
        [HumanMessage(content="question")],
        observability_core.get_trace_recorder(config),
        config,
    )
    assert result.response.content == "ainvoke"
    assert result.ttft_seconds is None
    assert model.stream_calls == 0
    assert model.invoke_calls == 1


@pytest.mark.asyncio
async def test_shadow_stream_merges_content_and_tool_call_chunks() -> None:
    chunks = [
        observability_core.AIMessageChunk(
            content="hel",
            tool_call_chunks=[
                {"name": "search", "args": '{"q":', "id": "call-1", "index": 0}
            ],
        ),
        observability_core.AIMessageChunk(
            content="lo",
            tool_call_chunks=[
                {"name": None, "args": '"x"}', "id": None, "index": 0}
            ],
        ),
    ]
    model = StreamModel(chunks)
    config = probe_config("shadow")
    result = await observability_core._ainvoke_model(
        model,
        [HumanMessage(content="question")],
        observability_core.get_trace_recorder(config),
        config,
    )
    assert result.response.content == "hello"
    assert result.response.tool_calls[0]["args"] == {"q": "x"}
    assert result.response.tool_calls[0]["id"] == "call-1"
    assert result.ttft_seconds is not None
    assert result.probe_status == "streamed"
    assert model.invoke_calls == 0


@pytest.mark.asyncio
async def test_structured_single_value_is_not_recorded_as_ttft() -> None:
    model = StreamModel([{"accepted": True}])
    config = probe_config("shadow")
    result = await observability_core._ainvoke_model(
        model,
        [HumanMessage(content="question")],
        observability_core.get_trace_recorder(config),
        config,
    )
    assert result.response == {"accepted": True}
    assert result.ttft_seconds is None
    assert result.probe_status == "non_streaming_wrapper"


def _clarify_model_cls():
    from pydantic import BaseModel

    class ClarifyProbe(BaseModel):
        need_clarification: bool = False
        verification: str = ""

    return ClarifyProbe


@pytest.mark.asyncio
async def test_structured_output_partials_return_final_parsed_value() -> None:
    # Regression for DeepSeek-compatible endpoints: runnables built via
    # ``with_structured_output(..., method="function_calling")`` stream one
    # same-typed pydantic partial per chunk and the final item is complete.
    # The probe must return it without a second (ainvoke) request.
    clarify = _clarify_model_cls()
    model = StreamModel(
        [
            clarify(need_clarification=False, verification=""),
            clarify(need_clarification=False, verification="go"),
            clarify(need_clarification=False, verification="go ahead"),
        ]
    )
    config = probe_config("shadow")
    result = await observability_core._ainvoke_model(
        model,
        [HumanMessage(content="question")],
        observability_core.get_trace_recorder(config),
        config,
    )
    assert result.response.verification == "go ahead"
    assert result.ttft_seconds is not None
    assert result.probe_status == "non_streaming_wrapper"
    assert model.invoke_calls == 0


@pytest.mark.asyncio
async def test_fragment_style_stream_falls_back_to_plain_invoke() -> None:
    # String fragments cannot be merged faithfully (the last fragment is not
    # the full value), so the probe degrades to the non-streaming invoke.
    model = StreamModel(["hel", "lo", "!"])
    config = probe_config("shadow")
    result = await observability_core._ainvoke_model(
        model,
        [HumanMessage(content="question")],
        observability_core.get_trace_recorder(config),
        config,
    )
    assert result.response.content == "ainvoke"
    assert result.probe_status == "fallback"
    assert model.invoke_calls == 1


@pytest.mark.asyncio
async def test_mixed_chunk_types_degrade_in_every_probe_mode() -> None:
    # A chunk-type change mid-stream used to raise TypeError even in shadow
    # mode (the first packet had already arrived, so the fallback guard was
    # bypassed); shape violations must degrade in shadow and enforced alike.
    for mode in ("shadow", "enforced"):
        model = StreamModel(
            [
                observability_core.AIMessageChunk(content="hel"),
                "lo",
            ]
        )
        config = probe_config(mode)
        result = await observability_core._ainvoke_model(
            model,
            [HumanMessage(content="question")],
            observability_core.get_trace_recorder(config),
            config,
        )
        assert result.response.content == "ainvoke"
        assert result.probe_status == "fallback"
        assert model.invoke_calls == 1


class UnsupportedStreamModel(StreamModel):
    async def astream(self, _messages, config=None):
        self.stream_calls += 1
        raise NotImplementedError("streaming is not supported")


@pytest.mark.asyncio
async def test_unsupported_stream_falls_back_before_first_chunk() -> None:
    model = UnsupportedStreamModel([])
    config = probe_config("shadow")
    result = await observability_core._ainvoke_model(
        model,
        [HumanMessage(content="question")],
        observability_core.get_trace_recorder(config),
        config,
    )
    assert result.response.content == "ainvoke"
    assert result.probe_status == "fallback"
    assert model.stream_calls == 1
    assert model.invoke_calls == 1


class StreamBadRequestError(RuntimeError):
    status_code = 400


class BadRequestBeforeFirstChunkModel(StreamModel):
    async def astream(self, _messages, config=None):
        self.stream_calls += 1
        raise StreamBadRequestError("stream=true rejected")


@pytest.mark.asyncio
async def test_shadow_falls_back_for_any_error_before_first_chunk() -> None:
    model = BadRequestBeforeFirstChunkModel([])
    config = probe_config("shadow")
    result = await observability_core._ainvoke_model(
        model,
        [HumanMessage(content="question")],
        observability_core.get_trace_recorder(config),
        config,
    )
    assert result.response.content == "ainvoke"
    assert result.probe_status == "fallback"
    assert model.stream_calls == 1
    assert model.invoke_calls == 1


@pytest.mark.asyncio
async def test_enforced_preserves_unrecognized_pre_chunk_error() -> None:
    model = BadRequestBeforeFirstChunkModel([])
    config = probe_config("enforced")
    with pytest.raises(StreamBadRequestError, match="stream=true rejected"):
        await observability_core._ainvoke_model(
            model,
            [HumanMessage(content="question")],
            observability_core.get_trace_recorder(config),
            config,
        )
    assert model.stream_calls == 1
    assert model.invoke_calls == 0


class BrokenAfterFirstChunkModel(StreamModel):
    async def astream(self, _messages, config=None):
        self.stream_calls += 1
        yield observability_core.AIMessageChunk(content="partial")
        raise RuntimeError("stream transport failed")


@pytest.mark.asyncio
async def test_stream_error_after_first_chunk_never_reinvokes() -> None:
    model = BrokenAfterFirstChunkModel([])
    config = probe_config("shadow")
    with pytest.raises(RuntimeError, match="stream transport failed"):
        await observability_core._ainvoke_model(
            model,
            [HumanMessage(content="question")],
            observability_core.get_trace_recorder(config),
            config,
        )
    assert model.stream_calls == 1
    assert model.invoke_calls == 0


class SlowFirstChunkModel(StreamModel):
    async def astream(self, _messages, config=None):
        self.stream_calls += 1
        await asyncio.sleep(0.05)
        yield observability_core.AIMessageChunk(content="late")


@pytest.mark.asyncio
async def test_enforced_first_packet_timeout_counts_once_after_exhaustion() -> None:
    model = SlowFirstChunkModel([])
    config = probe_config(
        "enforced",
        model_circuit_failure_threshold=1,
        model_transport_max_attempts=1,
        model_slow_first_packet_threshold_seconds=0.001,
        model_first_packet_timeout_seconds=0.01,
    )
    with pytest.raises(TimeoutError):
        await invoke_model_with_retry_observability(
            model,
            [HumanMessage(content="question")],
            config,
            span_name="test.circuit.ttft_timeout",
            model_name="openai:slow",
            max_attempts=1,
        )
    configuration = Configuration.from_runnable_config(config)
    breaker = get_model_circuit_registry().get(
        "openai:slow",
        ModelCircuitPolicy(
            failure_threshold=configuration.model_circuit_failure_threshold,
            failure_window_seconds=(
                configuration.model_circuit_failure_window_seconds
            ),
            open_cooldown_seconds=(
                configuration.model_circuit_open_cooldown_seconds
            ),
            max_cooldown_seconds=(
                configuration.model_circuit_max_cooldown_seconds
            ),
            slow_ratio_threshold=(
                configuration.model_circuit_slow_ratio_threshold
            ),
            slow_min_samples=configuration.model_circuit_slow_min_samples,
            first_packet_probe=configuration.model_first_packet_probe,
            slow_first_packet_threshold_seconds=(
                configuration.model_slow_first_packet_threshold_seconds
            ),
        ),
    )
    assert breaker is not None
    snapshot = await breaker.snapshot()
    assert snapshot.state is ModelCircuitState.OPEN
    assert snapshot.failure_count == 1


def test_circuit_public_payload_allowlists_preserve_only_safe_fields() -> None:
    payload = {
        "provider": "openai",
        "model": "gpt-test",
        "from_state": "closed",
        "to_state": "open",
        "reason": "failure_threshold:rate_limited",
        "failure_count": 5,
        "slow_count": 0,
        "sample_count": 0,
        "slow_ratio": 0.0,
        "cooldown_seconds": 60.0,
        "forced_probe": False,
        "authorization": "Bearer secret",
        "messages": ["private"],
    }
    public = sanitize_public_payload("model.circuit_state", payload)
    assert public["to_state"] == "open"
    assert public["failure_count"] == 5
    assert "authorization" not in public
    assert "messages" not in public

    activity = sanitize_task_activity_payload("model.circuit_open", payload)
    assert activity == {
        "provider": "openai",
        "model": "gpt-test",
        "reason": "failure_threshold:rate_limited",
        "failure_count": 5,
        "slow_count": 0,
        "sample_count": 0,
        "cooldown_seconds": 60.0,
    }
