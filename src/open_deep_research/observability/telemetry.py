"""External observability sinks for Langfuse and Prometheus."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from contextlib import ExitStack
from typing import Any

from prometheus_client import Counter, Histogram

from open_deep_research.configuration import Configuration

logger = logging.getLogger(__name__)


def _operation(name: str) -> str:
    """Bound dynamic report-section span names to a stable metric label."""
    if name.startswith("lead.section."):
        return "lead.section"
    return name


class PrometheusMetrics:
    """Record low-cardinality operational metrics from TraceRecorder spans."""

    def __init__(self, namespace: str = "open_deep_research"):
        """Create collectors for one Prometheus namespace."""
        # prometheus_client returns the already-registered collector through its
        # registry only indirectly. Cache instances per namespace in this module.
        self.namespace = namespace
        self.runs = Counter(
            "runs_total",
            "Completed research runs.",
            ["status"],
            namespace=namespace,
        )
        self.run_duration = Histogram(
            "run_duration_seconds",
            "End-to-end research run duration.",
            ["status"],
            namespace=namespace,
        )
        self.llm_requests = Counter(
            "llm_requests_total",
            "Completed LLM requests.",
            ["provider", "model", "agent_role", "operation", "status"],
            namespace=namespace,
        )
        self.llm_duration = Histogram(
            "llm_request_duration_seconds",
            "LLM request duration.",
            ["provider", "model", "agent_role", "operation"],
            namespace=namespace,
        )
        self.llm_tokens = Counter(
            "llm_tokens_total",
            "LLM tokens reported by providers.",
            ["provider", "model", "agent_role", "direction"],
            namespace=namespace,
        )
        self.tool_calls = Counter(
            "tool_calls_total",
            "Completed tool calls.",
            ["tool_name", "agent_role", "status"],
            namespace=namespace,
        )
        self.tool_duration = Histogram(
            "tool_call_duration_seconds",
            "Tool call duration.",
            ["tool_name", "agent_role"],
            namespace=namespace,
        )
        self.retries = Counter(
            "retries_total",
            "Retry attempts by operation and error class.",
            ["kind", "provider", "model", "agent_role", "error_type"],
            namespace=namespace,
        )
        self.rate_limits = Counter(
            "rate_limits_total",
            "Rate-limit retry attempts.",
            ["kind", "provider", "model", "agent_role"],
            namespace=namespace,
        )

    def observe_span(self, span: Any, status: str, duration_seconds: float) -> None:
        """Publish one completed LLM/tool span."""
        role = span.agent_role or "unknown"
        provider = span.provider or "unknown"
        model = span.model or "unknown"
        if span.kind == "llm":
            operation = _operation(span.name)
            self.llm_requests.labels(provider, model, role, operation, status).inc()
            self.llm_duration.labels(provider, model, role, operation).observe(duration_seconds)
            self.llm_tokens.labels(provider, model, role, "input").inc(span.usage.input_tokens)
            self.llm_tokens.labels(provider, model, role, "output").inc(span.usage.output_tokens)
        elif span.kind == "tool":
            tool_name = span.attributes.get("tool_name") or span.name.removeprefix("tool.")
            self.tool_calls.labels(str(tool_name), role, status).inc()
            self.tool_duration.labels(str(tool_name), role).observe(duration_seconds)

    def observe_retry(self, span: Any, error_type: str) -> None:
        """Publish one retry without run/trace identifiers as labels."""
        labels = (
            span.kind,
            span.provider or "unknown",
            span.model or "unknown",
            span.agent_role or "unknown",
        )
        self.retries.labels(*labels, error_type).inc()
        if error_type == "rate_limited":
            self.rate_limits.labels(*labels).inc()

    def observe_run(self, status: str, duration_seconds: float) -> None:
        """Publish one completed top-level run."""
        self.runs.labels(status).inc()
        self.run_duration.labels(status).observe(duration_seconds)


_prometheus_metrics: dict[str, PrometheusMetrics] = {}
_prometheus_lock = threading.Lock()


def get_prometheus_metrics(configuration: Configuration) -> PrometheusMetrics | None:
    """Return a process-wide collector set for the configured namespace."""
    if not configuration.observability_enabled or not configuration.prometheus_enabled:
        return None
    namespace = configuration.prometheus_namespace
    with _prometheus_lock:
        metrics = _prometheus_metrics.get(namespace)
        if metrics is None:
            metrics = PrometheusMetrics(namespace)
            _prometheus_metrics[namespace] = metrics
    return metrics


class LangfuseSpanBridge:
    """Context-managed Langfuse observation mirroring one TraceRecorder span."""

    def __init__(self, sink: LangfuseSink, span: Any):
        """Bind a local SpanContext to its external sink."""
        self.sink = sink
        self.span = span
        self.stack: ExitStack | None = None
        self.observation: Any = None

    def enter(self) -> None:
        """Start and activate the corresponding Langfuse observation."""
        as_type = {"run": "agent", "agent": "agent", "tool": "tool", "llm": "generation"}.get(
            self.span.kind,
            "span",
        )
        kwargs: dict[str, Any] = {
            "as_type": as_type,
            "name": self.span.name,
            "input": self.span.input_preview,
            "metadata": {
                **self.span.attributes,
                "run_id": self.span.run_id,
                "agent_role": self.span.agent_role or "",
                "provider": self.span.provider or "",
            },
        }
        if as_type == "generation":
            kwargs["model"] = self.span.model
        if self.span.parent_span_id is None:
            kwargs["trace_context"] = {
                "trace_id": self.sink.client.create_trace_id(seed=self.span.run_id),
            }

        stack = ExitStack()
        try:
            self.observation = stack.enter_context(
                self.sink.client.start_as_current_observation(**kwargs)
            )
            if self.span.kind == "run":
                stack.enter_context(
                    self.sink.propagate_attributes(
                        trace_name=self.span.name,
                        user_id=self.sink.user_id,
                        session_id=self.sink.session_id,
                        metadata={"run_id": self.span.run_id},
                    )
                )
        except Exception:
            stack.close()
            raise
        self.stack = stack

    def exit(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        """Update and close the mirrored observation."""
        try:
            if self.observation is not None:
                update: dict[str, Any] = {
                    "output": self.span.output_preview,
                    "metadata": {
                        "retry_count": self.span.retry_count,
                        "error_type": self.span.error_type,
                        "http_status": self.span.http_status,
                    },
                }
                if self.span.kind == "llm":
                    update["usage_details"] = {
                        "input": self.span.usage.input_tokens,
                        "output": self.span.usage.output_tokens,
                        "total": self.span.usage.total_tokens,
                    }
                if exc or self.span.error_type:
                    update["level"] = "ERROR"
                    update["status_message"] = str(exc or self.span.error_type)
                self.observation.update(**update)
        finally:
            if self.stack is not None:
                self.stack.__exit__(exc_type, exc, tb)


class LangfuseSink:
    """Fail-open Langfuse SDK adapter used by TraceRecorder."""

    def __init__(self, configuration: Configuration, runtime_config: Mapping[str, Any]):
        """Initialize the Langfuse singleton client from project configuration."""
        from langfuse import Langfuse, propagate_attributes

        self.configuration = configuration
        self.propagate_attributes = propagate_attributes
        self.client = Langfuse(
            public_key=configuration.langfuse_public_key,
            secret_key=configuration.langfuse_secret_key,
            base_url=configuration.langfuse_base_url,
            environment=configuration.langfuse_environment,
            release=configuration.langfuse_release,
            sample_rate=configuration.langfuse_sample_rate,
        )
        metadata = runtime_config.get("metadata", {})
        configurable = runtime_config.get("configurable", {})
        self.user_id = metadata.get("user_id") or metadata.get("owner") or configurable.get("user_id")
        self.session_id = configurable.get("thread_id") or metadata.get("run_id")

    def span(self, span: Any) -> LangfuseSpanBridge:
        """Build a bridge for a local span."""
        return LangfuseSpanBridge(self, span)

    def callback_handler(self) -> Any:
        """Create a LangChain callback that inherits the active Langfuse context."""
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()

    def flush(self) -> None:
        """Flush queued observations for short-lived processes."""
        self.client.flush()


def create_langfuse_sink(
    configuration: Configuration,
    runtime_config: Mapping[str, Any],
) -> LangfuseSink | None:
    """Create an enabled Langfuse sink, failing open on SDK/configuration errors."""
    if not configuration.observability_enabled or not configuration.langfuse_enabled:
        return None
    if not configuration.langfuse_public_key or not configuration.langfuse_secret_key:
        logger.warning("Langfuse is enabled but LANGFUSE_PUBLIC_KEY/SECRET_KEY are missing")
        return None
    try:
        return LangfuseSink(configuration, runtime_config)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse initialization failed: %s", exc)
        return None


def monotonic_time() -> float:
    """Return a monotonic timestamp for duration metrics."""
    return time.monotonic()
