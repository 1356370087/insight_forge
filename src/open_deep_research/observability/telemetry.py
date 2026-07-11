"""External observability sinks for Langfuse and Prometheus."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from contextlib import ExitStack
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

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
        self.llm_output_throughput = Histogram(
            "llm_output_tokens_per_second",
            "Completed output tokens divided by request duration.",
            ["provider", "model", "agent_role", "operation"],
            namespace=namespace,
        )
        self.llm_tokens = Counter(
            "llm_tokens_total",
            "LLM tokens reported by providers.",
            ["provider", "model", "agent_role", "direction"],
            namespace=namespace,
        )
        self.llm_cost = Counter(
            "llm_estimated_cost_usd_total",
            "Estimated LLM cost in USD from configured model prices.",
            ["provider", "model", "agent_role"],
            namespace=namespace,
        )
        self.llm_cache_requests = Counter(
            "llm_cache_requests_total",
            "LLM requests with input usage, split by prompt-cache outcome.",
            ["provider", "model", "agent_role", "operation", "cache_status"],
            namespace=namespace,
        )
        self.llm_cache_input_ratio = Histogram(
            "llm_cache_input_ratio",
            "Cached input tokens divided by total input tokens.",
            ["provider", "model", "agent_role", "operation"],
            buckets=(0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1),
            namespace=namespace,
        )
        self.llm_output_input_ratio = Histogram(
            "llm_output_input_token_ratio",
            "Output tokens divided by input tokens for completed LLM requests.",
            ["provider", "model", "agent_role", "operation"],
            buckets=(0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8),
            namespace=namespace,
        )
        self.llm_reasoning_output_ratio = Histogram(
            "llm_reasoning_output_token_ratio",
            "Reasoning tokens divided by output tokens when reported.",
            ["provider", "model", "agent_role", "operation"],
            buckets=(0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1),
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
        self.tool_result_size = Histogram(
            "tool_result_characters",
            "Serialized tool result size in characters.",
            ["tool_name", "agent_role"],
            namespace=namespace,
        )
        self.search_sources = Histogram(
            "search_unique_sources",
            "Unique URLs returned by one search tool call.",
            ["tool_name"],
            namespace=namespace,
        )
        self.tool_empty_results = Counter(
            "tool_empty_results_total",
            "Successful tool calls that returned an empty serialized result.",
            ["tool_name", "agent_role"],
            namespace=namespace,
        )
        self.search_zero_source_calls = Counter(
            "search_zero_source_calls_total",
            "Successful search calls that returned no unique source URL.",
            ["tool_name", "agent_role"],
            namespace=namespace,
        )
        self.agent_steps = Counter(
            "agent_steps_total",
            "Completed agent/runtime stages.",
            ["agent_role", "operation", "status"],
            namespace=namespace,
        )
        self.agent_duration = Histogram(
            "agent_step_duration_seconds",
            "Agent/runtime stage duration, including wait states.",
            ["agent_role", "operation"],
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
        self.terminal_rate_limits = Counter(
            "terminal_rate_limits_total",
            "LLM/tool calls that terminated with a rate-limit error.",
            ["kind", "provider", "model", "agent_role"],
            namespace=namespace,
        )
        self.export_errors = Counter(
            "observability_export_errors_total",
            "Failed writes to an observability backend.",
            ["component", "operation"],
            namespace=namespace,
        )
        self.quality_scores = Histogram(
            "quality_score",
            "Runtime quality and evidence scores.",
            ["score_name", "agent_role"],
            buckets=(0, 1, 2, 3, 4, 5),
            namespace=namespace,
        )
        self.evidence_counts = Histogram(
            "evidence_items",
            "Evidence, source, and validation item counts observed by quality gates.",
            ["metric", "agent_role"],
            buckets=(0, 1, 2, 3, 5, 10, 20, 50, 100),
            namespace=namespace,
        )
        self.report_sources = Histogram(
            "report_sources",
            "Unique sources included in a generated report.",
            buckets=(0, 1, 2, 3, 5, 10, 20, 50, 100),
            namespace=namespace,
        )
        self.report_citations = Histogram(
            "report_citation_markers",
            "Citation markers included in a generated report.",
            buckets=(0, 1, 2, 3, 5, 10, 20, 50, 100, 200),
            namespace=namespace,
        )
        self.report_characters = Histogram(
            "report_characters",
            "Generated report length in characters.",
            buckets=(0, 1000, 2500, 5000, 10000, 20000, 50000, 100000),
            namespace=namespace,
        )
        self.report_sections = Histogram(
            "report_sections",
            "Sections included in a generated report.",
            buckets=(0, 1, 2, 3, 5, 8, 10, 15, 20, 30),
            namespace=namespace,
        )
        self.report_citation_density = Histogram(
            "report_citations_per_1000_characters",
            "Citation markers per 1000 report characters.",
            buckets=(0, 0.25, 0.5, 1, 2, 3, 5, 10, 20),
            namespace=namespace,
        )
        self.report_source_coverage = Histogram(
            "report_cited_source_ratio",
            "Unique numeric citation markers divided by available report sources.",
            buckets=(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1),
            namespace=namespace,
        )
        self.task_events = Counter(
            "research_tasks_total",
            "Research task lifecycle transitions.",
            ["event"],
            namespace=namespace,
        )
        self.task_queue_wait = Histogram(
            "research_task_queue_wait_seconds",
            "Time from task creation until execution starts.",
            namespace=namespace,
        )
        self.task_duration = Histogram(
            "research_task_duration_seconds",
            "End-to-end research task duration by terminal outcome.",
            ["outcome"],
            namespace=namespace,
        )
        self.task_starts = Counter(
            "research_task_starts_total",
            "Research task execution starts split by initial or reassigned attempt.",
            ["attempt_type"],
            namespace=namespace,
        )
        self.task_assignment_attempts = Histogram(
            "research_task_assignment_attempts",
            "Assignment attempts consumed by a terminal research task.",
            ["outcome"],
            buckets=(1, 2, 3, 4, 5, 8, 10),
            namespace=namespace,
        )
        self.pending_tasks = Gauge(
            "research_tasks_pending",
            "Tasks waiting for a teammate.",
            namespace=namespace,
        )
        self.active_tasks = Gauge(
            "research_tasks_active",
            "Tasks currently executing or waiting for approval.",
            namespace=namespace,
        )

    def observe_span(self, span: Any, status: str, duration_seconds: float) -> None:
        """Publish one completed LLM/tool span."""
        role = span.agent_role or "unknown"
        provider = span.provider or "unknown"
        model = span.model or "unknown"
        if span.error_type == "rate_limited":
            self.terminal_rate_limits.labels(span.kind, provider, model, role).inc()
        if span.kind == "llm":
            operation = _operation(span.name)
            self.llm_requests.labels(provider, model, role, operation, status).inc()
            self.llm_duration.labels(provider, model, role, operation).observe(duration_seconds)
            if duration_seconds > 0 and span.usage.output_tokens > 0:
                self.llm_output_throughput.labels(provider, model, role, operation).observe(
                    span.usage.output_tokens / duration_seconds
                )
            self.llm_tokens.labels(provider, model, role, "input").inc(span.usage.input_tokens)
            self.llm_tokens.labels(provider, model, role, "output").inc(span.usage.output_tokens)
            self.llm_tokens.labels(provider, model, role, "cached_input").inc(
                span.usage.cached_input_tokens
            )
            self.llm_tokens.labels(provider, model, role, "reasoning").inc(
                span.usage.reasoning_tokens
            )
            self.llm_cost.labels(provider, model, role).inc(span.usage.estimated_cost_usd)
            if span.usage.input_tokens > 0:
                cache_status = "hit" if span.usage.cached_input_tokens > 0 else "miss"
                cache_labels = (provider, model, role, operation)
                self.llm_cache_requests.labels(*cache_labels, cache_status).inc()
                self.llm_cache_input_ratio.labels(*cache_labels).observe(
                    min(1.0, span.usage.cached_input_tokens / span.usage.input_tokens)
                )
                self.llm_output_input_ratio.labels(*cache_labels).observe(
                    span.usage.output_tokens / span.usage.input_tokens
                )
            if span.usage.output_tokens > 0 and span.usage.reasoning_tokens > 0:
                self.llm_reasoning_output_ratio.labels(
                    provider, model, role, operation
                ).observe(min(1.0, span.usage.reasoning_tokens / span.usage.output_tokens))
        elif span.kind == "tool":
            tool_name = span.attributes.get("tool_name") or span.name.removeprefix("tool.")
            self.tool_calls.labels(str(tool_name), role, status).inc()
            self.tool_duration.labels(str(tool_name), role).observe(duration_seconds)
            self.tool_result_size.labels(str(tool_name), role).observe(
                float(span.attributes.get("result_chars", 0))
            )
            if status == "success" and not span.attributes.get("result_chars", 0):
                self.tool_empty_results.labels(str(tool_name), role).inc()
            if "search" in str(tool_name).lower():
                self.search_sources.labels(str(tool_name)).observe(
                    float(span.attributes.get("source_count", 0))
                )
                if status == "success" and not span.attributes.get("source_count", 0):
                    self.search_zero_source_calls.labels(str(tool_name), role).inc()
        elif span.kind == "agent":
            operation = _operation(span.name)
            self.agent_steps.labels(role, operation, status).inc()
            self.agent_duration.labels(role, operation).observe(duration_seconds)

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

    def observe_export_error(self, component: str, operation: str) -> None:
        """Count a fail-open backend write failure."""
        self.export_errors.labels(component, operation).inc()

    def observe_score(self, name: str, value: Any, agent_role: str) -> None:
        """Publish numeric runtime quality scores."""
        if not isinstance(value, int | float | bool):
            return
        numeric = float(value)
        report_collectors = {
            "report.source_count": self.report_sources,
            "report.citation_marker_count": self.report_citations,
            "report.character_count": self.report_characters,
            "report.section_count": self.report_sections,
            "report.citation_density_per_1k_chars": self.report_citation_density,
            "report.cited_source_ratio": self.report_source_coverage,
        }
        report_collector = report_collectors.get(name)
        if report_collector is not None:
            report_collector.observe(numeric)
            return
        metric = name.rsplit(".", 1)[-1]
        if metric in {"source_count", "error_count", "evidence_result_count"}:
            self.evidence_counts.labels(metric, agent_role).observe(numeric)
            return
        if metric in {
            "accepted",
            "passed",
            "relevance",
            "source_quality",
            "evidence_coverage",
            "corroboration",
            "groundedness",
        }:
            self.quality_scores.labels(name, agent_role).observe(numeric)

    def observe_task_transition(self, record: Any, event: str) -> None:
        """Publish queue and lifecycle metrics for one research task transition."""
        event_name = str(getattr(event, "value", event))
        self.task_events.labels(event_name).inc()
        if event_name == "task.started":
            assignment_attempt = max(1, int(getattr(record, "assignment_attempt", 0)))
            attempt_type = "reassigned" if assignment_attempt > 1 else "initial"
            self.task_starts.labels(attempt_type).inc()
            if record.started_at is not None:
                self.task_queue_wait.observe(max(0.0, record.started_at - record.created_at))
        elif event_name in {"task.completed", "task.failed", "task.timed_out", "task.cancelled"}:
            outcome = event_name.removeprefix("task.")
            self.task_duration.labels(outcome).observe(max(0.0, record.elapsed_seconds))
            self.task_assignment_attempts.labels(outcome).observe(
                max(1, int(getattr(record, "assignment_attempt", 0)))
            )

    def set_task_counts(self, pending: int, active: int) -> None:
        """Set queue gauges from the authoritative task-state store."""
        self.pending_tasks.set(max(0, pending))
        self.active_tasks.set(max(0, active))


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
        trace_context = {
            "trace_id": self.sink.client.create_trace_id(seed=self.span.run_id),
        }
        langfuse_parent_span_id = getattr(
            self.span, "langfuse_parent_span_id", None
        )
        if langfuse_parent_span_id is not None:
            trace_context["parent_span_id"] = langfuse_parent_span_id
        kwargs["trace_context"] = trace_context

        stack = ExitStack()
        try:
            self.observation = stack.enter_context(
                self.sink.client.start_as_current_observation(**kwargs)
            )
            observation_id_getter = getattr(
                self.sink.client, "get_current_observation_id", None
            )
            if observation_id_getter is not None:
                self.span.langfuse_observation_id = observation_id_getter()
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
                        **self.span.attributes,
                        "retry_count": self.span.retry_count,
                        "error_type": self.span.error_type,
                        "http_status": self.span.http_status,
                        "status": getattr(self.span, "final_status", None),
                    },
                }
                if self.span.kind == "llm":
                    usage_details = {
                        "input": self.span.usage.input_tokens,
                        "output": self.span.usage.output_tokens,
                        "total": self.span.usage.total_tokens,
                    }
                    if self.span.usage.cached_input_tokens:
                        usage_details["cached_input"] = self.span.usage.cached_input_tokens
                    if self.span.usage.cache_creation_input_tokens:
                        usage_details["cache_creation_input"] = (
                            self.span.usage.cache_creation_input_tokens
                        )
                    if self.span.usage.reasoning_tokens:
                        usage_details["reasoning"] = self.span.usage.reasoning_tokens
                    update["usage_details"] = usage_details
                    if self.span.usage.estimated_cost_usd > 0:
                        update["cost_details"] = {
                            "total": self.span.usage.estimated_cost_usd,
                        }
                if exc or self.span.error_type:
                    update["level"] = "ERROR"
                    update["status_message"] = (
                        getattr(self.span, "error_message", None)
                        or str(self.span.error_type)
                    )
                elif getattr(self.span, "final_status", None) == "cancelled":
                    update["level"] = "WARNING"
                    update["status_message"] = "cancelled"
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

    def score(
        self,
        span: Any,
        name: str,
        value: float | str | bool,
        comment: str | None = None,
    ) -> None:
        """Attach a score to the exact mirrored observation."""
        data_type = "BOOLEAN" if isinstance(value, bool) else (
            "NUMERIC" if isinstance(value, int | float) else "CATEGORICAL"
        )
        normalized: float | str = float(value) if isinstance(value, bool) else value
        self.client.create_score(
            name=name,
            value=normalized,
            trace_id=self.client.create_trace_id(seed=span.run_id),
            observation_id=span.langfuse_observation_id,
            data_type=data_type,
            comment=comment,
        )

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
