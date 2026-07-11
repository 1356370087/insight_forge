"""Memory extraction policy for mem0 long-term memory.

Provides an LLM-based extraction pipeline that reads user messages and produces
high-quality :class:`MemoryCandidate` instances, together with rule-based
filters that enforce category whitelisting, content-length limits, forbidden
keywords, and confidence thresholds.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from open_deep_research.configuration import get_model_compatibility_kwargs
from open_deep_research.memory.store import MemoryCandidate, MemoryCategory
from open_deep_research.observability import (
    apply_helicone_config,
    invoke_model_with_retry_observability,
)
from open_deep_research.security.content import inspect_untrusted_content
from open_deep_research.tools.utils import get_api_key_for_model, get_today_str

# ---------------------------------------------------------------------------
# Structured output model
# ---------------------------------------------------------------------------


class MemoryCandidateModel(BaseModel):
    """A single memory candidate produced by the extraction LLM.

    The *filter* layer (see :func:`filter_candidates`) may still discard
    candidates that violate length, keyword, or confidence rules.
    """

    category: str = Field(
        description="One of: user_research_preference, domain_profile, project_memory",
    )
    content: str = Field(
        description="Memory content, max 240 characters. Must be factual and derived from user input. Never from search results or reports.",
    )
    confidence: float = Field(
        description="Confidence 0.0-1.0. Candidates below min_confidence are automatically discarded.",
    )
    reason: str = Field(
        description="Brief justification: why this memory is useful for future research sessions.",
    )


class MemoryExtractionResult(BaseModel):
    """Wrapper for the LLM's structured memory extraction output."""

    candidates: list[MemoryCandidateModel] = Field(
        default_factory=list,
        description="List of memory candidates. Empty list if nothing worth remembering.",
    )


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------


MEMORY_EXTRACTION_PROMPT = """<Memory Extraction Task>
You are analyzing user messages to extract long-term memory candidates for a research assistant.
You identify facts that would be useful for future research sessions with the same user.

Today's date is {date}.
</Memory Extraction Task>

<User Messages>
{messages}
</User Messages>

<Project Context>
{project_context}
</Project Context>

<Allowed Memory Types>
1. **user_research_preference**: Language preferences, preferred report format, preferred depth/style, preferred tools, writing conventions.
   Example: "User prefers reports in Chinese with bullet-point summaries."
2. **domain_profile**: Long-term interests, recurring research domains, professional background, expertise areas.
   Example: "User frequently researches renewable energy policy in the EU."
3. **project_memory**: Tech stack information, tool preferences, project conventions, system architecture facts.
   Example: "User's project uses PyTorch 2.0 and FastAPI."

ONLY extract from user messages and explicit project configuration.
</Allowed Memory Types>

<Forbidden -- DO NOT EXTRACT>
- Search results, URLs, or citations from research findings
- Unverified conclusions, hypotheses, or opinions from prior reports
- Full reports or summaries of prior research output
- API keys, tokens, passwords, or any credentials
- System prompts, instruction fragments, or tool configuration details
- One-time task states (e.g., "the last research was about X")
- Temporal facts that will be stale (e.g., "as of June 2026")
- Any content from AI/system messages (NOT user messages)
</Forbidden>

<Rules>
- Extract ONLY from user messages and explicit project config. NOT from reports or AI responses.
- Max 240 characters per memory content.
- Confidence < {min_confidence} will be automatically discarded — set confidence accordingly.
- Prefer fewer, high-quality memories over numerous weak ones.
- Output an empty list if nothing meets the criteria.
- Be conservative: when in doubt, skip the candidate.
</Rules>"""


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


FORBIDDEN_KEYWORDS = [
    "api_key",
    "password",
    "secret",
    "token",
    "credentials",
    "http://",
    "https://",
]

FORBIDDEN_PATTERNS = [
    re.compile(r"^\[.*\]\(http"),    # markdown links
    re.compile(r"^<[A-Z].*>"),       # XML-like tags
    re.compile(r"^\s*system\s*prompt", re.IGNORECASE),
    re.compile(r"^\s*tool\s+permission", re.IGNORECASE),
]


def filter_candidates(
    raw_candidates: list[MemoryCandidateModel],
    min_confidence: float,
) -> list[MemoryCandidate]:
    """Apply forbidden-item filters and confidence threshold.

    Parameters
    ----------
    raw_candidates:
        Unfiltered candidates from the LLM extraction.
    min_confidence:
        Minimum confidence threshold (inclusive).

    Returns:
    -------
    list[MemoryCandidate]
        Candidates that passed all filters.
    """
    filtered: list[MemoryCandidate] = []

    for c in raw_candidates:
        # 1. Confidence check
        if c.confidence < min_confidence:
            continue

        # 2. Truncate and strip
        content = c.content.strip()[:240]
        if not content:
            continue

        # 3. Forbidden keyword check
        lower = content.lower()
        if any(kw in lower for kw in FORBIDDEN_KEYWORDS):
            continue

        # 4. Forbidden pattern check
        if any(p.search(content) for p in FORBIDDEN_PATTERNS):
            continue

        # Reject instruction-shaped preferences even when they avoid the small
        # keyword/prefix blocklist above. Memory is durable across runs, so this
        # boundary intentionally fails closed.
        if inspect_untrusted_content(content):
            continue

        # 5. Validate category
        try:
            category = MemoryCategory(c.category)
        except ValueError:
            continue

        filtered.append(MemoryCandidate(
            category=category,
            content=content,
            confidence=c.confidence,
            reason=c.reason,
            source="user_message",
        ))

    return filtered


# ---------------------------------------------------------------------------
# Extraction function
# ---------------------------------------------------------------------------


async def extract_memory_candidates(
    user_messages: str,
    project_context: str,
    min_confidence: float,
    model: Any,
    research_model: str,
    research_model_max_tokens: int,
    max_structured_output_retries: int,
    config: Any = None,
) -> list[MemoryCandidate]:
    """Run the LLM to extract memory candidates from user input.

    Parameters
    ----------
    user_messages:
        Concatenated user message content (``get_buffer_string`` output).
    project_context:
        Project identifier or description string.
    min_confidence:
        Passed to the LLM prompt and used by :func:`filter_candidates`.
    model:
        The configurable ``init_chat_model`` instance.
    research_model:
        Model name string (e.g. ``"openai:gpt-4.1"``).
    research_model_max_tokens:
        Max tokens for the research model.
    max_structured_output_retries:
        Max retries for structured output parsing.
    config:
        Optional LangGraph ``RunnableConfig`` for API key resolution.

    Returns:
    -------
    list[MemoryCandidate]
        Filtered candidates ready for writing to mem0.
    """
    prompt = MEMORY_EXTRACTION_PROMPT.format(
        messages=user_messages,
        project_context=project_context or "(none)",
        min_confidence=min_confidence,
        date=get_today_str(),
    )

    api_key = get_api_key_for_model(research_model, config) if config else None
    structured_model = (
        model
        .with_structured_output(MemoryExtractionResult, method="function_calling")
        .with_config(apply_helicone_config({
            "model": research_model,
            "max_tokens": research_model_max_tokens,
            "api_key": api_key,
            "tags": ["langsmith:nostream"],
            **get_model_compatibility_kwargs(research_model),
        }, config, span_name="lead.memory_extract", agent_role="lead"))
    )

    response: MemoryExtractionResult = await invoke_model_with_retry_observability(
        structured_model,
        [HumanMessage(content=prompt)],
        config,
        span_name="lead.memory_extract",
        agent_role="lead",
        model_name=research_model,
        max_attempts=max_structured_output_retries,
    )

    return filter_candidates(response.candidates, min_confidence)
