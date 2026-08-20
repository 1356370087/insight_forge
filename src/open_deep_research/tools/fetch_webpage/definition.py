"""Definition and implementation of the legacy webpage fetch tool."""

from __future__ import annotations

from urllib.parse import urljoin, urlsplit

import aiohttp
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import ToolException, tool

from open_deep_research.configuration import Configuration
from open_deep_research.sandbox.policy import allowed_domains, is_enforced_mode
from open_deep_research.security.network import (
    validate_public_http_url,
    validate_response_peer,
)
from open_deep_research.tools.adapters import adapt_langchain_tool
from open_deep_research.tools.availability import legacy_fetch_enabled
from open_deep_research.tools.base import ToolOrigin
from open_deep_research.tools.fetch_webpage.prompt import DESCRIPTION, render_prompt
from open_deep_research.tools.tavily_search import summarization


def _egress_urls(args: dict) -> list[str]:
    url = args.get("url")
    return [url] if isinstance(url, str) else []


@tool("fetch_webpage", description=DESCRIPTION)
async def _fetch_webpage_call(
    url: str,
    max_chars: int = 20_000,
    summarize: bool = True,
    config: RunnableConfig = None,
) -> str:
    """Fetch a known URL and return its optionally summarized text content."""
    configurable = Configuration.from_runnable_config(config)
    max_bytes = configurable.max_external_content_bytes
    requested_chars = max(1, min(int(max_chars), max_bytes))
    timeout = aiohttp.ClientTimeout(total=30)
    current_url = url
    raw = ""
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for redirect_count in range(6):
            try:
                await validate_public_http_url(current_url)
            except ValueError as exc:
                raise ToolException(f"Unsafe webpage URL rejected: {exc}") from exc
            async with session.get(current_url, allow_redirects=False) as response:
                validate_response_peer(response)
                if response.status in {301, 302, 303, 307, 308}:
                    location = getattr(response, "headers", {}).get("Location")
                    if not location or redirect_count >= 5:
                        raise ToolException("fetch_webpage redirect limit exceeded")
                    next_url = urljoin(current_url, location)
                    current_host = urlsplit(current_url).hostname
                    next_host = urlsplit(next_url).hostname
                    if next_host != current_host and is_enforced_mode(configurable):
                        approved = next_host in (
                            set(allowed_domains(configurable))
                            | {
                                str(value).lower()
                                for value in (config or {}).get("metadata", {}).get(
                                    "sandbox_gateway_authorized_hosts", []
                                )
                            }
                        )
                        if not approved:
                            raise ToolException(
                                "Cross-domain redirect target requires separate approval: "
                                f"{next_host}"
                            )
                    current_url = next_url
                    continue
                if response.status >= 400:
                    raise ToolException(
                        f"fetch_webpage got HTTP {response.status} for {current_url}"
                    )
                content_type = str(
                    getattr(response, "headers", {}).get("Content-Type", "")
                ).lower()
                if content_type and not any(
                    allowed in content_type
                    for allowed in (
                        "text/",
                        "application/json",
                        "application/xml",
                        "application/xhtml+xml",
                    )
                ):
                    raise ToolException(
                        f"Unsupported webpage content type: {content_type[:80]}"
                    )
                body = getattr(response, "content", None)
                if body is not None and hasattr(body, "iter_chunked"):
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in body.iter_chunked(16_384):
                        size += len(chunk)
                        if size > max_bytes:
                            raise ToolException(
                                "Webpage response exceeds configured byte limit"
                            )
                        chunks.append(chunk)
                    raw = b"".join(chunks).decode(
                        getattr(response, "charset", None) or "utf-8",
                        errors="replace",
                    )
                else:
                    raw = await response.text(errors="replace")
                    if len(raw.encode("utf-8", errors="replace")) > max_bytes:
                        raise ToolException(
                            "Webpage response exceeds configured byte limit"
                        )
                break
        else:  # pragma: no cover - loop exits through break or redirect error
            raise ToolException("fetch_webpage redirect limit exceeded")

    raw = raw[:requested_chars]
    if not summarize:
        return f"<url>{current_url}</url>\n<content>\n{raw}\n</content>"
    summary = await summarization.summarize_webpage(
        summarization.build_summarization_model(config),
        raw,
        config=config,
        model_name=configurable.summarization_model,
    )
    return f"<url>{current_url}</url>\n{summary}"


fetch_webpage = adapt_langchain_tool(
    _fetch_webpage_call,
    origin=ToolOrigin.SYSTEM,
    retryable=True,
    concurrency_safe=True,
    prompt=render_prompt,
    is_enabled=legacy_fetch_enabled,
    egress_urls=_egress_urls,
)
