"""Unit tests for reference rendering and source extraction."""

from __future__ import annotations

from open_deep_research.report.models import SourceRef
from open_deep_research.report.profiles import ReferenceStyle
from open_deep_research.report.references import (
    dedupe_sources,
    parse_sources_from_text,
    render_references,
    replace_sources_section,
)


def test_parse_sources_from_text_handles_all_patterns():
    text = (
        "--- SOURCE 1: First Source ---\n"
        "URL: https://first.example.com/a\n\n"
        "Some narrative with a [Markdown Link](https://second.example.com/b) in it.\n\n"
        "### Sources\n"
        "[1] Third Source: https://third.example.com/c\n"
        "https://fourth.example.com/d bare url\n"
    )
    refs = parse_sources_from_text(text)
    urls = [r.url for r in refs]
    assert urls == [
        "https://first.example.com/a",
        "https://second.example.com/b",
        "https://third.example.com/c",
        "https://fourth.example.com/d",
    ]
    # Source-block title wins for the first entry
    assert refs[0].title == "First Source"
    # markdown-link title captured
    assert refs[1].title == "Markdown Link"


def test_parse_sources_dedupes_by_url():
    text = "[A](https://x.io) and again [B](https://x.io)"
    refs = parse_sources_from_text(text)
    assert len(refs) == 1
    assert refs[0].url == "https://x.io"


def test_parse_sources_empty_for_no_urls():
    assert parse_sources_from_text("") == []
    assert parse_sources_from_text("just prose, no links here") == []


def test_dedupe_sources_normalizes_mixed_input():
    mixed = [
        SourceRef(title="One", url="https://a.io"),
        {"title": "Two", "url": "https://b.io"},
        "https://a.io",  # dup of first
    ]
    refs = dedupe_sources(mixed)
    assert [r.url for r in refs] == ["https://a.io", "https://b.io"]
    assert refs[0].title == "One"


def test_render_references_numbered():
    sources = [
        SourceRef(title="Alpha", url="https://a.io"),
        SourceRef(title="Beta", url="https://b.io"),
    ]
    out = render_references(sources, ReferenceStyle.NUMBERED)
    assert out == "[1] Alpha: https://a.io\n[2] Beta: https://b.io"


def test_render_references_bibtex_like():
    sources = [SourceRef(title="Alpha", url="https://a.io")]
    out = render_references(sources, ReferenceStyle.BIBTEX_LIKE)
    assert out.startswith("@misc{ref1,")
    assert "title = {Alpha}" in out
    assert "howpublished = {https://a.io}" in out


def test_render_references_empty_when_no_sources():
    assert render_references([], ReferenceStyle.NUMBERED) == ""
    assert render_references([], ReferenceStyle.BIBTEX_LIKE) == ""


def test_replace_sources_section_replaces_existing():
    md = "# Title\n\nBody.\n\n### Sources\n[1] Old: https://old.io\n"
    out = replace_sources_section(md, "[1] New: https://new.io")
    assert "### Sources\n[1] New: https://new.io" in out
    assert "Old" not in out
    # title + body preserved
    assert out.startswith("# Title")
    assert "Body." in out


def test_replace_sources_section_appends_when_absent():
    md = "# Title\n\nBody with [x](https://x.io)."
    out = replace_sources_section(md, "[1] X: https://x.io")
    assert "### Sources\n[1] X: https://x.io" in out
    assert "Body with" in out


def test_replace_sources_section_noop_when_empty_block():
    md = "# Title\n\n### Sources\n[1] Old: https://old.io"
    assert replace_sources_section(md, "") == md
