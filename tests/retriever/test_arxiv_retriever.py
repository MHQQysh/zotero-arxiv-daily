"""Tests for ArxivRetriever."""

import time
import os
from types import SimpleNamespace

import feedparser

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    # The RSS fixture gives us paper IDs.  After feedparser, the code calls
    # arxiv.Client().results(search) which makes real HTTP requests.  We mock
    # the arxiv Client so the test stays offline.
    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]
    paper_ids = [e.id.removeprefix("oai:arXiv.org:") for e in new_entries]

    # Build fake ArxivResult-like objects matching each RSS entry
    fake_results = []
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results.append(SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            primary_category=config.source.arxiv.category[0],
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
        ))

    class FakeClient:
        def __init__(self, **kw):
            pass
        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    # Skip file downloads in convert_to_paper
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)
    assert all(p.full_text is None for p in papers)


def test_arxiv_retriever_uses_two_days_ago_date(config, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    captured = {}

    class FixedDatetime(arxiv_retriever.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 15, tzinfo=tz)

    class FakeClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            captured["query"] = search.query
            return iter([])

    monkeypatch.setattr(arxiv_retriever, "datetime", FixedDatetime)
    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert papers == []
    assert "submittedDate:[202607130000 TO 202607132359]" in captured["query"]


def test_convert_to_paper_skips_full_text_downloads(config, monkeypatch):
    def fail_download(*args, **kwargs):
        raise AssertionError("arXiv conversion should not download full text")

    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", fail_download)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", fail_download)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", fail_download)

    raw_paper = SimpleNamespace(
        title="Title only",
        authors=[SimpleNamespace(name="Author A")],
        summary="Abstract only.",
        pdf_url="https://arxiv.org/pdf/2607.00001",
        entry_id="https://arxiv.org/abs/2607.00001",
    )
    paper = ArxivRetriever(config).convert_to_paper(raw_paper)

    assert paper.title == "Title only"
    assert paper.abstract == "Abstract only."
    assert paper.full_text is None


def test_run_with_hard_timeout_returns_value():
    timeout = 10 if os.name == "nt" else 1
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=timeout, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    timeout = 10 if os.name == "nt" else 1
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=timeout, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]
