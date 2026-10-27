"""Offline tests of how scripts/tools/check_preprints.py decides a preprint was published.

The check itself runs monthly against Semantic Scholar, OpenAlex and Crossref, so a
mistake in these decisions would otherwise only show up as a wrong report.
"""

from __future__ import annotations

import pytest

# refaudit is in the optional `citations` dependency group
pytest.importorskip("refaudit")

from refaudit import Entry
from refaudit.models import Found, NotFound, Record

from scripts.tools.check_preprints import (
    Version,
    confirm,
    parse_openalex_work,
    parse_semantic_scholar_paper,
    preprint_arxiv_id,
)

MTEB_TITLE = "MTEB: Massive Text Embedding Benchmark"
MTEB_DOI = "10.18653/v1/2023.eacl-main.148"


class FakeCrossref:
    """Answers every DOI lookup with the same outcome."""

    def __init__(self, outcome: Found | NotFound) -> None:
        self.outcome = outcome

    def resolve(self, entry: Entry) -> Found | NotFound:
        return self.outcome


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"eprint": "2210.07316"}, "2210.07316"),
        ({"eprint": "2210.07316", "doi": "10.48550/arXiv.2210.07316"}, "2210.07316"),
        (
            {"eprint": "2210.07316", "journal": "arXiv preprint arXiv:2210.07316"},
            "2210.07316",
        ),
        ({"eprint": "2210.07316", "booktitle": "EACL"}, None),
        ({"eprint": "2210.07316", "doi": MTEB_DOI}, None),
        ({"title": MTEB_TITLE}, None),
    ],
)
def test_preprint_arxiv_id(fields: dict[str, str], expected: str | None):
    assert preprint_arxiv_id(Entry("mteb", "misc", fields)) == expected


def test_semantic_scholar_paper_with_published_doi():
    version = parse_semantic_scholar_paper(
        {
            "paperId": "p",
            "title": MTEB_TITLE,
            "venue": "EACL",
            "externalIds": {"DOI": MTEB_DOI},
        }
    )
    assert version == Version(
        "EACL", MTEB_DOI, MTEB_TITLE, f"https://doi.org/{MTEB_DOI}"
    )


def test_semantic_scholar_paper_only_on_arxiv():
    paper = {
        "paperId": "p",
        "title": MTEB_TITLE,
        "venue": "arXiv.org",
        "externalIds": {"DOI": "10.48550/arXiv.2210.07316"},
    }
    assert parse_semantic_scholar_paper(paper) is None


def test_semantic_scholar_paper_with_venue_and_acl_id():
    version = parse_semantic_scholar_paper(
        {
            "paperId": "p",
            "title": "FollowIR",
            "venue": "NAACL",
            "externalIds": {"ACL": "2025.naacl-long.597"},
        }
    )
    assert version is not None
    assert not version.doi
    assert version.url == "https://aclanthology.org/2025.naacl-long.597"


def test_openalex_work_merged_with_published_version():
    work = {
        "doi": f"https://doi.org/{MTEB_DOI}",
        "display_name": MTEB_TITLE,
        "locations": [
            {
                "landing_page_url": "http://arxiv.org/abs/2210.07316v3",
                "version": "submittedVersion",
                "source": {"type": "repository", "display_name": "arXiv"},
            },
            {
                "landing_page_url": f"https://doi.org/{MTEB_DOI}",
                "version": "publishedVersion",
                "source": {"type": "conference", "display_name": "EACL"},
            },
        ],
    }
    assert parse_openalex_work(work) == (
        {"2210.07316"},
        Version("EACL", MTEB_DOI, MTEB_TITLE, f"https://doi.org/{MTEB_DOI}"),
    )


@pytest.mark.parametrize(
    "doi", ["10.48550/arxiv.2210.07316", "10.1101/2020.01.01.123456", None]
)
def test_openalex_work_without_published_doi(doi: str | None):
    assert (
        parse_openalex_work({"doi": doi and f"https://doi.org/{doi}", "locations": []})
        is None
    )


def test_confirm_accepts_matching_crossref_title():
    crossref = FakeCrossref(Found(Record("crossref:doi", MTEB_TITLE, year=2023)))
    version = Version("EACL", MTEB_DOI, MTEB_TITLE, f"https://doi.org/{MTEB_DOI}")
    entry = Entry("mteb", "misc", {"title": MTEB_TITLE})
    assert confirm(entry, version, crossref) == (MTEB_TITLE, 2023)


def test_confirm_rejects_doi_of_a_different_work():
    # OpenAlex once merged the Codex paper with an unrelated LIPIcs paper
    crossref = FakeCrossref(
        Found(
            Record(
                "crossref:doi", "Software Engineering Education Revisited", year=2026
            )
        )
    )
    version = Version("", "10.4230/lipics.esem.2026.34", "", "")
    entry = Entry(
        "codex", "misc", {"title": "Evaluating Large Language Models Trained on Code"}
    )
    assert confirm(entry, version, crossref) is None


def test_confirm_rejects_doi_crossref_does_not_hold():
    # e.g. the recorded conference talk, which DataCite holds
    crossref = FakeCrossref(NotFound("crossref:doi"))
    version = Version("", "10.17023/rhe8-g867", "", "")
    entry = Entry(
        "wav2clip",
        "misc",
        {"title": "Wav2CLIP: Learning Robust Audio Representations From CLIP"},
    )
    assert confirm(entry, version, crossref) is None


def test_confirm_without_doi_compares_titles():
    entry = Entry(
        "followir",
        "misc",
        {"title": "FollowIR: Evaluating and Teaching Information Retrieval Models"},
    )
    crossref = FakeCrossref(NotFound("crossref:doi"))
    same = Version(
        "NAACL",
        "",
        "FollowIR: Evaluating and Teaching Information Retrieval Models",
        "",
    )
    other = Version("NAACL", "", "A Different Paper Entirely", "")
    assert confirm(entry, same, crossref) == (same.title, None)
    assert confirm(entry, other, crossref) is None
