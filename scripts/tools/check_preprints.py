"""Find cited arXiv preprints that have since been published.

Many tasks, benchmarks and models cite a paper as an arXiv preprint although it
has since appeared at a conference or in a journal. This lists those citations,
so they can be updated to cite the published version.

A citation counts as a preprint when it has an arXiv id, no DOI other than
arXiv's own, and no journal or booktitle other than arXiv. They are looked up by
arXiv id in [Semantic Scholar](https://www.semanticscholar.org/product/api),
which links a preprint to its published version. A published DOI is only
reported once Crossref confirms it belongs to a work with the cited title; a
venue without a DOI (e.g. NeurIPS or ICLR) only when the titles match.

Run it with `make preprint-check`, or directly:

    python -m scripts.tools.check_preprints [--report preprints.md]

Set `SEMANTIC_SCHOLAR_API_KEY` to use mteb's API key; without one the requests go
to a shared pool that is often refused. It exits with 1 when it finds a published
preprint. It runs monthly in CI, which opens an issue listing them, and not on
pull requests: like the citation check it depends on external services, which
should not block a PR.

A preprint that should stay cited as one - e.g. a technical report that was only
published in a shorter workshop version - is listed in `preprint_allowlist.json`
with the reason.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from refaudit import Entry, default_resolvers
from refaudit.models import Found
from refaudit.normalize import clean_arxiv_id, clean_doi, similarity

from tests.test_validate_metadata.check_citations import CONTACT_EMAIL, citations

if TYPE_CHECKING:
    from refaudit.resolvers import Resolver

ALLOWLIST_PATH = Path(__file__).parent / "preprint_allowlist.json"

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_BATCH_SIZE = 500
"""The most ids Semantic Scholar takes in one batch request."""
S2_ATTEMPTS = 6

ARXIV_DOI_PREFIX = "10.48550/arxiv."
PREPRINT_VENUE = re.compile(
    r"arxiv|corr|preprint|biorxiv|medrxiv|ssrn|social science research network"
    r"|research square|techrxiv",
    re.IGNORECASE,
)

TITLE_MATCH = 0.75
"""The similarity refaudit treats as the same work."""


@dataclass(frozen=True)
class Published:
    key: str
    cited_by: str
    arxiv_id: str
    title: str
    venue: str
    year: int | None
    url: str

    @property
    def published_in(self) -> str:
        return " ".join(str(part) for part in (self.venue, self.year) if part) or "-"


def preprint_arxiv_id(entry: Entry) -> str | None:
    """The arXiv id of an entry that cites only the preprint, else None."""
    arxiv_id = clean_arxiv_id(entry.arxiv_id)
    if not arxiv_id:
        return None
    doi = clean_doi(entry.doi)
    if doi and not doi.lower().startswith(ARXIV_DOI_PREFIX):
        return None
    for field in ("journal", "booktitle"):
        venue = entry.get(field)
        if venue and not PREPRINT_VENUE.search(venue):
            return None
    return arxiv_id


def semantic_scholar_papers(arxiv_ids: list[str]) -> list[dict | None]:
    """The Semantic Scholar paper for each arXiv id, None where it has none."""
    headers = {"User-Agent": f"mteb citation check (mailto:{CONTACT_EMAIL})"}
    if api_key := os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
        headers["x-api-key"] = api_key
    papers: list[dict | None] = []
    for start in range(0, len(arxiv_ids), S2_BATCH_SIZE):
        batch = arxiv_ids[start : start + S2_BATCH_SIZE]
        # the API allows one request per second; back off exponentially when refused
        for attempt in range(S2_ATTEMPTS):
            time.sleep(2**attempt)
            try:
                response = requests.post(
                    S2_BATCH_URL,
                    params={"fields": "title,venue,externalIds"},
                    json={"ids": [f"arXiv:{i}" for i in batch]},
                    headers=headers,
                    timeout=120,
                )
            except (requests.ConnectionError, requests.Timeout):
                continue
            if response.status_code != 429 and response.status_code < 500:
                response.raise_for_status()
                papers += response.json()
                break
        else:
            raise RuntimeError(
                f"Semantic Scholar refused {S2_ATTEMPTS} attempts"
                + ("" if api_key else "; set SEMANTIC_SCHOLAR_API_KEY")
            )
    return papers


def published_version(paper: dict) -> tuple[str, str] | None:
    """The venue and DOI of the published version, if Semantic Scholar knows one."""
    doi = clean_doi((paper.get("externalIds") or {}).get("DOI") or "").lower()
    if doi.startswith(ARXIV_DOI_PREFIX):
        doi = ""
    venue = paper.get("venue") or ""
    if PREPRINT_VENUE.search(venue):
        venue = ""
    return (venue, doi) if venue or doi else None


def confirm(
    entry: Entry, paper: dict, doi: str, crossref: Resolver
) -> tuple[str, int | None] | None:
    """The title and year of the published version, if it is the cited work.

    Crossref holds the papers publishers register, under the publication year.
    Without a DOI only Semantic Scholar's title can be compared, and its year is
    the preprint's, so none is given.
    """
    if not doi:
        if similarity(entry.title, paper["title"] or "") < TITLE_MATCH:
            return None
        return paper["title"], None
    outcome = crossref.resolve(
        Entry(entry.key, "misc", {"title": entry.title, "doi": doi})
    )
    if not isinstance(outcome, Found):
        return None
    if similarity(entry.title, outcome.record.title) < TITLE_MATCH:
        return None
    return outcome.record.title, outcome.record.year


def url(paper: dict, doi: str) -> str:
    """Where to get the published version's BibTeX."""
    if doi:
        return f"https://doi.org/{doi}"
    if acl_id := (paper.get("externalIds") or {}).get("ACL"):
        return f"https://aclanthology.org/{acl_id}"
    return f"https://www.semanticscholar.org/paper/{paper['paperId']}"


def find_published(
    entries: dict[str, tuple[Entry, set[str]]], preprints: dict[str, str]
) -> list[Published]:
    keys = sorted(preprints)
    papers = semantic_scholar_papers([preprints[key] for key in keys])
    print(
        f"{sum(p is not None for p in papers)} found, confirming the published "
        "versions",
        flush=True,
    )

    (crossref,) = default_resolvers(CONTACT_EMAIL, only=["crossref:doi"])
    published = []
    for key, paper in zip(keys, papers, strict=True):
        if paper is None or (version := published_version(paper)) is None:
            continue
        venue, doi = version
        entry, cited_by = entries[key]
        if confirmed := confirm(entry, paper, doi, crossref):
            title, year = confirmed
            published.append(
                Published(
                    key,
                    ", ".join(sorted(cited_by)[:3]),
                    preprints[key],
                    title,
                    venue,
                    year,
                    url(paper, doi),
                )
            )
        else:
            print(f"could not confirm {key}: {venue or '-'} {doi or ''}")
    return published


def write_report(path: Path, published: list[Published]) -> None:
    rows = "\n".join(
        f"| `{p.key}` | {p.cited_by} "
        f"| [{p.arxiv_id}](https://arxiv.org/abs/{p.arxiv_id}) "
        f"| {p.published_in} "
        f"| {p.url} |"
        for p in published
    )
    path.write_text(
        f"{len(published)} works cited as arXiv preprints have since been "
        "published. Please update their citations to the published version, "
        "or add the key to `scripts/tools/preprint_allowlist.json` "
        "with the reason it should stay on the preprint.\n\n"
        "| Citation | Cited by | arXiv | Published in | Link |\n"
        "| --- | --- | --- | --- | --- |\n"
        f"{rows}\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        help="also write the published preprints to this markdown file, if there are any",
    )
    args = parser.parse_args()

    entries = citations()
    allowlist = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    preprints = {
        key: arxiv_id
        for key, (entry, _) in entries.items()
        if key not in allowlist and (arxiv_id := preprint_arxiv_id(entry))
    }
    print(
        f"looking up {len(preprints)} citations of arXiv preprints in Semantic Scholar",
        flush=True,
    )
    published = find_published(entries, preprints)
    if not published:
        print("\nno cited preprint has a published version")
        return 0

    for p in published:
        print(
            f"\n{p.key} (arXiv {p.arxiv_id}) cited by {p.cited_by}"
            f"\n  published: {p.title}"
            f"\n  {p.published_in}: {p.url}"
        )
    print(
        f"\n{len(published)} cited preprints have been published. Please cite the "
        "published version, e.g. the BibTeX from the DOI or the ACL Anthology. If a "
        "citation should stay on the preprint, add its key to "
        f"{ALLOWLIST_PATH.name} with the reason."
    )
    if args.report:
        write_report(args.report, published)
    return 1


if __name__ == "__main__":
    sys.exit(main())
