"""Find cited arXiv preprints that have since been published.

Many tasks, benchmarks and models cite a paper as an arXiv preprint although it
has since appeared at a conference or in a journal. This lists those citations,
so they can be updated to cite the published version.

A citation counts as a preprint when it has an arXiv id, no DOI other than
arXiv's own, and no journal or booktitle other than arXiv. They are looked up by
arXiv id in [Semantic Scholar](https://www.semanticscholar.org/product/api),
which links a preprint to its published version. Neither index links every
preprint, so the ones Semantic Scholar knows no published version of are looked
up in [OpenAlex](https://openalex.org) as well. A published DOI is only reported
once Crossref confirms it belongs to a work with the cited title; a venue without
a DOI (e.g. NeurIPS or ICLR) only when the titles match.

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
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from refaudit import Entry, default_resolvers
from refaudit.http import HttpError, TransportError
from refaudit.models import Found
from refaudit.normalize import clean_arxiv_id, clean_doi, similarity

from scripts.tools.check_citations import CONTACT_EMAIL, citations

if TYPE_CHECKING:
    from refaudit.resolvers import Resolver

ALLOWLIST_PATH = Path(__file__).parent / "preprint_allowlist.json"

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_BATCH_SIZE = 500
"""The most ids Semantic Scholar takes in one batch request."""
S2_ATTEMPTS = 6

OPENALEX_BATCH_SIZE = 50
"""arXiv ids per OpenAlex request, which accepts up to 100 values in one filter."""

ARXIV_DOI_PREFIX = "10.48550/arxiv."
PREPRINT_DOI_PREFIXES = (
    "10.48550/",  # arXiv
    "10.1101/",  # bioRxiv, medRxiv
    "10.2139/",  # SSRN
    "10.21203/",  # Research Square
    "10.36227/",  # TechRxiv
)
"""Crossref holds preprint servers' DOIs too, so its check cannot tell them apart."""
PREPRINT_VENUE = re.compile(
    r"arxiv|corr|preprint|biorxiv|medrxiv|ssrn|social science research network"
    r"|research square|techrxiv",
    re.IGNORECASE,
)

TITLE_MATCH = 0.75
"""The similarity refaudit treats as the same work."""


@dataclass(frozen=True)
class Version:
    """A published version of a preprint, as an index reports it."""

    venue: str
    doi: str
    title: str
    url: str
    """Where to get its BibTeX."""


@dataclass(frozen=True)
class Published:
    key: str
    cited_by: str
    arxiv_id: str
    title: str
    year: int | None
    version: Version

    @property
    def published_in(self) -> str:
        parts = (self.version.venue, self.year)
        return " ".join(str(part) for part in parts if part) or "-"


def published_doi(doi: str | None) -> str:
    """The DOI, lowercased, unless it is missing or a preprint server's."""
    doi = clean_doi(doi or "").lower()
    return "" if doi.startswith(PREPRINT_DOI_PREFIXES) else doi


def preprint_arxiv_id(entry: Entry) -> str | None:
    """The arXiv id of an entry that cites only the preprint, else None."""
    arxiv_id = clean_arxiv_id(entry.arxiv_id)
    if not arxiv_id or published_doi(entry.doi):
        return None
    for field in ("journal", "booktitle"):
        venue = entry.get(field)
        if venue and not PREPRINT_VENUE.search(venue):
            return None
    return arxiv_id


def lookup_semantic_scholar(arxiv_ids: list[str]) -> dict[str, Version]:
    """The published version Semantic Scholar knows of each arXiv id."""
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
    return {
        arxiv_id: version
        for arxiv_id, paper in zip(arxiv_ids, papers, strict=True)
        if paper and (version := parse_semantic_scholar_paper(paper))
    }


def parse_semantic_scholar_paper(paper: dict) -> Version | None:
    """The published version in a Semantic Scholar paper, if it has one."""
    external_ids = paper.get("externalIds") or {}
    doi = published_doi(external_ids.get("DOI"))
    venue = paper.get("venue") or ""
    if PREPRINT_VENUE.search(venue):
        venue = ""
    if not venue and not doi:
        return None
    if doi:
        url = f"https://doi.org/{doi}"
    elif acl_id := external_ids.get("ACL"):
        url = f"https://aclanthology.org/{acl_id}"
    else:
        url = f"https://www.semanticscholar.org/paper/{paper['paperId']}"
    return Version(venue, doi, paper.get("title") or "", url)


def lookup_openalex(arxiv_ids: list[str]) -> dict[str, Version]:
    """The published version OpenAlex knows of each arXiv id.

    It is only a fallback, so an outage is reported rather than failing the run.
    """
    # refaudit's client paces, retries and identifies the requests
    (openalex,) = default_resolvers(CONTACT_EMAIL, only=["openalex"])
    versions = {}
    for start in range(0, len(arxiv_ids), OPENALEX_BATCH_SIZE):
        batch = arxiv_ids[start : start + OPENALEX_BATCH_SIZE]
        # OpenAlex lowercases the arXiv URLs, e.g. of math.GT/0309136
        wanted = {i.lower(): i for i in batch}
        query = urllib.parse.urlencode(
            {
                "filter": "doi:" + "|".join(ARXIV_DOI_PREFIX + i for i in batch),
                "select": "doi,display_name,locations",
                "per-page": 100,
                "mailto": CONTACT_EMAIL,
            }
        )
        try:
            works = openalex.http.get(f"{openalex.api_base}?{query}").json()["results"]
        except (HttpError, TransportError, ValueError, KeyError) as e:
            print(f"OpenAlex is unavailable, skipping it: {e}", flush=True)
            break
        for work in works:
            if parsed := parse_openalex_work(work):
                work_arxiv_ids, version = parsed
                versions |= {wanted[i]: version for i in work_arxiv_ids if i in wanted}
    return versions


def parse_openalex_work(work: dict) -> tuple[set[str], Version] | None:
    """The arXiv ids and published version of an OpenAlex work, if it has one.

    OpenAlex merges a preprint with its published version and then gives the work
    the publisher's DOI; the preprint's arXiv id is among its locations.
    """
    doi = published_doi(work.get("doi"))
    if not doi:
        return None
    arxiv_ids, venue = set(), ""
    for location in work.get("locations") or []:
        url = (location.get("landing_page_url") or "").lower()
        if match := re.search(r"arxiv\.org/abs/(.+?)(?:v\d+)?$", url) or re.search(
            re.escape(ARXIV_DOI_PREFIX) + r"(.+)$", url
        ):
            arxiv_ids.add(match.group(1))
        source = location.get("source") or {}
        if (
            not venue
            and location.get("version") == "publishedVersion"
            and source.get("type") != "repository"
        ):
            venue = source.get("display_name") or ""
    version = Version(
        venue, doi, work.get("display_name") or "", f"https://doi.org/{doi}"
    )
    return arxiv_ids, version


def confirm(
    entry: Entry, version: Version, crossref: Resolver
) -> tuple[str, int | None] | None:
    """The title and year of the published version, if it is the cited work.

    Crossref holds the papers publishers register, under the publication year.
    Without a DOI only the index's title can be compared, and its year is the
    preprint's, so none is given.
    """
    if not version.doi:
        if similarity(entry.title, version.title) < TITLE_MATCH:
            return None
        return version.title, None
    outcome = crossref.resolve(
        Entry(entry.key, "misc", {"title": entry.title, "doi": version.doi})
    )
    if not isinstance(outcome, Found):
        return None
    if similarity(entry.title, outcome.record.title) < TITLE_MATCH:
        return None
    return outcome.record.title, outcome.record.year


def find_published(
    entries: dict[str, tuple[Entry, set[str]]], preprints: dict[str, str]
) -> list[Published]:
    arxiv_ids = sorted(set(preprints.values()))
    versions = lookup_semantic_scholar(arxiv_ids)
    missing = [i for i in arxiv_ids if i not in versions]
    print(
        f"Semantic Scholar knows a published version of {len(versions)}, "
        f"asking OpenAlex about the other {len(missing)}",
        flush=True,
    )
    versions |= lookup_openalex(missing)

    (crossref,) = default_resolvers(CONTACT_EMAIL, only=["crossref:doi"])
    published = []
    for key, arxiv_id in sorted(preprints.items()):
        if (version := versions.get(arxiv_id)) is None:
            continue
        entry, cited_by = entries[key]
        if confirmed := confirm(entry, version, crossref):
            title, year = confirmed
            cited_by_names = ", ".join(sorted(cited_by)[:3])
            published.append(
                Published(key, cited_by_names, arxiv_id, title, year, version)
            )
        else:
            print(f"could not confirm {key}: {version.venue or '-'} {version.doi}")
    return published


def write_report(path: Path, published: list[Published]) -> None:
    rows = "\n".join(
        f"| `{p.key}` | {p.cited_by} "
        f"| [{p.arxiv_id}](https://arxiv.org/abs/{p.arxiv_id}) "
        f"| {p.published_in} "
        f"| {p.version.url} |"
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
            f"\n  {p.published_in}: {p.version.url}"
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
