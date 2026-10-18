"""Find cited arXiv preprints that have since been published.

Many tasks, benchmarks and models cite a paper as an arXiv preprint although it
has since appeared at a conference or in a journal. This lists those citations,
so they can be updated to cite the published version.

A citation counts as a preprint when it has an arXiv id, no DOI other than
arXiv's own, and no journal or booktitle other than arXiv. Each one is looked up
in [OpenAlex](https://openalex.org), which merges a preprint with its published
version and then gives the work the publisher's DOI. OpenAlex sometimes merges
the wrong works, so a DOI is only reported once Crossref confirms it belongs to
a work with the same title. Publishers register papers with Crossref; DataCite
holds e.g. the recorded conference talk, which is not what to cite.

Run it with `make preprint-check`, or directly:

    python tests/test_validate_metadata/check_preprints.py [--report preprints.md]

It exits with 1 when it finds a published preprint. It runs monthly in CI, which
opens an issue listing them, and not on pull requests: like the citation check
it depends on external services, which should not block a PR.

A preprint that should stay cited as one - e.g. a technical report that was only
published in a shorter workshop version - is listed in `preprint_allowlist.json`
with the reason.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from check_citations import CONTACT_EMAIL, citations
from refaudit import Entry, default_resolvers
from refaudit.models import Found
from refaudit.normalize import clean_arxiv_id, clean_doi, similarity

ALLOWLIST_PATH = Path(__file__).parent / "preprint_allowlist.json"

OPENALEX_BATCH = 50
"""arXiv ids per OpenAlex request, which accepts up to 100 values in one filter."""

ARXIV_DOI_PREFIX = "10.48550/arxiv."
PREPRINT_DOI_PREFIXES = (
    "10.48550/",  # arXiv
    "10.1101/",  # bioRxiv, medRxiv
    "10.2139/",  # SSRN
    "10.21203/",  # Research Square
    "10.36227/",  # TechRxiv
    "10.20944/",  # Preprints.org
    "10.31219/",  # OSF Preprints
)
"""A work whose DOI is another preprint server's has not been published either."""

PREPRINT_VENUE = re.compile(r"arxiv|corr|preprint", re.IGNORECASE)

TITLE_MATCH = 0.75
"""The similarity refaudit treats as the same work."""


@dataclass(frozen=True)
class Published:
    key: str
    cited_by: str
    arxiv_id: str
    title: str
    doi: str
    venue: str
    year: int | None


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


def openalex_works(arxiv_ids: list[str]) -> dict[str, dict]:
    """The OpenAlex work holding each arXiv id, by lowercased arXiv id."""
    # refaudit's client paces, retries and identifies the requests the way it does
    # for its own OpenAlex lookups
    (openalex,) = default_resolvers(CONTACT_EMAIL, only=["openalex"])
    works = {}
    for start in range(0, len(arxiv_ids), OPENALEX_BATCH):
        batch = arxiv_ids[start : start + OPENALEX_BATCH]
        query = urllib.parse.urlencode(
            {
                "filter": "doi:" + "|".join(ARXIV_DOI_PREFIX + i for i in batch),
                "select": "doi,locations",
                "per-page": 100,
                "mailto": CONTACT_EMAIL,
            }
        )
        data = openalex.http.get(f"{openalex.api_base}?{query}").json()
        # a merged work carries the publisher's DOI, so find the arXiv id in its locations
        for work in data["results"]:
            for location in work["locations"]:
                url = (location.get("landing_page_url") or "").lower()
                match = re.search(r"arxiv\.org/abs/(.+?)(v\d+)?$", url) or re.search(
                    re.escape(ARXIV_DOI_PREFIX) + r"(.+)$", url
                )
                if match:
                    works[match.group(1)] = work
    return works


def published_doi(work: dict) -> tuple[str, str] | None:
    """The DOI and venue of the published version, if OpenAlex knows one."""
    doi = clean_doi(work.get("doi") or "").lower()
    if not doi or doi.startswith(PREPRINT_DOI_PREFIXES):
        return None
    venue = next(
        (
            source["display_name"]
            for location in work["locations"]
            if location.get("version") == "publishedVersion"
            and (source := location.get("source"))
            and source.get("type") != "repository"
        ),
        "",
    )
    return doi, venue


def confirm(
    candidates: dict[str, tuple[str, str, str]],
    entries: dict[str, tuple[Entry, set[str]]],
) -> tuple[list[Published], list[str]]:
    """The candidates whose DOI Crossref holds under the cited title, and the DOIs it does not hold."""
    (crossref,) = default_resolvers(CONTACT_EMAIL, only=["crossref:doi"])
    published, unconfirmed = [], []
    for key, (arxiv_id, doi, venue) in sorted(candidates.items()):
        entry, cited_by = entries[key]
        outcome = crossref.resolve(
            Entry(key, "misc", {"title": entry.title, "doi": doi})
        )
        if not isinstance(outcome, Found):
            unconfirmed.append(f"{key} ({doi})")
        elif similarity(entry.title, outcome.record.title) >= TITLE_MATCH:
            published.append(
                Published(
                    key,
                    ", ".join(sorted(cited_by)[:3]),
                    arxiv_id,
                    outcome.record.title,
                    doi,
                    venue,
                    outcome.record.year,
                )
            )
        # else OpenAlex merged the preprint with a different work
    return published, unconfirmed


def write_report(path: Path, published: list[Published]) -> None:
    rows = "\n".join(
        f"| `{p.key}` | {p.cited_by} "
        f"| [{p.arxiv_id}](https://arxiv.org/abs/{p.arxiv_id}) "
        f"| {p.venue or '-'} {p.year or ''} "
        f"| [{p.doi}](https://doi.org/{p.doi}) |"
        for p in published
    )
    path.write_text(
        f"{len(published)} works cited as arXiv preprints have since been "
        "published. Please update their citations to the published version, "
        "or add the key to `tests/test_validate_metadata/preprint_allowlist.json` "
        "with the reason it should stay on the preprint.\n\n"
        "| Citation | Cited by | arXiv | Published in | DOI |\n"
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
        f"looking up {len(preprints)} citations of arXiv preprints in OpenAlex",
        flush=True,
    )
    works = openalex_works(sorted({i.lower() for i in preprints.values()}))

    candidates = {}
    for key, arxiv_id in preprints.items():
        work = works.get(arxiv_id.lower())
        if work and (found := published_doi(work)):
            candidates[key] = (arxiv_id, *found)
    print(
        f"{len(works)} found, {len(candidates)} with a published DOI, "
        "confirming those with Crossref",
        flush=True,
    )

    published, unconfirmed = confirm(candidates, entries)
    if unconfirmed:
        print(
            f"\nCrossref does not hold {len(unconfirmed)} of the DOIs, skipping them: "
            + ", ".join(unconfirmed)
        )
    if not published:
        print("\nno cited preprint has a published version")
        return 0

    for p in published:
        print(
            f"\n{p.key} (arXiv {p.arxiv_id}) cited by {p.cited_by}"
            f"\n  published: {p.title}"
            f"\n  {p.venue or 'venue unknown'}, {p.year or 'year unknown'}: "
            f"https://doi.org/{p.doi}"
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
