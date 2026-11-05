"""Find cited arXiv preprints that have since been published, and cite the published version.

Many tasks, benchmarks and models cite a paper as an arXiv preprint although it
has since appeared at a conference or in a journal. This finds those citations
and, with `--update`, replaces them with the published version.

A citation counts as a preprint when it has an arXiv id, no DOI other than
arXiv's own, and no journal or booktitle other than arXiv. They are looked up by
arXiv id in [Semantic Scholar](https://www.semanticscholar.org/product/api),
which links a preprint to its published version. Neither index links every
preprint, so the ones Semantic Scholar knows no published version of are looked
up in [OpenAlex](https://openalex.org) as well.

Only findings we can be sure of are reported: the published version must have a
DOI, and Crossref must hold that DOI under the cited title and first author. A
venue alone (e.g. ICLR or NeurIPS without a DOI) is not enough: indexes label
some preprints with a venue they never appeared at.

With `--update`, each citation is replaced by the BibTeX from the ACL Anthology
or doi.org, keeping its key, and `reference` links to its arXiv version point to
the published one. A citation is only replaced when that BibTeX parses and
carries the cited title and first author; the others are listed as skipped.

Run it with `make preprint-check`, or directly:

    python -m scripts.tools.check_preprints [--update] [--report preprints.md]

Set `SEMANTIC_SCHOLAR_API_KEY` to use mteb's API key; without one the requests go
to a shared pool that is often refused. It exits with 1 when it finds a published
preprint. It runs monthly in CI, which opens a pull request with the updates, and
not on pull requests: like the citation check it depends on external services,
which should not block a PR.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import bibtexparser
import requests
from bibtexparser.model import Field
from refaudit import Entry, default_resolvers, parse_string
from refaudit.http import HttpError, TransportError
from refaudit.models import Found
from refaudit.normalize import (
    clean_arxiv_id,
    clean_doi,
    first_surname,
    similarity,
    surnames_match,
)

from scripts.tools.check_citations import CONTACT_EMAIL, citations
from scripts.tools.format_citations import format_bibtex

if TYPE_CHECKING:
    from refaudit.resolvers import Resolver

REPO = Path(__file__).parents[2]
USER_AGENT = f"mteb citation check (mailto:{CONTACT_EMAIL})"

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_BATCH_SIZE = 500
"""The most ids Semantic Scholar takes in one batch request."""
S2_ATTEMPTS = 6

OPENALEX_BATCH_SIZE = 50
"""arXiv ids per OpenAlex request, which accepts up to 100 values in one filter."""

ARXIV_DOI_PREFIX = "10.48550/arxiv."
ACL_DOI_PREFIX = "10.18653/v1/"
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

SAME_TITLE = 0.9
"""How similar two titles must be to count as the same work."""

DROPPED_FIELDS = {"abstract", "cdate", "crossref"}
"""Fields of the fetched BibTeX we do not keep; DBLP's internal ones among them."""


@dataclass(frozen=True)
class Version:
    """A published version of a preprint, as an index reports it."""

    venue: str
    doi: str


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

    @property
    def link(self) -> str:
        return f"https://doi.org/{self.version.doi}"


def published_doi(doi: str | None) -> str:
    """The DOI, lowercased, unless it is missing or a preprint server's."""
    doi = clean_doi(doi or "").lower()
    return "" if doi.startswith(PREPRINT_DOI_PREFIXES) else doi


def same_first_author(a: str, b: str) -> bool:
    """Whether two BibTeX author fields start with the same person; unknown is no."""
    first_a, first_b = first_surname(a), first_surname(b)
    return bool(first_a and first_b and surnames_match(first_a, first_b))


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
    headers = {"User-Agent": USER_AGENT}
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
                    params={"fields": "venue,externalIds"},
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
    """The published version in a Semantic Scholar paper, if it has a DOI."""
    external_ids = paper.get("externalIds") or {}
    doi = published_doi(external_ids.get("DOI"))
    if not doi and (acl_id := external_ids.get("ACL")):
        # the ACL Anthology registers its papers under this prefix
        doi = published_doi(ACL_DOI_PREFIX + acl_id)
    if not doi:
        return None
    venue = paper.get("venue") or ""
    return Version("" if PREPRINT_VENUE.search(venue) else venue, doi)


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
                "select": "doi,locations",
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
    return arxiv_ids, Version(venue, doi)


def confirm(
    entry: Entry, version: Version, crossref: Resolver
) -> tuple[str, int | None] | None:
    """The title and year of the published version, if it is the cited work.

    Crossref holds the papers publishers register: the DOI must be among them,
    under the cited title and first author.
    """
    outcome = crossref.resolve(
        Entry(entry.key, "misc", {"title": entry.title, "doi": version.doi})
    )
    if not isinstance(outcome, Found):
        return None
    record = outcome.record
    if similarity(entry.title, record.title) < SAME_TITLE:
        return None
    if not same_first_author(entry.get("author"), record.first_author_surname):
        return None
    return record.title, record.year


def find_published(
    entries: dict[str, tuple[Entry, set[str]]], preprints: dict[str, str]
) -> list[Published]:
    arxiv_ids = sorted(set(preprints.values()))
    versions = lookup_semantic_scholar(arxiv_ids)
    missing = [i for i in arxiv_ids if i not in versions]
    print(
        f"Semantic Scholar knows a published DOI of {len(versions)}, "
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
            print(f"could not confirm {key}: {version.doi}")
    return published


def fetch_bibtex(doi: str) -> str | None:
    """The published version's BibTeX, from the ACL Anthology or doi.org."""
    urls = []
    if doi.startswith(ACL_DOI_PREFIX):
        acl_id = doi[len(ACL_DOI_PREFIX) :]
        if re.fullmatch(r"[a-z]\d{2}-\d{4}", acl_id):  # older ids are D19-1454 etc.
            acl_id = acl_id[0].upper() + acl_id[1:]
        urls.append((f"https://aclanthology.org/{acl_id}.bib", {}))
    urls.append((f"https://doi.org/{doi}", {"Accept": "application/x-bibtex"}))
    for url, headers in urls:
        try:
            response = requests.get(
                url, headers={"User-Agent": USER_AGENT, **headers}, timeout=60
            )
        except (requests.ConnectionError, requests.Timeout):
            continue
        if response.ok and response.text.lstrip().startswith("@"):
            return response.text.strip()
    return None


def published_entry(bibtex: str, cited: Entry) -> str | None:
    """The fetched BibTeX under the cited key, if it is the cited work."""
    # the source's own key can trip the parser, e.g. d{\'e}fossez2023high
    bibtex = re.sub(r"^\s*@(\w+)\s*\{[^,]*,", r"@\1{placeholder,", bibtex, count=1)
    library = bibtexparser.parse_string(bibtex)
    if not library.entries or library.failed_blocks:
        return None
    entry = library.entries[0]
    fields = {f.key.lower(): " ".join(str(f.value).split()) for f in entry.fields}
    title = fields.get("title", "").replace("{", "").replace("}", "")
    if similarity(cited.title, title) < SAME_TITLE:
        return None
    if not same_first_author(cited.get("author"), fields.get("author", "")):
        return None
    if "dx.doi.org/" in fields.get("url", ""):
        fields["url"] = "https://doi.org/" + fields["url"].split("dx.doi.org/", 1)[1]
    entry.key = cited.key
    entry.fields = [
        Field(name, value)
        for name, value in sorted(fields.items())
        if name not in DROPPED_FIELDS
    ]
    for block in list(library.blocks):
        if block is not entry:
            library.remove(block)
    return format_bibtex(bibtexparser.write_string(library))


def replace_citation(key: str, new_entry: str, arxiv_id: str, url: str) -> int:
    """Replace the citations of `key` in mteb, and arXiv `reference` links to them.

    Only entries citing the same arXiv paper are replaced: a key is sometimes
    reused for another work, e.g. a paper and its follow-up.
    """
    entry_start = re.compile(r"@\w+\s*\{\s*" + re.escape(key) + r"\s*,")
    arxiv_reference = re.compile(
        r'reference="https?://(?:www\.)?arxiv\.org/(?:abs|pdf|html)/'
        + re.escape(arxiv_id)
        + r'(?:v\d+)?(?:\.pdf)?/?"',
        re.IGNORECASE,
    )
    replaced = 0
    for path in (REPO / "mteb").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        spans = [
            (m.start(), m.start() + citation_length(text[m.start() :]))
            for m in entry_start.finditer(text)
        ]
        spans = [
            (start, end) for start, end in spans if cites(text[start:end], arxiv_id)
        ]
        if not spans:
            continue
        for start, end in reversed(spans):
            opening = text.rfind('"""', 0, start)
            # a plain string needs its backslashes escaped to hold the same text
            raw = opening > 0 and text[opening - 1] in "rR"
            text = (
                text[:start]
                + (new_entry if raw else new_entry.replace("\\", "\\\\"))
                + text[end:]
            )
        text = arxiv_reference.sub(f'reference="{url}"', text)
        path.write_text(text, encoding="utf-8")
        replaced += len(spans)
    return replaced


def cites(bibtex: str, arxiv_id: str) -> bool:
    """Whether the BibTeX entry cites the given arXiv paper."""
    entries = parse_string(bibtex.replace("\\\\", "\\"))
    return (
        bool(entries)
        and clean_arxiv_id(entries[0].arxiv_id).lower() == arxiv_id.lower()
    )


def citation_length(text: str) -> int:
    """The length of the BibTeX entry `text` starts with."""
    depth = 0
    for i, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("unbalanced braces in a citation")


def update_citations(
    published: list[Published], entries: dict[str, tuple[Entry, set[str]]]
) -> tuple[list[Published], list[Published]]:
    """Cite the published version where its BibTeX is the cited work."""
    updated, skipped = [], []
    for p in published:
        bibtex = fetch_bibtex(p.version.doi)
        new_entry = bibtex and published_entry(bibtex, entries[p.key][0])
        if not new_entry:
            skipped.append(p)
            continue
        url = bibtexparser.parse_string(new_entry).entries[0].fields_dict.get("url")
        acl_url = url.value if url and "aclanthology.org" in url.value else None
        replace_citation(p.key, new_entry, p.arxiv_id, acl_url or p.link)
        updated.append(p)
    return updated, skipped


def table(published: list[Published]) -> str:
    return (
        "| Citation | Cited by | arXiv | Published in | DOI |\n| --- | --- | --- | --- | --- |\n"
        + "\n".join(
            f"| `{p.key}` | {p.cited_by} "
            f"| [{p.arxiv_id}](https://arxiv.org/abs/{p.arxiv_id}) "
            f"| {p.published_in} | {p.link} |"
            for p in published
        )
    )


def write_report(
    path: Path, updated: list[Published], skipped: list[Published], update: bool
) -> None:
    if not update:
        text = (
            f"{len(skipped)} works cited as arXiv preprints have since been "
            "published.\n\n" + table(skipped)
        )
    else:
        text = (
            f"Cites the published version of {len(updated)} works that were cited "
            "as arXiv preprints. Each one's DOI is confirmed by Crossref under the "
            "cited title and first author, and its BibTeX comes from the ACL "
            "Anthology or doi.org.\n\n" + table(updated)
        )
        if skipped:
            text += (
                f"\n\n{len(skipped)} more have been published, but the BibTeX "
                "found for them does not carry the cited title and first author, "
                "so they are left as they are:\n\n" + table(skipped)
            )
    path.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="replace the citations with the published version",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="also write what was found to this markdown file, if anything was",
    )
    args = parser.parse_args()

    entries = citations()
    preprints = {
        key: arxiv_id
        for key, (entry, _) in entries.items()
        if (arxiv_id := preprint_arxiv_id(entry))
    }
    print(
        f"looking up {len(preprints)} citations of arXiv preprints in Semantic Scholar",
        flush=True,
    )
    published = find_published(entries, preprints)
    if not published:
        print("\nno cited preprint has a published version")
        return 0

    if args.update:
        updated, skipped = update_citations(published, entries)
    else:
        updated, skipped = [], published
    for p in published:
        status = "updated" if p in updated else "found"
        print(
            f"\n{p.key} (arXiv {p.arxiv_id}) cited by {p.cited_by}: {status}"
            f"\n  published: {p.title}"
            f"\n  {p.published_in}: {p.link}"
        )
    print(
        f"\n{len(published)} cited preprints have been published"
        + (f", {len(updated)} updated" if args.update else "")
        + "."
    )
    if args.report:
        write_report(args.report, updated, skipped, args.update)
    return 1


if __name__ == "__main__":
    sys.exit(main())
