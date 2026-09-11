#!/usr/bin/env python3
"""Build the Spoken Wikipedia speech-text retrieval tasks for MTEB.

Source is Wikimedia Commons, where volunteers read Wikipedia articles aloud and upload the
recording, and the matching article text from each Wikipedia. Everything on Commons is
free by site policy, and the article text is CC-BY-SA, so the published set is CC-BY-SA
4.0. Nothing here is scraped from outside Wikimedia.

The pairing is "which article is this a recording of", which stays true no matter how much
of the recording is kept. Recordings run to tens of minutes, so only the opening is kept:
readers start at the article lead, so the opening and the lead describe the same thing.

Finding the article for a recording takes two passes, because neither route is reliable
alone. First the file's global usage is read and the first main-namespace page on the
matching Wikipedia is taken; that misses files no longer embedded in their article, and
project pages such as `Portal:Gesprochene_Wikipedia` have to be filtered out. Where that
finds nothing, the article is parsed out of the filename, whose shape differs per language
(`En-Title-article.oga`, `FR-Title.ogg`, `ES-Title-article.ogg`).

Articles whose lead is too short to identify are dropped, as are duplicates, since one
article read twice would be relevant to only one of its recordings.

Examples:
  # Build the reshaped tables locally.
  uv run python scripts/data/spoken_wikipedia/create_data.py --stage build

  # Build and publish.
  uv run python scripts/data/spoken_wikipedia/create_data.py --stage all --push
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
from math import gcd
from pathlib import Path

import requests
import soundfile as sf
from datasets import Audio, Dataset
from huggingface_hub import HfApi
from scipy.signal import resample_poly

_TARGET = "vnahata/SpokenWikipedia-retrieval"
_LICENSE = "cc-by-sa-4.0"
_UA = "mteb-dataset-builder/1.0 (https://github.com/embeddings-benchmark/mteb)"
_COMMONS = "https://commons.wikimedia.org/w/api.php"

# published code -> (Commons category language word, wiki subdomain, filename prefixes)
_LANGUAGES = {
    "nld": ("Dutch", "nl", ["Nl-"]),
    "eng": ("English", "en", ["En-"]),
    "deu": ("German", "de", ["De-"]),
    "spa": ("Spanish", "es", ["ES-", "Es-"]),
    "fra": ("French", "fr", ["FR-", "Fr-"]),
}

_PER_LANGUAGE = 150
_CLIP_SECONDS = 60
_MIN_LEAD_CHARS = 200
_TARGET_RATE = 16000


def _get(url: str, params: dict) -> dict:
    for attempt in range(4):
        try:
            r = requests.get(
                url,
                params={**params, "format": "json"},
                headers={"User-Agent": _UA},
                timeout=60,
            )
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    return {}


def _category_files(category: str) -> list[str]:
    out, cont = [], {}
    while True:
        d = _get(
            _COMMONS,
            {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmtype": "file",
                "cmlimit": 500,
                **cont,
            },
        )
        out += [m["title"] for m in d.get("query", {}).get("categorymembers", [])]
        if "continue" not in d:
            return out
        cont = {"cmcontinue": d["continue"]["cmcontinue"]}


def _article_titles(files: list[str], wiki: str, prefixes: list[str]) -> dict[str, str]:
    """Map Commons file title to a main-namespace article on `wiki`."""
    found: dict[str, str] = {}
    for i in range(0, len(files), 40):
        batch = files[i : i + 40]
        d = _get(
            _COMMONS,
            {
                "action": "query",
                "titles": "|".join(batch),
                "prop": "globalusage",
                "gulimit": 50,
            },
        )
        for page in (d.get("query") or {}).get("pages", {}).values():
            for use in page.get("globalusage", []):
                # main namespace only; project and portal pages carry a colon
                if use["wiki"] == f"{wiki}.wikipedia.org" and ":" not in use["title"]:
                    found[page["title"]] = (use["title"].replace("_", " "), "usage")
                    break
        time.sleep(0.3)

    for f in files:
        if f in found:
            continue
        stem = re.sub(
            r"\.(oga|ogg|wav|mp3|flac)$", "", f.replace("File:", ""), flags=re.I
        )
        for p in prefixes:
            if stem.startswith(p):
                stem = stem[len(p) :]
                break
        # a missing prefix is not disqualifying; some uploads are named after the article
        # alone. A guess that is not a real article is dropped later, when its lead fails
        # to resolve.
        stem = re.sub(r"\s*-\s*article$", "", stem, flags=re.I).strip()
        if stem:
            found[f] = (stem, "filename")
    return found


def _leads(titles: list[str], wiki: str) -> dict[str, str]:
    """Fetch the lead section of each article."""
    out: dict[str, str] = {}
    api = f"https://{wiki}.wikipedia.org/w/api.php"
    for i in range(0, len(titles), 20):
        d = _get(
            api,
            {
                "action": "query",
                "titles": "|".join(titles[i : i + 20]),
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "redirects": 1,
            },
        )
        for page in (d.get("query") or {}).get("pages", {}).values():
            text = (page.get("extract") or "").strip()
            if len(text) >= _MIN_LEAD_CHARS:
                out[page["title"]] = " ".join(text.split())
        time.sleep(0.3)
    return out


def _clip(url: str) -> bytes | None:
    r = requests.get(url, headers={"User-Agent": _UA}, timeout=180)
    r.raise_for_status()
    data, rate = sf.read(io.BytesIO(r.content), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    data = data[: int(_CLIP_SECONDS * rate)]
    if len(data) < rate * 5:
        return None
    if rate != _TARGET_RATE:
        factor = gcd(int(rate), _TARGET_RATE)
        data = resample_poly(data, _TARGET_RATE // factor, int(rate) // factor)
    buf = io.BytesIO()
    sf.write(buf, data, _TARGET_RATE, format="OGG", subtype="OPUS")
    return buf.getvalue()


def stage_build(work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for code, (category_word, wiki, prefixes) in _LANGUAGES.items():
        out_path = work / f"{code}.parquet"
        if out_path.exists():
            import pyarrow.parquet as pq

            counts[code] = pq.read_metadata(out_path).num_rows
            print(f"  {code}: cached {counts[code]}", flush=True)
            continue

        files = _category_files(f"Spoken {category_word} Wikipedia")
        mapping = _article_titles(files, wiki, prefixes)
        leads = _leads(sorted({t for t, _ in mapping.values()}), wiki)
        by_usage = sum(1 for _, route in mapping.values() if route == "usage")
        print(
            f"  {code}: {len(files)} files, {len(mapping)} mapped "
            f"({by_usage} via usage), {len(leads)} leads",
            flush=True,
        )

        rows, seen_article, routes = [], set(), {"usage": 0, "filename": 0}
        for f, (article, route) in mapping.items():
            if len(rows) >= _PER_LANGUAGE:
                break
            lead = leads.get(article)
            if not lead or article in seen_article:
                continue
            d = _get(
                _COMMONS,
                {"action": "query", "titles": f, "prop": "imageinfo", "iiprop": "url"},
            )
            page = next(iter((d.get("query") or {}).get("pages", {}).values()), {})
            info = (page.get("imageinfo") or [{}])[0]
            if not info.get("url"):
                continue
            try:
                audio = _clip(info["url"])
            except Exception:
                continue
            if audio is None:
                continue
            seen_article.add(article)
            routes[route] += 1
            rows.append(
                {
                    "id": f"{code}-{len(rows)}",
                    "audio": {"bytes": audio, "path": None},
                    "text": lead,
                }
            )
            time.sleep(0.2)

        Dataset.from_list(rows).cast_column(
            "audio", Audio(sampling_rate=_TARGET_RATE)
        ).to_parquet(str(out_path))
        counts[code] = len(rows)
        print(f"  {code}: {len(rows)} pairs, routes {routes}", flush=True)

    (work / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print("built", json.dumps(counts))
    return counts


def _card(codes: list[str]) -> str:
    lines = [
        "---",
        f"license: {_LICENSE}",
        "task_categories:",
        "  - automatic-speech-recognition",
        "language:",
        *[f"  - {c}" for c in codes],
        "tags:",
        "  - mteb",
        "  - retrieval",
        "  - multilingual",
        "configs:",
    ]
    for c in codes:
        lines += [
            f"  - config_name: {c}",
            "    data_files:",
            "      - split: test",
            f"        path: {c}.parquet",
        ]
    lines += [
        "---",
        "",
        "# Spoken Wikipedia speech-text retrieval (MTEB)",
        "",
        "Volunteer readings of Wikipedia articles paired with the article lead, in Dutch,",
        "English, German, Spanish and French.",
        "",
        "Recordings come from Wikimedia Commons, which is free by site policy, and the lead",
        f"text from each Wikipedia, which is CC-BY-SA. The set is published as {_LICENSE}.",
        f"Only the first {_CLIP_SECONDS} seconds of each reading is kept, since readers start",
        "at the lead. One recording per article.",
        "",
        "Built by `scripts/data/spoken_wikipedia/create_data.py` in the MTEB repo.",
    ]
    return "\n".join(lines) + "\n"


def stage_push(work: Path) -> None:
    api = HfApi()
    api.create_repo(_TARGET, repo_type="dataset", exist_ok=True)
    codes = sorted(p.stem for p in work.glob("*.parquet"))
    for code in codes:
        api.upload_file(
            path_or_fileobj=str(work / f"{code}.parquet"),
            path_in_repo=f"{code}.parquet",
            repo_id=_TARGET,
            repo_type="dataset",
        )
        print(f"  pushed {code}", flush=True)
    api.upload_file(
        path_or_fileobj=io.BytesIO(_card(codes).encode()),
        path_in_repo="README.md",
        repo_id=_TARGET,
        repo_type="dataset",
    )
    print(f"pushed {_TARGET}: {len(codes)} languages")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["build", "push", "all"], default="all")
    parser.add_argument("--work-dir", type=Path, default=Path("spokenwiki_work"))
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    print(f"source: Wikimedia Commons and Wikipedia (license {_LICENSE})")
    if args.stage in ("build", "all"):
        stage_build(args.work_dir)
    if args.stage == "push" or (args.push and args.stage == "all"):
        stage_push(args.work_dir)


if __name__ == "__main__":
    main()
