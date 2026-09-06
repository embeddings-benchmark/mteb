#!/usr/bin/env python3
"""Build the MTVQA multilingual image+text retrieval task for MTEB.

Source is `ByteDance/MTVQA` at a pinned revision (Tang et al. 2024), a text-centric visual
question answering benchmark whose questions are about text appearing inside the image.
It ships an official `test` split, which is the only split used here.

mteb has thirteen `it2t` tasks and none of them is multilingual, so this is the point of
the build: a query that is an image plus a question, in nine languages, against the
answers for that language.

Answers repeat, because different questions about different images can share a short
answer such as a price or a name. The corpus is therefore deduplicated by answer text and
every question with that answer points at the single surviving document. Dropping the
repeats instead would throw away questions that are perfectly well posed.

`qa_pairs` arrives as a Python literal rather than JSON, so it is read with
`ast.literal_eval`; the answers contain apostrophes, which makes quote substitution
followed by a JSON parse unsafe.

Examples:
  # Build the reshaped tables locally.
  uv run python scripts/data/mtvqa_retrieval/create_data.py --stage build

  # Build and publish.
  uv run python scripts/data/mtvqa_retrieval/create_data.py --stage all --push
"""

from __future__ import annotations

import argparse
import ast
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, Image, load_dataset
from huggingface_hub import HfApi

_SOURCE_REPO = "ByteDance/MTVQA"
_SOURCE_REV = "7cdca63ec6cd71cdd0b52f79fc3176bb9ce36a9c"
_TARGET = "vnahata/MTVQA-it2t-retrieval"
_LICENSE = "cc-by-nc-4.0"

# source `lang` value -> published subset name
_LANGUAGES = {
    "AR": "ara",
    "DE": "deu",
    "FR": "fra",
    "IT": "ita",
    "JA": "jpn",
    "KR": "kor",
    "RU": "rus",
    "TH": "tha",
    "VI": "vie",
}

_MIN_ANSWER_CHARS = 1


def stage_build(work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    # decode=False keeps the compressed bytes, so a query holds a small dict rather than a
    # decoded image. Several questions share one image, and decoding each copy exhausts
    # memory partway through the build.
    rows = load_dataset(_SOURCE_REPO, split="test", revision=_SOURCE_REV).cast_column(
        "image", Image(decode=False)
    )

    counts: dict[str, dict[str, int]] = {}
    for source_lang, code in _LANGUAGES.items():
        wanted = [i for i, lang in enumerate(rows["lang"]) if lang == source_lang]
        if not wanted:
            continue

        queries, corpus_ids, qrels = [], {}, []
        corpus_rows = []
        for n, i in enumerate(wanted):
            try:
                pairs = ast.literal_eval(rows[i]["qa_pairs"])
            except (ValueError, SyntaxError):
                continue
            for j, pair in enumerate(pairs):
                question = (pair.get("question") or "").strip()
                answer = (pair.get("answer") or "").strip()
                if not question or len(answer) < _MIN_ANSWER_CHARS:
                    continue
                if answer not in corpus_ids:
                    doc_id = f"{code}-doc-{len(corpus_ids)}"
                    corpus_ids[answer] = doc_id
                    corpus_rows.append(
                        {"id": doc_id, "modality": "text", "text": answer}
                    )
                query_id = f"{code}-q-{n}-{j}"
                queries.append(
                    {
                        "id": query_id,
                        "modality": "image,text",
                        "image": rows[i]["image"],
                        "text": question,
                    }
                )
                qrels.append(
                    {
                        "query-id": query_id,
                        "corpus-id": corpus_ids[answer],
                        "score": 1,
                    }
                )

        Dataset.from_list(queries).cast_column("image", Image()).to_parquet(
            str(work / f"{code}_queries.parquet")
        )
        pq.write_table(
            pa.Table.from_pylist(corpus_rows), work / f"{code}_corpus.parquet"
        )
        pq.write_table(pa.Table.from_pylist(qrels), work / f"{code}_qrels.parquet")
        counts[code] = {
            "queries": len(queries),
            "corpus": len(corpus_rows),
            "images": len(wanted),
        }
        print(f"  {code}: {counts[code]}", flush=True)

    (work / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print("built", json.dumps(counts))
    return counts


def _card(codes: list[str]) -> str:
    lines = [
        "---",
        f"license: {_LICENSE}",
        "task_categories:",
        "  - visual-question-answering",
        "language:",
        *[f"  - {c}" for c in codes],
        "tags:",
        "  - mteb",
        "  - retrieval",
        "  - multilingual",
        "configs:",
    ]
    for c in codes:
        for kind in ("queries", "corpus", "qrels"):
            split = "corpus" if kind == "corpus" else "test"
            lines += [
                f"  - config_name: {c}-{kind}",
                "    data_files:",
                f"      - split: {split}",
                f"        path: {c}_{kind}.parquet",
            ]
    lines += [
        "---",
        "",
        "# MTVQA multilingual image+text retrieval (MTEB)",
        "",
        "A query is an image plus a question about the text inside it; the corpus is the",
        "answers for that language. Nine languages.",
        "",
        f"Source: `{_SOURCE_REPO}` at revision `{_SOURCE_REV[:7]}`, {_LICENSE}, official",
        "`test` split. Answers repeat across questions, so the corpus is deduplicated by",
        "answer text and every question with that answer points at the surviving document.",
        "",
        "Built by `scripts/data/mtvqa_retrieval/create_data.py` in the MTEB repo.",
    ]
    return "\n".join(lines) + "\n"


def stage_push(work: Path) -> None:
    api = HfApi()
    api.create_repo(_TARGET, repo_type="dataset", exist_ok=True)
    codes = [c for c in _LANGUAGES.values() if (work / f"{c}_queries.parquet").exists()]
    for code in codes:
        for kind in ("queries", "corpus", "qrels"):
            api.upload_file(
                path_or_fileobj=str(work / f"{code}_{kind}.parquet"),
                path_in_repo=f"{code}_{kind}.parquet",
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
    parser.add_argument("--work-dir", type=Path, default=Path("mtvqa_work"))
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    print(f"source: {_SOURCE_REPO}@{_SOURCE_REV[:7]} (license {_LICENSE})")
    if args.stage in ("build", "all"):
        stage_build(args.work_dir)
    if args.stage == "push" or (args.push and args.stage == "all"):
        stage_push(args.work_dir)


if __name__ == "__main__":
    main()
