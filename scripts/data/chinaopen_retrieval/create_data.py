"""Package ChinaOpen-1k into MTEB multilingual video retrieval tasks.

Source: https://huggingface.co/datasets/AIMClab-RUC/ChinaOpen (CC BY-NC-SA 4.0),
the manually annotated 1k test set of ChinaOpen (ACM MM 2023), sourced from
Bilibili.

Why ChinaOpen
-------------
MTEB has no multilingual video retrieval task: every video retrieval task is
English-only. ChinaOpen-1k is annotated the other way around from the usual
translated benchmark: the ``Manual-caption`` field is written in Chinese by
human annotators watching Bilibili videos, and the English field is a
translation of that Chinese caption. The non-English side is therefore the
native one, and the videos themselves are Chinese-web content rather than the
YouTube pool that MSR-VTT, VATEX and DiDeMo all draw from.

Because both languages describe the same 1,092 videos, the two language subsets
are a controlled comparison: any gap between the ``zho-Hans`` and ``eng-Latn``
scores is the model's language handling, not a difference in visual content.
This mirrors how XM3600 is used for images.

Construction
------------
captions  ``Captions.Manual-caption`` (zh) and ``Captions-en.Manual-caption``
          (en). The ``User-title`` fields are ignored: they are uploader-written
          video titles, often clickbait or punning, not descriptions of content.
corpus    the 1,092 mp4 files shipped in the release, unmodified.
qrels     instance level, score 1.

Duplicate captions
------------------
A handful of captions describe more than one video (2 texts in zh, 6 in en, e.g.
"一个人在跑步机上跑步。"). Treating these as instance-level would mark a
correct retrieval wrong, the incomplete-qrels failure mode. Queries are
therefore deduplicated by caption text and every video carrying that caption is
marked relevant, so t2v queries are multi-positive where the source data is
genuinely ambiguous.

Directions
----------
t2v  query = caption, corpus = video  (one repo)
v2t  query = video,   corpus = caption (one repo)

Both repos expose one config triple per language (``<lang>-corpus``,
``<lang>-queries``, ``<lang>-qrels``), which is the layout MTEB's default
multilingual retrieval loader expects, so the tasks need no custom ``load_data``.
The video parquet shards are written once per repo and referenced by both
language configs rather than duplicated.

Usage:
  uv run python scripts/data/chinaopen_retrieval/create_data.py \
      --source work/chinaopen --work work/chinaopen_out
  uv run python scripts/data/chinaopen_retrieval/create_data.py \
      --source work/chinaopen --work work/chinaopen_out \
      --namespace <hf-user> --push
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, Features, Value, Video

LANGS = {"zho-Hans": ("Captions", "zh"), "eng-Latn": ("Captions-en", "en")}
T2V_REPO = "ChinaOpen1k-T2V"
V2T_REPO = "ChinaOpen1k-V2T"


def read_annotations(source: Path) -> dict[str, dict[str, str]]:
    """Return {video_id: {lang: caption}} for videos whose mp4 is present."""
    ann = json.loads((source / "ChinaOpen-1k-annotations.json").read_text())
    out = {}
    for vid, entry in ann.items():
        if not (source / f"{vid}.mp4").exists():
            continue
        out[vid] = {
            lang: entry[field]["Manual-caption"].strip()
            for lang, (field, _) in LANGS.items()
        }
    return out


def build_language_tables(
    captions: dict[str, dict[str, str]], lang: str, direction: str
) -> tuple[Dataset, Dataset]:
    """Build the deduplicated caption table and direction-aware qrels.

    Captions are deduplicated by text, so a caption shared by several videos
    becomes one entry linked to all of them. ``direction`` decides which side
    of that link is the query: ``t2v`` asks caption -> video (multi-positive
    where a caption is shared), ``v2t`` asks video -> caption (always exactly
    one correct caption per video).
    """
    by_text: dict[str, list[str]] = defaultdict(list)
    for vid, caps in captions.items():
        by_text[caps[lang]].append(vid)

    text_ids, texts, pairs = [], [], []
    for i, (text, vids) in enumerate(sorted(by_text.items(), key=lambda kv: kv[1][0])):
        tid = f"q{i}"
        text_ids.append(tid)
        texts.append(text)
        pairs.extend((tid, vid) for vid in sorted(vids))

    text_table = Dataset.from_dict(
        {"id": text_ids, "text": texts},
        features=Features({"id": Value("string"), "text": Value("string")}),
    )
    qrels_ds = Dataset.from_list(
        [
            {
                "query-id": tid if direction == "t2v" else vid,
                "corpus-id": vid if direction == "t2v" else tid,
                "score": 1,
            }
            for tid, vid in pairs
        ],
        features=Features(
            {
                "query-id": Value("string"),
                "corpus-id": Value("string"),
                "score": Value("int32"),
            }
        ),
    )
    return text_table, qrels_ds


def write_videos(
    captions: dict[str, dict[str, str]], source: Path, out: Path, shards: int = 4
) -> int:
    """Write the video table (id, video) as parquet shards under ``out``.

    The mp4 bytes are embedded rather than referenced so the published dataset
    is self-contained, and the rows are sharded to keep each file well under
    the size the Hub is comfortable serving.
    """
    ids = sorted(captions)
    features = Features({"id": Value("string"), "video": Video()})
    out.mkdir(parents=True, exist_ok=True)

    per_shard = -(-len(ids) // shards)
    for shard in range(shards):
        chunk = ids[shard * per_shard : (shard + 1) * per_shard]
        if not chunk:
            continue
        ds = Dataset.from_dict(
            {
                "id": chunk,
                "video": [
                    {
                        "bytes": (source / f"{vid}.mp4").read_bytes(),
                        "path": f"{vid}.mp4",
                    }
                    for vid in chunk
                ],
            },
            features=features,
        )
        ds.to_parquet(str(out / f"test-{shard:05d}-of-{shards:05d}.parquet"))
    return len(ids)


def write_table(ds: Dataset, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(out / "test-00000-of-00001.parquet"))


def card(direction: str, video_dir: str, text_dir_fmt: str) -> str:
    """Dataset card whose YAML maps each language config onto shared files."""
    lines = [
        "---",
        "license: cc-by-nc-sa-4.0",
        "language:",
        "- zh",
        "- en",
        "task_categories:",
        "- video-text-to-text",
        "configs:",
    ]
    for lang, (_, short) in LANGS.items():
        text_dir = text_dir_fmt.format(short=short)
        corpus_dir = video_dir if direction == "t2v" else text_dir
        queries_dir = text_dir if direction == "t2v" else video_dir
        lines += [
            f"- config_name: {lang}-corpus",
            "  data_files:",
            "  - split: test",
            f"    path: {corpus_dir}/test-*.parquet",
            f"- config_name: {lang}-queries",
            "  data_files:",
            "  - split: test",
            f"    path: {queries_dir}/test-*.parquet",
            f"- config_name: {lang}-qrels",
            "  data_files:",
            "  - split: test",
            f"    path: qrels_{short}/test-*.parquet",
        ]
    lines += [
        "---",
        "",
        f"# ChinaOpen-1k {direction.upper()} retrieval",
        "",
        "Multilingual video retrieval built from",
        "[ChinaOpen-1k](https://huggingface.co/datasets/AIMClab-RUC/ChinaOpen):",
        "1,092 Bilibili videos with human-written Chinese captions and their",
        "English translations. Both language configs share the same videos, so",
        "the two subsets form a controlled comparison.",
        "",
        "Built by `scripts/data/chinaopen_retrieval/create_data.py` in",
        "[mteb](https://github.com/embeddings-benchmark/mteb).",
        "",
        "Please cite the original dataset:",
        "",
        "```bibtex",
        "@inproceedings{chen2023chinaopen,",
        "  title = {ChinaOpen: A Dataset for Open-world Multimodal Learning},",
        "  author = {Chen, Aozhu and Wang, Ziyuan and Dong, Chengbo and Tian, Kaibin",
        "            and Zhao, Ruixiang and Liang, Xun and Kang, Zhanhui and Li, Xirong},",
        "  booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},",
        "  year = {2023},",
        "  doi = {10.1145/3581783.3612156},",
        "}",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True, help="extracted ChinaOpen-1k")
    ap.add_argument("--work", type=Path, required=True, help="output directory")
    ap.add_argument("--namespace", type=str, default=None)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    captions = read_annotations(args.source)
    print(f"videos with annotations: {len(captions)}")

    for direction, repo in (("t2v", T2V_REPO), ("v2t", V2T_REPO)):
        root = args.work / repo
        if root.exists():
            shutil.rmtree(root)
        n_videos = write_videos(captions, args.source, root / "videos")
        for lang, (_, short) in LANGS.items():
            texts, qrels = build_language_tables(captions, lang, direction)
            write_table(texts, root / f"texts_{short}")
            write_table(qrels, root / f"qrels_{short}")
            n_multi = len(qrels) - len(texts)
            print(
                f"  {repo} {lang}: {len(texts)} captions, {len(qrels)} qrels "
                f"({n_multi} extra positives from duplicate captions)"
            )
        (root / "README.md").write_text(
            card(direction, "videos", "texts_{short}"), encoding="utf-8"
        )
        print(f"  {repo}: {n_videos} videos -> {root}")

        if args.push:
            if not args.namespace:
                raise SystemExit("--push requires --namespace")
            from huggingface_hub import HfApi

            api = HfApi()
            repo_id = f"{args.namespace}/{repo}"
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
            api.upload_folder(
                folder_path=str(root), repo_id=repo_id, repo_type="dataset"
            )
            print(f"  pushed {repo_id}")


if __name__ == "__main__":
    main()
