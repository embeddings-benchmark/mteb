"""Package ChinaOpen-1k into a single combined MTEB multilingual video dataset.

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
qrels     instance level, score 1, plus the duplicate-caption multi-positive
          links described below.

Duplicate captions
------------------
A handful of captions describe more than one video (2 texts in zh, 6 in en, e.g.
"一个人在跑步机上跑步。"). Treating these as instance-level would mark a
correct retrieval wrong, the incomplete-qrels failure mode. Captions are
therefore deduplicated by text, and every video carrying that caption is linked
to it, so whichever side ends up as the t2v query is multi-positive where the
source data is genuinely ambiguous.

Layout (one repo, shared video config)
---------------------------------------
Earlier drafts of this script wrote two full repos (one per direction), each
with the video bytes duplicated across both language configs -- 4x total. This
version writes ONE repo with:

  ``videos``       id, video           -- the 1,092 clips, stored once and
                                           shared by both languages and both
                                           directions.
  ``<lang>-texts``  id, text            -- the deduplicated captions for that
                                           language (``id`` is a synthetic
                                           ``q{n}``, unrelated to the video id).
  ``<lang>-links``  text-id, video-id   -- the caption<->video pairs. Every
                                           row is one positive; a caption
                                           shared by several videos produces
                                           several rows with the same
                                           ``text-id``.

Direction (t2v: query=caption, corpus=video; v2t: query=video, corpus=caption)
is applied at task load time, not at data-build time: both mteb tasks read the
same ``videos``/``<lang>-texts``/``<lang>-links`` configs and a shared
``_load_chinaopen(task, direction)`` loader flips which side of each link
becomes the query. See ``mteb/tasks/retrieval/multilingual/chinaopen_retrieval.py``.

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
REPO = "ChinaOpen1k"


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
    captions: dict[str, dict[str, str]], lang: str
) -> tuple[Dataset, Dataset]:
    """Build the deduplicated caption table and its caption<->video link table.

    Captions are deduplicated by text, so a caption shared by several videos
    becomes one entry linked to all of them via ``links``. Direction (which
    side of a link is the query) is decided later, in the task's
    ``load_data``, not here -- so this table serves both ChinaOpenT2VRetrieval
    and ChinaOpenV2TRetrieval.
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
    links_ds = Dataset.from_list(
        [{"text-id": tid, "video-id": vid} for tid, vid in pairs],
        features=Features({"text-id": Value("string"), "video-id": Value("string")}),
    )
    return text_table, links_ds


def write_videos(
    captions: dict[str, dict[str, str]], source: Path, out: Path, shards: int = 4
) -> int:
    """Write the shared video table (id, video) as parquet shards under ``out``.

    The mp4 bytes are embedded rather than referenced so the published dataset
    is self-contained, and the rows are sharded to keep each file well under
    the size the Hub is comfortable serving. Written once for the whole repo.
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


def card(video_dir: str, text_dir_fmt: str, links_dir_fmt: str) -> str:
    """Dataset card whose YAML maps each config onto its files.

    Both directions (t2v/v2t) are served from these same configs -- the mteb
    tasks assemble queries/corpus/qrels from them at load time.
    """
    lines = [
        "---",
        "license: cc-by-nc-sa-4.0",
        "language:",
        "- zh",
        "- en",
        "task_categories:",
        "- video-text-to-text",
        "configs:",
        "- config_name: videos",
        "  data_files:",
        "  - split: test",
        f"    path: {video_dir}/test-*.parquet",
    ]
    for lang, (_, short) in LANGS.items():
        text_dir = text_dir_fmt.format(short=short)
        links_dir = links_dir_fmt.format(short=short)
        lines += [
            f"- config_name: {lang}-texts",
            "  data_files:",
            "  - split: test",
            f"    path: {text_dir}/test-*.parquet",
            f"- config_name: {lang}-links",
            "  data_files:",
            "  - split: test",
            f"    path: {links_dir}/test-*.parquet",
        ]
    lines += [
        "---",
        "",
        "# ChinaOpen-1k multilingual video retrieval",
        "",
        "Multilingual video retrieval built from",
        "[ChinaOpen-1k](https://huggingface.co/datasets/AIMClab-RUC/ChinaOpen):",
        "1,092 Bilibili videos with human-written Chinese captions and their",
        "English translations. Both language configs share the same `videos`",
        "config, so the two subsets form a controlled comparison. Captions are",
        "deduplicated by text; the `<lang>-links` config records every",
        "caption<->video pair, so a caption shared by more than one video is",
        "multi-positive rather than incorrectly scored.",
        "",
        "Used by `ChinaOpenT2VRetrieval` and `ChinaOpenV2TRetrieval` in",
        "[mteb](https://github.com/embeddings-benchmark/mteb), which apply the",
        "direction (which side of a link is the query) at load time via a",
        "shared loader, so both tasks read the same configs.",
        "",
        "Built by `scripts/data/chinaopen_retrieval/create_data.py` in mteb.",
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

    root = args.work / REPO
    if root.exists():
        shutil.rmtree(root)

    n_videos = write_videos(captions, args.source, root / "videos")
    print(f"  videos: {n_videos} -> {root / 'videos'}")

    for lang, (_, short) in LANGS.items():
        texts, links = build_language_tables(captions, lang)
        write_table(texts, root / f"texts_{short}")
        write_table(links, root / f"links_{short}")
        n_multi = len(links) - len(texts)
        print(
            f"  {lang}: {len(texts)} captions, {len(links)} links "
            f"({n_multi} extra positives from duplicate captions)"
        )

    (root / "README.md").write_text(
        card("videos", "texts_{short}", "links_{short}"), encoding="utf-8"
    )

    if args.push:
        if not args.namespace:
            raise SystemExit("--push requires --namespace")
        from huggingface_hub import HfApi

        api = HfApi()
        repo_id = f"{args.namespace}/{REPO}"
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=str(root), repo_id=repo_id, repo_type="dataset")
        print(f"  pushed {repo_id}")


if __name__ == "__main__":
    main()
