from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from datasets import Dataset, DatasetDict, Video
from huggingface_hub import hf_hub_download


SOURCE_REPO = "syn-omni-sony/finecvr"
SOURCE_REVISION = "410365ff0e457c5f8ed2d1814fe68a0ed2d1656f"


def load_annotations() -> list[dict]:
    path = hf_hub_download(
        repo_id=SOURCE_REPO,
        filename="test.json",
        repo_type="dataset",
        revision=SOURCE_REVISION,
    )
    with open(path) as f:
        return json.load(f)


def build_metadata(data: list[dict]):
    corpus_ids = sorted(
        {row["source"] for row in data}
        | {row["target"] for row in data}
    )

    queries = []
    qrels = []
    top_ranked = []

    for i, row in enumerate(data):
        query_id = f"query-{i:05d}"
        source_id = row["source"]
        target_id = row["target"]

        queries.append(
            {
                "id": query_id,
                "text": row["instruct"],
                "source_id": source_id,
                "category": source_id.split("/", 1)[0],
            }
        )

        qrels.append(
            {
                "query-id": query_id,
                "corpus-id": target_id,
                "score": 1,
            }
        )

        # FineCVR evaluates against a global gallery while excluding
        # the query's own reference video from retrieval candidates.
        top_ranked.append(
            {
                "query-id": query_id,
                "corpus-ids": [
                    corpus_id
                    for corpus_id in corpus_ids
                    if corpus_id != source_id
                ],
            }
        )

    return corpus_ids, queries, qrels, top_ranked


def validate(
    corpus_ids: list[str],
    queries: list[dict],
    qrels: list[dict],
    top_ranked: list[dict],
) -> None:
    assert len(corpus_ids) == 2165
    assert len(queries) == 2000
    assert len(qrels) == 2000
    assert len(top_ranked) == 2000

    assert len(set(corpus_ids)) == len(corpus_ids)
    assert len({q["id"] for q in queries}) == len(queries)

    for query, qrel, ranked in zip(queries, qrels, top_ranked):
        assert query["id"] == qrel["query-id"]
        assert query["id"] == ranked["query-id"]
        assert query["source_id"] != qrel["corpus-id"]
        assert qrel["corpus-id"] in corpus_ids
        assert query["source_id"] in corpus_ids
        assert query["source_id"] not in ranked["corpus-ids"]
        assert qrel["corpus-id"] in ranked["corpus-ids"]
        assert len(ranked["corpus-ids"]) == 2164





def pack_frames_to_video(frame_paths: list[Path], output_path: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as f:
        for frame_path in sorted(frame_paths):
            f.write(f"file '{frame_path}'\n")
        f.flush()
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-r", "2",
                "-i", f.name,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )



def pack_all_videos(
    corpus_ids: list[str],
    frames_root: Path,
    video_dir: Path,
) -> None:
    video_dir.mkdir(parents=True, exist_ok=True)

    for clip_id in corpus_ids:
        output_path = video_dir / f"{clip_id.replace('/', '__')}.mp4"
        if output_path.exists():
            continue

        frame_paths = sorted((frames_root / clip_id).glob("*.jpg"))
        if len(frame_paths) != 8:
            raise ValueError(f"{clip_id}: expected 8 frames, got {len(frame_paths)}")

        pack_frames_to_video(frame_paths, output_path)



def build_datasets(
    corpus_ids: list[str],
    queries: list[dict],
    qrels: list[dict],
    top_ranked: list[dict],
    video_dir: Path,
) -> dict[str, Dataset]:
    corpus_ds = Dataset.from_list(
        [
            {
                "id": corpus_id,
                "video": str(video_dir / f"{corpus_id.replace('/', '__')}.mp4"),
            }
            for corpus_id in corpus_ids
        ]
    )

    corpus_ds = corpus_ds.cast_column("video", Video())

    queries_ds = Dataset.from_list(
        [
            {
                "id": query["id"],
                "text": query["text"],
                "video": str(
                    video_dir / f"{query['source_id'].replace('/', '__')}.mp4"
                ),
            }
            for query in queries
        ]
    )

    queries_ds = queries_ds.cast_column("video", Video())

    qrels_ds = Dataset.from_list(qrels)
    top_ranked_ds = Dataset.from_list(top_ranked)

    return {
        "corpus": corpus_ds,
        "queries": queries_ds,
        "qrels": qrels_ds,
        "top_ranked": top_ranked_ds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames-root",
        type=Path,
        required=True,
        help="Path to the extracted finecvr frame root.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        required=True,
        help="Directory for packed FineCVR mp4 files.",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument(
        "--repo-id",
        default="myang333/FineCVRVT2VRetrieval",
    )
    args = parser.parse_args()

    data = load_annotations()
    corpus_ids, queries, qrels, top_ranked = build_metadata(data)

    validate(corpus_ids, queries, qrels, top_ranked)
    pack_all_videos(corpus_ids, args.frames_root, args.video_dir)

    datasets = build_datasets(
        corpus_ids,
        queries,
        qrels,
        top_ranked,
        args.video_dir,
    )

    print(f"corpus: {len(corpus_ids)}")
    print(f"queries: {len(queries)}")
    print(f"qrels: {len(qrels)}")
    print(f"top_ranked: {len(top_ranked)}")
    print(f"candidates/query: {len(top_ranked[0]['corpus-ids'])}")
    print()
    for name, dataset in datasets.items():
        print(f"{name} features:", dataset.features)

    if args.push_to_hub:
        for name, dataset in datasets.items():
            print(f"Pushing {name}...")
            DatasetDict({"test": dataset}).push_to_hub(
                args.repo_id,
                name,
            )

    print()
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
