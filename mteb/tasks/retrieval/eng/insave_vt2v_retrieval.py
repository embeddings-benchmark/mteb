from __future__ import annotations

import csv
import hashlib
import logging
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import Dataset, Features, Value, Video
from huggingface_hub import hf_hub_download

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

_DATASET_PATH = "suimu/InsAVE-80K"
_DATASET_REVISION = "8ba8ccaad1d8f97b08e218dae1f2b439bd3d2289"
_EVAL_CSV = "eval_data/eval.csv"
_EVAL_TAR = "eval_data/eval/part_00.tar"

# Columns of the upstream eval metadata that this task actually consumes. The
# release also ships `instruction_reverse`, which the forward source -> target task
# does not use, so it is not required here; any additional upstream column is
# ignored rather than rejected.
_SOURCE_COLUMN = "original_video"
_TARGET_COLUMN = "target_video"
_INSTRUCTION_COLUMN = "instruction"
_REQUIRED_COLUMNS = frozenset({_SOURCE_COLUMN, _TARGET_COLUMN, _INSTRUCTION_COLUMN})

_QUERY_ID_PREFIX = "q-"
_HASH_CHUNK_BYTES = 1 << 20
_EXTRACTION_MARKER = ".extraction_complete"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _document_id(video_path: str) -> str:
    """Derive a stable corpus id from the logical path recorded in the eval CSV."""
    return Path(video_path).stem


def _archive_media_members(tar_path: Path) -> dict[str, str]:
    """Map each archive media member to the basename it is extracted under.

    Members are flattened to their basename so that archive layout cannot escape the
    target directory. Flattening is only safe while basenames are unique: a nested
    archive holding both `a/00001.mp4` and `b/00001.mp4` would otherwise collapse two
    distinct clips onto one file, and the surviving content would depend on member
    order. Uniqueness is therefore verified rather than assumed, and this check runs
    on every load rather than only when the archive is first extracted.
    """
    members: dict[str, str] = {}
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name).name
            if not name or name.startswith("."):
                continue
            previous = members.setdefault(name, member.name)
            _require(
                previous == member.name,
                f"archive members {previous!r} and {member.name!r} share the basename "
                f"{name!r}, so flattening them would silently drop one clip",
            )
    _require(bool(members), f"no media members found in {tar_path.name}")
    return members


def _extract_once(tar_path: Path, target_dir: Path, members: dict[str, str]) -> None:
    """Extract the validated media members exactly once, guarded by a marker."""
    marker = target_dir / _EXTRACTION_MARKER
    if marker.exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting InsAVE eval shard to %s", target_dir)
    basename_by_member = {member: name for name, member in members.items()}
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            basename = basename_by_member.get(member.name)
            if basename is None:
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            with (target_dir / basename).open("wb") as out:
                while chunk := extracted.read(_HASH_CHUNK_BYTES):
                    out.write(chunk)
    marker.touch()


def _require(condition: bool, message: str) -> None:
    """Fail loudly when an upstream invariant no longer holds."""
    if not condition:
        raise ValueError(
            f"InsAVE-80K eval release at revision {_DATASET_REVISION} violates an "
            f"expected invariant: {message}. The task must be re-audited before use."
        )


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = _REQUIRED_COLUMNS - set(reader.fieldnames or ())
        _require(
            not missing_columns,
            f"eval.csv is missing required column(s) {sorted(missing_columns)} "
            f"(found {reader.fieldnames})",
        )
        rows = [dict(row) for row in reader]
    _require(len(rows) > 0, "eval.csv is empty")
    for row in rows:
        for column in (_SOURCE_COLUMN, _TARGET_COLUMN, _INSTRUCTION_COLUMN):
            _require(
                bool((row.get(column) or "").strip()),
                f"empty value in required column {column!r}",
            )
        _require(
            row[_SOURCE_COLUMN] != row[_TARGET_COLUMN],
            f"source equals target for {row[_SOURCE_COLUMN]!r}",
        )
    return rows


def _resolve_corpus_files(
    rows: list[dict[str, str]], video_dir: Path
) -> dict[str, Path]:
    """Map every clip referenced by the eval metadata to its extracted file.

    The corpus is exactly the media contents of the official eval shard: both the
    source and the target column, with no distractors added and nothing dropped.
    """
    referenced: dict[str, Path] = {}
    logical_by_id: dict[str, str] = {}
    for row in rows:
        for column in (_SOURCE_COLUMN, _TARGET_COLUMN):
            logical = row[column]
            doc_id = _document_id(logical)
            # Document ids are path stems, so two differently-nested clips could claim
            # the same id and silently collapse into one corpus entry. Compare the
            # logical paths themselves rather than the flattened ones, which would
            # already have merged.
            previous = logical_by_id.setdefault(doc_id, logical)
            _require(
                previous == logical,
                f"document id {doc_id!r} is claimed by two different clips "
                f"({previous!r} and {logical!r}), which would collapse into one entry",
            )
            referenced[doc_id] = video_dir / Path(logical).name

    missing = sorted(doc for doc, file in referenced.items() if not file.is_file())
    _require(
        not missing,
        f"{len(missing)} referenced clips absent from the eval tar "
        f"(e.g. {missing[:5]})",
    )

    on_disk = {file.stem for file in video_dir.glob("*.mp4")}
    _require(
        on_disk == set(referenced),
        f"eval tar contents and eval.csv references disagree "
        f"({len(on_disk - set(referenced))} unreferenced, "
        f"{len(set(referenced) - on_disk)} missing)",
    )
    return referenced


def _build_video_dataset(
    ids: Iterable[str], paths: Iterable[Path], texts: Iterable[str] | None = None
) -> Dataset:
    columns: dict[str, list] = {
        "id": list(ids),
        "video": [str(path) for path in paths],
    }
    features = {"id": Value("string"), "video": Video()}
    if texts is not None:
        columns["text"] = list(texts)
        features["text"] = Value("string")
    return Dataset.from_dict(columns, features=Features(features))


def _build_queries_and_qrels(
    rows: list[dict[str, str]], referenced: dict[str, Path]
) -> tuple[Dataset, dict[str, dict[str, int]]]:
    """Build composed video+text queries and their relevance judgements.

    Some clips are released under more than one name: the split contains exact
    add/remove reverse couples, where one row's source file is byte-for-byte the next
    row's target file. Clips are grouped by SHA-256 so that every id holding the gold
    bytes is marked relevant and a model is not penalised for returning the identical
    video under the other name.

    Grouping is on **exact byte identity only**. Visual, semantic or near-duplicate
    similarity never qualifies, so a query's own source clip cannot become relevant by
    resembling the target -- only by being the very same bytes, which the release never
    does within a pair.
    """
    content_groups: dict[str, list[str]] = {}
    for doc_id, file_path in referenced.items():
        content_groups.setdefault(_sha256(file_path), []).append(doc_id)

    ids: list[str] = []
    paths: list[Path] = []
    texts: list[str] = []
    qrels: dict[str, dict[str, int]] = {}
    for row in rows:
        source_id = _document_id(row[_SOURCE_COLUMN])
        query_id = f"{_QUERY_ID_PREFIX}{source_id}"
        _require(query_id not in qrels, f"duplicate query id {query_id!r}")
        target_file = referenced[_document_id(row[_TARGET_COLUMN])]
        qrels[query_id] = dict.fromkeys(content_groups[_sha256(target_file)], 1)
        ids.append(query_id)
        paths.append(referenced[source_id])
        texts.append(row[_INSTRUCTION_COLUMN])

    _require(all(qrels.values()), "at least one query has no relevant document")
    _require(
        not (set(ids) & set(referenced)),
        "query ids collide with corpus ids, which would silently drop documents "
        "from evaluation",
    )
    return _build_video_dataset(ids, paths, texts), qrels


class InsAVE80KVT2VRetrieval(AbsTaskRetrieval):
    """Composed video+text retrieval derived from the InsAVE-80K evaluation split.

    This is a **new MOEB retrieval construction**, not an official InstructAV2AV
    benchmark. InstructAV2AV uses these 1,000 pairs to score *generation* quality
    (FVD, FAD, PEAVS, Sync-C/D and similar); neither the paper nor the official code
    defines or reports any retrieval evaluation. The retrieval framing, the candidate
    pool and the relevance judgements below are introduced here and should not be
    compared against numbers reported in the paper.

    Construction, derived entirely from the pinned release:

    * Queries are the 1,000 official pairs: the source clip plus the forward
      `instruction`, kept verbatim (including the `<S>`/`<E>` speech markers and the
      13 rows whose instruction text contains upstream pipeline artefacts).
    * The corpus is all 2,000 released evaluation clips, i.e. the complete media
      contents of the eval shard. Nothing is added and nothing is dropped.
    * A query's own source clip therefore stays in the pool as a hard negative and is
      deliberately **not** excluded, following the `CIRRIT2IRetrieval` and
      `FashionIQIT2IRetrieval` precedent. The instruction is what separates the edited
      clip from the unedited one.
    * Relevance is expanded over **exact byte identity only**. The split contains 60
      add/remove reverse couples, where one row's source *file* is byte-for-byte the
      next row's target *file*, so the gold video is present in the corpus under two
      ids. Both ids are marked relevant so a model is not penalised for returning the
      identical bytes under the other name. This affects 120 of the 1,000 queries,
      which carry 2 gold ids; the remaining 880 carry 1. Visual or semantic similarity
      never qualifies -- only an identical SHA-256 digest does -- so a source clip can
      never become relevant merely by resembling the target.

    The closest upstream precedent for content-identity handling is `GreekCivicsQA`
    and `XQuADRetrieval`, which hash document content to derive ids and thereby
    collapse duplicates into a single document. That approach is not used here because
    the corpus is meant to preserve every released file as its own candidate; qrel
    expansion encodes the same fact without dropping ids.
    """

    metadata = TaskMetadata(
        name="InsAVE80KVT2VRetrieval",
        description=(
            "Instruction-conditioned composed audio-video retrieval built from the "
            "held-out evaluation split of InsAVE-80K, the audio-video editing dataset "
            "released with InstructAV2AV. Each query pairs a source clip with the "
            "natural-language editing instruction that was applied to it, and the "
            "relevant document is the corresponding edited clip. The candidate pool is "
            "the complete set of evaluation clips, so every query's own source clip "
            "stays in the pool as a hard negative and the instruction is required to "
            "prefer the edited clip over the unedited one. Clips carry synchronised "
            "audio, and instructions may contain <S>/<E> markers delimiting intended "
            "spoken content. This is a new MOEB retrieval construction derived from the "
            "official 1,000-pair generation-evaluation split: InstructAV2AV scores "
            "generation quality on these pairs and defines no retrieval evaluation, so "
            "scores here are not comparable to any number reported in the paper."
        ),
        reference="https://arxiv.org/abs/2605.18467",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="vt2v",
        modalities=["video", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        # Upstream publishes no source-media collection dates. Both endpoints are
        # exact public timestamps: arXiv:2605.18467v1 was submitted 2026-05-18
        # 14:27:05 UTC (first public description of the dataset) and the pinned HF
        # revision was committed 2026-07-28 07:56:17 UTC.
        date=("2026-05-18", "2026-07-28"),
        domains=["Scene", "Spoken", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        # The HF repository declares MIT, but the clips are derived from YouTube and
        # from other datasets (MovieBench, Condensed Movies, Short-Films-20K,
        # VGGSound), and the dataset card itself disclaims the rights on the
        # underlying media. The licence of the video data is therefore not
        # established, matching YouCook2 / ActivityNetCaptions.
        license="not specified",
        annotations_creators="automatic-and-reviewed",
        dialect=[],
        sample_creation="multiple",
        bibtex_citation=r"""
@article{zheng2026instructav2av,
  author = {Zheng, Haojie and Yang, Yixin and Yang, Siqi and Weng, Shuchen and Shi, Boxin},
  journal = {arXiv preprint arXiv:2605.18467},
  title = {InstructAV2AV: Instruction-Guided Audio-Video Joint Editing},
  year = {2026},
}
""",
        prompt={
            "query": "Given a source video and an editing instruction, retrieve the video that results from applying the instruction to the source video."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        path = self.metadata.dataset["path"]
        revision = self.metadata.dataset["revision"]
        csv_path = Path(
            hf_hub_download(path, _EVAL_CSV, repo_type="dataset", revision=revision)
        )
        tar_path = Path(
            hf_hub_download(path, _EVAL_TAR, repo_type="dataset", revision=revision)
        )
        video_dir = tar_path.parent / f"{tar_path.stem}_extracted"
        _extract_once(tar_path, video_dir, _archive_media_members(tar_path))

        rows = _read_rows(csv_path)
        referenced = _resolve_corpus_files(rows, video_dir)
        queries, qrels = _build_queries_and_qrels(rows, referenced)

        corpus_ids = sorted(referenced)
        corpus = _build_video_dataset(
            corpus_ids, [referenced[doc_id] for doc_id in corpus_ids]
        )

        self.dataset = {
            "default": {
                "test": RetrievalSplitData(
                    corpus=corpus,
                    queries=queries,
                    relevant_docs=qrels,
                    top_ranked=None,
                )
            }
        }
        self.data_loaded = True
