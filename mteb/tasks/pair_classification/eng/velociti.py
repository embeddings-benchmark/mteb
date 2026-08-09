from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from collections.abc import Mapping

_CACHE_DIR = Path(tempfile.gettempdir()) / "mteb_velociti_pairclassification_cache"


class VELOCITIPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="VELOCITIPairClassification",
        description=(
            "Tests whether a model correctly binds actions to the agents performing "
            "them across a video, by pairing each video with a correct caption and a "
            "subtly wrong one and measuring how well embedding similarity separates "
            "the two. Negatives are constructed via in-video negation (swapping "
            "agents/actions that both occur in the same video) or text-inspired "
            "negation (LLM-generated contradictions), forming a strict entailment "
            "test rather than a coarse mismatch. From the VELOCITI benchmark."
        ),
        reference="https://arxiv.org/abs/2406.10889",
        dataset={
            "path": "yaswanth169/VELOCITI-PC",
            "revision": "cab7ae8f379937ee687375fb23aac3ffe1039802",
        },
        type="VideoPairClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2024-06-16", "2024-06-16"),
        domains=["Activity", "Web"],
        task_subtypes=["Caption Pairing"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="LM-generated and verified",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{saravanan2025velociti,
  author = {Saravanan, Darshana and Gupta, Varun and Singh, Darshan and Khan, Zeeshan and Gandhi, Vineet and Tapaswi, Makarand},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  title = {VELOCITI: Benchmarking Video-Language Compositional Reasoning with Strict Entailment},
  year = {2025},
}
""",
    )

    input1_column_name: ClassVar[Mapping[str, str]] = {"video": "video"}
    input2_column_name: ClassVar[Mapping[str, str]] = {"text": "text"}
    label_column_name: str = "label"

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        # Imported lazily: datasets.Video requires the optional torchcodec
        # backend, and importing it at module level would break importing
        # this task (and thus the whole mteb.tasks package) without it.
        from datasets import Features, Value, Video

        raw = load_dataset(
            self.metadata.dataset["path"],
            revision=self.metadata.dataset["revision"],
            split="test",
        )

        zip_path = hf_hub_download(
            repo_id=self.metadata.dataset["path"],
            filename="videos.zip",
            repo_type="dataset",
            revision=self.metadata.dataset["revision"],
        )
        if not _CACHE_DIR.exists():
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(_CACHE_DIR)

        features = Features(
            {"video": Video(), "text": Value("string"), "label": Value("int64")}
        )
        records = [
            {
                "video": {"path": str(_CACHE_DIR / row["video_id"]), "bytes": None},
                "text": row["text"],
                "label": row["label"],
            }
            for row in raw
        ]
        self.dataset = {"test": Dataset.from_list(records, features=features)}
        self.data_loaded = True
