from __future__ import annotations

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.types import PromptType


class VideoConPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="VideoConPairClassification",
        description=(
            "Pair classification on the VideoCon dataset: "
            "determining whether a text caption correctly describes "
            "a video or is a semantically-plausible contrast caption "
            "(e.g. entity/action/attribute swaps, event order flips). "
            "Tests video-language alignment robustness."
        ),
        reference="https://arxiv.org/abs/2311.10111",
        dataset={
            "path": "zachz/VideoCon-PC",
            "revision": "93a3f53b21061fc66538854e4005f4cceddc0cd8",
        },
        type="VideoPairClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2023-11-01", "2023-11-30"),
        domains=["Scene"],
        task_subtypes=["Caption Pairing"],
        license="mit",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="LM-generated and verified",
        is_beta=True,
        bibtex_citation=r"""
@article{bansal2023videocon,
  author = {Bansal, Hritik and Bitton, Yonatan and Szpektor, Idan and Chang, Kai-Wei and Grover, Aditya},
  journal = {arXiv preprint arXiv:2311.10111},
  title = {VideoCon: Robust Video-Language Alignment via Contrast Captions},
  year = {2023},
}
""",
    )

    input1_column_name: str = "video"
    input2_column_name: str = "text"
    label_column_name: str = "label"
    input1_prompt_type: PromptType | None = PromptType.query
    input2_prompt_type: PromptType | None = PromptType.document

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        for split in self.metadata.eval_splits:
            ds = self.dataset[split]
            ds = ds.rename_column("caption", "text")
            ds = ds.add_column("id", list(range(len(ds))))
            self.dataset[split] = ds
