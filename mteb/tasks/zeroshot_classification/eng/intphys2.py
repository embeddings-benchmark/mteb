from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import AbsTaskZeroShotClassification


class IntPhys2VideoZeroShot(AbsTaskZeroShotClassification):
    _DATASET_PATH = "mteb/IntPhys2VideoZeroShot"
    _DATASET_REVISION = "503c0a2f8e624472ff3e44cdc6065ec68c9df2ed"
    _LABEL_NAMES = (
        "object behavior is inconsistent with Earth's physical laws",
        "object behavior is consistent with Earth's physical laws",
    )

    metadata = TaskMetadata(
        name="IntPhys2VideoZeroShot",
        description=(
            "Classify synthetic videos as physically possible or impossible across "
            "permanence, immutability, continuity, and solidity conditions."
        ),
        reference="https://arxiv.org/abs/2506.09849",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="VideoZeroshotClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        prompt="Represent the input for physical plausibility classification.",
        date=("2025-06-11", "2025-06-11"),
        domains=["Constructed"],
        task_subtypes=["Physical plausibility classification"],
        license="cc-by-nc-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@misc{bordes2025intphys2benchmarkingintuitive,
  archiveprefix = {arXiv},
  author = {Florian Bordes and Quentin Garrido and Justine T Kao and Adina Williams and Michael Rabbat and Emmanuel Dupoux},
  eprint = {2506.09849},
  primaryclass = {cs.CV},
  title = {IntPhys 2: Benchmarking Intuitive Physics Understanding In Complex Synthetic Environments},
  url = {https://arxiv.org/abs/2506.09849},
  year = {2025},
}
""",
        is_beta=True,
    )

    input_column_name = "video"
    label_column_name = "label"

    def get_candidate_labels(self) -> list[str]:
        return [f"a video where {label}" for label in self._LABEL_NAMES]
