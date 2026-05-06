from __future__ import annotations

from typing import Any

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class Diving48ClassificationV1(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Diving48Classification.V1",
        description="Diving48 is a fine-grained video dataset of competitive diving, consisting of ~18k trimmed video clips of 48 unambiguous dive sequences (standardized by FINA). This proves to be a challenging task for modern action recognition systems as dives may differ in three stages (takeoff, flight, entry) and thus require modeling of long-term temporal dynamics. ",
        reference="https://openaccess.thecvf.com/content_ECCV_2018/papers/Yingwei_Li_RESOUND_Towards_Action_ECCV_2018_paper.pdf",
        dataset={
            "path": "mteb/diving48",
            "revision": "99b1f700c7675268169dcedc439e779e55fe7471",
        },
        type="VideoClassification",
        category="v2c",
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="accuracy",
        date=(
            "2014-06-23",
            "2018-01-01",
        ),
        domains=["Sport"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{Li_2018_ECCV,
  author = {Li, Yingwei and Li, Yi and Vasconcelos, Nuno},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  month = {September},
  title = {RESOUND: Towards Action Recognition without Representation Bias},
  year = {2018},
}
""",
        superseded_by="Diving48Classification.V2",
    )

    input_column_name = "video"
    label_column_name = "label"

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        self.dataset = self.stratified_subsampling(
            self.dataset,
            seed=self.seed,
            n_samples=2048,
        )


class Diving48ClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Diving48Classification.V2",
        description="Diving48 is a fine-grained video dataset of competitive diving, consisting of ~18k trimmed video clips of 48 unambiguous dive sequences (standardized by FINA). This proves to be a challenging task for modern action recognition systems as dives may differ in three stages (takeoff, flight, entry) and thus require modeling of long-term temporal dynamics. ",
        reference="https://openaccess.thecvf.com/content_ECCV_2018/papers/Yingwei_Li_RESOUND_Towards_Action_ECCV_2018_paper.pdf",
        dataset={
            "path": "mteb/diving48v2",
            "revision": "21edd1210183fc59d2576fa5d0145b8ead87e767",
        },
        type="VideoClassification",
        category="v2c",
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="accuracy",
        date=(
            "2014-06-23",
            "2018-01-01",
        ),
        domains=["Sport"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{Li_2018_ECCV,
  author = {Li, Yingwei and Li, Yi and Vasconcelos, Nuno},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  month = {September},
  title = {RESOUND: Towards Action Recognition without Representation Bias},
  year = {2018},
}
""",
        adapted_from=["Diving48Classification.V1"],
    )

    input_column_name = "video"
    label_column_name = "label"

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        self.dataset = self.stratified_subsampling(
            self.dataset,
            seed=self.seed,
            n_samples=2048,
        )
