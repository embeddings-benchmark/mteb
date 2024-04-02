from __future__ import annotations

from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.TaskMetadata import TaskMetadata


class SickrPLSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="SICK-R-PL",
        hf_hub_name="PL-MTEB/sickr-pl-sts",
        description="Polish version of SICK dataset for textual relatedness.",
        reference="https://aclanthology.org/2020.lrec-1.207",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pl"],
        main_score="cosine_spearman",
        revision="fd5c2441b7eeff8676768036142af4cfa42c1339",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"test": 9812},
        avg_character_length={"test": 42.8},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["min_score"] = 1
        metadata_dict["max_score"] = 5
        return metadata_dict


class CdscrSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="CDSC-R",
        hf_hub_name="PL-MTEB/cdscr-sts",
        description="Compositional Distributional Semantics Corpus for textual relatedness.",
        reference="https://aclanthology.org/P17-1073.pdf",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pl"],
        main_score="cosine_spearman",
        revision="1cd6abbb00df7d14be3dbd76a7dcc64b3a79a7cd",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["min_score"] = 1
        metadata_dict["max_score"] = 5
        return metadata_dict
