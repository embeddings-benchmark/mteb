from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class SickrSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="SICK-R",
        hf_hub_name="MMathematica/sickr-sts",
        description="Semantic Textual Similarity SICK-R dataset as described here:",
        reference="https://aclanthology.org/2020.lrec-1.207",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="cosine_spearman",
        revision="a6ea5a8cab320b040a23452cc28066d9beae2cee",
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
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
