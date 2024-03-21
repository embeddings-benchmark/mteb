from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class LccSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LccSentimentClassification",
        hf_hub_name="DDSC/lcc",
        description="The leipzig corpora collection, annotated for sentiment",
        reference="https://github.com/fnielsen/lcc-sentiment",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["da"],
        main_score="accuracy",
        revision="de7ba3406ee55ea2cc52a0a41408fa6aede6d3c6",
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
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 16
        return metadata_dict
