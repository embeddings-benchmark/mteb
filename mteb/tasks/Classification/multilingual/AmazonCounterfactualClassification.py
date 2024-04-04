from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification, MultilingualTask

_LANGUAGES = ["en", "de", "en-ext", "ja"]


class AmazonCounterfactualClassification(MultilingualTask, AbsTaskClassification):
    metadata = TaskMetadata(
        name="AmazonCounterfactualClassification",
        dataset={
            "path": "mteb/amazon_counterfactual",
            "revision": "e8379541af4e31359cca9fbcf4b00f2671dba205",
        },
        description=(
            "A collection of Amazon customer reviews annotated for counterfactual detection pair classification."
        ),
        reference="https://arxiv.org/abs/2104.06893",
        category="s2s",
        type="Classification",
        eval_splits=["validation", "test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
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
        n_samples={"validation": 335, "test": 670},
        avg_character_length={"validation": 109.2, "test": 106.1},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 32
        return metadata_dict
