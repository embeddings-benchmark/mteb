from __future__ import annotations

from mteb.abstasks import AbsTaskSummarization
from mteb.abstasks.TaskMetadata import TaskMetadata


class SummEvalFrSummarization(AbsTaskSummarization):
    metadata = TaskMetadata(
        name="SummEvalFr",
        description="News Article Summary Semantic Similarity Estimation translated from english to french with DeepL.",
        reference="https://github.com/Yale-LILY/SummEval",
        hf_hub_name="lyon-nlp/summeval",
        type="Summarization",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["fr"],
        main_score="cosine_spearman",
        revision="b385812de6a9577b6f4d0f88c6a6e35395a94054",
        date=None,
        form=["written"],
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation="machine-translated",
        bibtex_citation=None,
        n_samples={},
        avg_character_length={},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5

        return metadata_dict
