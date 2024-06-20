from __future__ import annotations

from mteb.abstasks import AbsTaskSummarization
from mteb.abstasks.TaskMetadata import TaskMetadata


class SummEvalFrSummarization(AbsTaskSummarization):
    metadata = TaskMetadata(
        name="SummEvalFr",
        description="News Article Summary Semantic Similarity Estimation translated from english to french with DeepL.",
        reference="https://github.com/Yale-LILY/SummEval",
        dataset={
            "path": "lyon-nlp/summarization-summeval-fr-p2p",
            "revision": "b385812de6a9577b6f4d0f88c6a6e35395a94054",
        },
        type="Summarization",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="cosine_spearman",
        date=("2016-01-01", "2016-12-31"),
        form=["written"],
        domains=["News"],
        task_subtypes=[],
        license="mit",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="machine-translated",
        bibtex_citation="""@article{fabbri2020summeval,
  title={SummEval: Re-evaluating Summarization Evaluation},
  author={Fabbri, Alexander R and Kry{\'s}ci{\'n}ski, Wojciech and McCann, Bryan and Xiong, Caiming and Socher, Richard and Radev, Dragomir},
  journal={arXiv preprint arXiv:2007.12626},
  year={2020}
}""",
        n_samples={"test": 2800},
        avg_character_length={"test": 407.1},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5

        return metadata_dict
