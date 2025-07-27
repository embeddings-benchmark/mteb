from __future__ import annotations

from mteb.abstasks.AbsTaskSummarization import AbsTaskSummarization
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.evaluation.evaluators.SummarizationEvaluator import (
    DeprecatedSummarizationEvaluator,
)


class SummEvalSummarization(AbsTaskSummarization):
    superseded_by = "SummEvalSummarization.v2"
    evalutor = DeprecatedSummarizationEvaluator

    metadata = TaskMetadata(
        name="SummEval",
        description="News Article Summary Semantic Similarity Estimation.",
        reference="https://github.com/Yale-LILY/SummEval",
        dataset={
            "path": "mteb/summeval",
            "revision": "cda12ad7615edc362dbf25a00fdd61d3b1eaf93c",
        },
        type="Summarization",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2016-01-01", "2016-12-31"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@article{fabbri2020summeval,
  author = {Fabbri, Alexander R and Kry{\'s}ci{\'n}ski, Wojciech and McCann, Bryan and Xiong, Caiming and Socher, Richard and Radev, Dragomir},
  journal = {arXiv preprint arXiv:2007.12626},
  title = {SummEval: Re-evaluating Summarization Evaluation},
  year = {2020},
}
""",
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict


class SummEvalSummarizationv2(AbsTaskSummarization):
    metadata = TaskMetadata(
        name="SummEvalSummarization.v2",
        description="News Article Summary Semantic Similarity Estimation. This version fixes a bug in the evaluation script that caused the main score to be computed incorrectly.",
        reference="https://github.com/Yale-LILY/SummEval",
        dataset={
            "path": "mteb/summeval",
            "revision": "cda12ad7615edc362dbf25a00fdd61d3b1eaf93c",
        },
        type="Summarization",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2016-01-01", "2016-12-31"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@article{fabbri2020summeval,
  author = {Fabbri, Alexander R and Kry{\'s}ci{\'n}ski, Wojciech and McCann, Bryan and Xiong, Caiming and Socher, Richard and Radev, Dragomir},
  journal = {arXiv preprint arXiv:2007.12626},
  title = {SummEval: Re-evaluating Summarization Evaluation},
  year = {2020},
}
""",
        adapted_from=["SummEvalSummarization"],
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
