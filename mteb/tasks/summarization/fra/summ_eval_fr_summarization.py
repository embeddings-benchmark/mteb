from mteb._evaluators.text.summarization_evaluator import (
    DeprecatedSummarizationEvaluator,
)
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.summarization import AbsTaskSummarization


class SummEvalFrSummarization(AbsTaskSummarization):
    evaluator = DeprecatedSummarizationEvaluator
    metadata = TaskMetadata(
        name="SummEvalFr",
        description="News Article Summary Semantic Similarity Estimation translated from english to french with DeepL.",
        reference="https://github.com/Yale-LILY/SummEval",
        dataset={
            "path": "lyon-nlp/summarization-summeval-fr-p2p",
            "revision": "b385812de6a9577b6f4d0f88c6a6e35395a94054",
        },
        type="Summarization",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="cosine_spearman",
        date=("2016-01-01", "2016-12-31"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@article{fabbri2020summeval,
  author = {Fabbri, Alexander R and Kry{\'s}ci{\'n}ski, Wojciech and McCann, Bryan and Xiong, Caiming and Socher, Richard and Radev, Dragomir},
  journal = {arXiv preprint arXiv:2007.12626},
  title = {SummEval: Re-evaluating Summarization Evaluation},
  year = {2020},
}
""",
        superseded_by="SummEvalFrSummarization.v2",
    )

    min_score = 0
    max_score = 5


class SummEvalFrSummarizationv2(AbsTaskSummarization):
    metadata = TaskMetadata(
        name="SummEvalFrSummarization.v2",
        description="News Article Summary Semantic Similarity Estimation translated from english to french with DeepL. This version fixes a bug in the evaluation script that caused the main score to be computed incorrectly.",
        reference="https://github.com/Yale-LILY/SummEval",
        dataset={
            "path": "lyon-nlp/summarization-summeval-fr-p2p",
            "revision": "b385812de6a9577b6f4d0f88c6a6e35395a94054",
        },
        type="Summarization",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="cosine_spearman",
        date=("2016-01-01", "2016-12-31"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@article{fabbri2020summeval,
  author = {Fabbri, Alexander R and Kry{\'s}ci{\'n}ski, Wojciech and McCann, Bryan and Xiong, Caiming and Socher, Richard and Radev, Dragomir},
  journal = {arXiv preprint arXiv:2007.12626},
  title = {SummEval: Re-evaluating Summarization Evaluation},
  year = {2020},
}
""",
        adapted_from=["SummEvalFrSummarization"],
    )

    min_score = 0
    max_score = 5
