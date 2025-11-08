from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class ClincIntentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ClincIntentClassification",
        description="Task-oriented dialog systems need to know when a query falls outside their range of supported intents, but current text classification corpora only define label sets that cover every example",
        dataset={
            "path": "DeepPavlov/clinc_oos",
            "revision": "9b995dc4a780cfabf0c7bb044bab7f48a3762c8d",
        },
        reference="https://huggingface.co/datasets/clinc/clinc_oos",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs={
            "small": ["eng-Latn"],
            "plus": ["eng-Latn"],
            "imbalanced": ["eng-Latn"],
        },
        main_score="accuracy",
        date=("2019-01-01", "2019-01-01"),
        domains=["Financial", "Web", "Social"],
        task_subtypes=["Intent classification"],
        license="cc-by-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""@inproceedings{larson-etal-2019-evaluation,
        title = "An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction",
        author = "Larson, Stefan  and
          Mahendran, Anish  and
          Peper, Joseph J.  and
          Clarke, Christopher  and
          Lee, Andrew  and
          Hill, Parker  and
          Kummerfeld, Jonathan K.  and
          Leach, Kevin  and
          Laurenzano, Michael A.  and
          Tang, Lingjia  and
          Mars, Jason",
        booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
        year = "2019",
        url = "https://www.aclweb.org/anthology/D19-1131"
    }""",
    )
