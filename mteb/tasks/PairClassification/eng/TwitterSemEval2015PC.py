from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class TwitterSemEval2015PC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="TwitterSemEval2015",
        dataset={
            "path": "mteb/twittersemeval2015-pairclassification",
            "revision": "70970daeab8776df92f5ea462b6173c0b46fd2d1",
        },
        description="Paraphrase-Pairs of Tweets from the SemEval 2015 workshop.",
        reference="https://alt.qcri.org/semeval2015/task1/",
        category="s2s",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=None,
        domains=["Social", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{xu-etal-2015-semeval,
  address = {Denver, Colorado},
  author = {Xu, Wei  and
Callison-Burch, Chris  and
Dolan, Bill},
  booktitle = {Proceedings of the 9th International Workshop on Semantic Evaluation ({S}em{E}val 2015)},
  doi = {10.18653/v1/S15-2001},
  editor = {Nakov, Preslav  and
Zesch, Torsten  and
Cer, Daniel  and
Jurgens, David},
  month = jun,
  pages = {1--11},
  publisher = {Association for Computational Linguistics},
  title = {{S}em{E}val-2015 Task 1: Paraphrase and Semantic Similarity in {T}witter ({PIT})},
  url = {https://aclanthology.org/S15-2001},
  year = {2015},
}
""",
        prompt="Retrieve tweets that are semantically similar to the given tweet",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")
