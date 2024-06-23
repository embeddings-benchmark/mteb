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
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ap",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@inproceedings{xu-etal-2015-semeval,
        title = "{S}em{E}val-2015 Task 1: Paraphrase and Semantic Similarity in {T}witter ({PIT})",
        author = "Xu, Wei  and
        Callison-Burch, Chris  and
        Dolan, Bill",
        editor = "Nakov, Preslav  and
        Zesch, Torsten  and
        Cer, Daniel  and
        Jurgens, David",
        booktitle = "Proceedings of the 9th International Workshop on Semantic Evaluation ({S}em{E}val 2015)",
        month = jun,
        year = "2015",
        address = "Denver, Colorado",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/S15-2001",
        doi = "10.18653/v1/S15-2001",
        pages = "1--11",
    }""",
        n_samples={"test": 16777},
        avg_character_length={"test": 38.3},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")
