from __future__ import annotations

from mteb.abstasks import AbsTaskMultilabelClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class BrazilianToxicTweetsClassification(AbsTaskMultilabelClassification):
    metadata = TaskMetadata(
        name="BrazilianToxicTweetsClassification",
        description="""
        ToLD-Br is the biggest dataset for toxic tweets in Brazilian Portuguese, crowdsourced by 42 annotators selected from
        a pool of 129 volunteers. Annotators were selected aiming to create a plural group in terms of demographics (ethnicity,
        sexual orientation, age, gender). Each tweet was labeled by three annotators in 6 possible categories: LGBTQ+phobia,
        Xenophobia, Obscene, Insult, Misogyny and Racism.
        """,
        reference="https://paperswithcode.com/dataset/told-br",
        dataset={
            "path": "told-br",
            "revision": "fb4f11a5bc68b99891852d20f1ec074be6289768",
            "name": "multilabel",
        },
        type="MultilabelClassification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="accuracy",
        date=("2019-08-01", "2019-08-16"),
        form=["written"],
        domains=["Constructed"],
        task_subtypes=["Sentiment/Hate speech"],
        license="CC BY-SA 4.0",
        socioeconomic_status="medium",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        text_creation="found",
        bibtex_citation="""@article{DBLP:journals/corr/abs-2010-04543,
            author    = {Joao Augusto Leite and
                        Diego F. Silva and
                        Kalina Bontcheva and
                        Carolina Scarton},
            title     = {Toxic Language Detection in Social Media for Brazilian Portuguese:
                        New Dataset and Multilingual Analysis},
            journal   = {CoRR},
            volume    = {abs/2010.04543},
            year      = {2020},
            url       = {https://arxiv.org/abs/2010.04543},
            eprinttype = {arXiv},
            eprint    = {2010.04543},
            timestamp = {Tue, 15 Dec 2020 16:10:16 +0100},
            }""",
        n_samples={"test": 2048},
        avg_character_length={"test": 85.05},
    )

    def dataset_transform(self):
        cols_ = ["homophobia", "obscene", "insult", "racism", "misogyny", "xenophobia"]
        n_size = len(self.dataset["train"])
        labels = [[] for _ in range(n_size)]
        for c in cols_:
            col_list = self.dataset["train"][c]
            for i in range(n_size):
                if col_list[i] > 0:
                    labels[i].append(c)
        self.dataset = self.dataset["train"].add_column("label", labels)
        del labels
        self.dataset = self.dataset.remove_columns(cols_)
        self.dataset = self.dataset.train_test_split(
            train_size=2048, test_size=2048, seed=self.seed
        )
