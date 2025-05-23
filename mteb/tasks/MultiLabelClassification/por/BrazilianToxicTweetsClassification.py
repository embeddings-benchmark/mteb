from __future__ import annotations

from mteb.abstasks.AbsTaskMultilabelClassification import (
    AbsTaskMultilabelClassification,
)
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
            "path": "mteb/told-br",
            "revision": "f333c1fcfa3ab43f008a327c8bd0140441354d34",
        },
        type="MultilabelClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="accuracy",
        date=("2019-08-01", "2019-08-16"),
        domains=["Constructed", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="found",
        bibtex_citation=r"""
@article{DBLP:journals/corr/abs-2010-04543,
  author = {Joao Augusto Leite and
Diego F. Silva and
Kalina Bontcheva and
Carolina Scarton},
  eprint = {2010.04543},
  eprinttype = {arXiv},
  journal = {CoRR},
  timestamp = {Tue, 15 Dec 2020 16:10:16 +0100},
  title = {Toxic Language Detection in Social Media for Brazilian Portuguese:
New Dataset and Multilingual Analysis},
  url = {https://arxiv.org/abs/2010.04543},
  volume = {abs/2010.04543},
  year = {2020},
}
""",
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
            train_size=len(self.dataset) // 2,
            test_size=len(self.dataset) // 2,
            seed=self.seed,
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"], n_samples=8192
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"], n_samples=2048
        )
