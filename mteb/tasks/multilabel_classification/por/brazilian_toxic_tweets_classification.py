from mteb.abstasks.multilabel_classification import (
    AbsTaskMultilabelClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class BrazilianToxicTweetsClassification(AbsTaskMultilabelClassification):
    metadata = TaskMetadata(
        name="BrazilianToxicTweetsClassification",
        description="ToLD-Br is the biggest dataset for toxic tweets in Brazilian Portuguese, crowdsourced by 42 annotators selected from a pool of 129 volunteers. Annotators were selected aiming to create a plural group in terms of demographics (ethnicity, sexual orientation, age, gender). Each tweet was labeled by three annotators in 6 possible categories: LGBTQ+phobia, Xenophobia, Obscene, Insult, Misogyny and Racism.",
        reference="https://paperswithcode.com/dataset/told-br",
        dataset={
            "path": "mteb/BrazilianToxicTweetsClassification",
            "revision": "d7af1f3c9ff00fbb4e1e40021788a32d78a936bd",
        },
        type="MultilabelClassification",
        category="t2c",
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
