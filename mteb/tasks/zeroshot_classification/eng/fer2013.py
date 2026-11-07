from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FER2013ZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FER2013ZeroShot",
        description="Classifying facial emotions.",
        reference="https://arxiv.org/abs/1412.6572",
        dataset={
            "path": "mteb/FER2013ZeroShot",
            "revision": "a6c51377a3ea46b68a3e1e3635525052d73fb4c7",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2014-01-01",
            "2014-12-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Emotion recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{goodfellow2015explainingharnessingadversarialexamples,
  archiveprefix = {arXiv},
  author = {Ian J. Goodfellow and Jonathon Shlens and Christian Szegedy},
  booktitle = {3rd International Conference on Learning Representations, {ICLR} 2015},
  eprint = {1412.6572},
  primaryclass = {stat.ML},
  title = {Explaining and Harnessing Adversarial Examples},
  year = {2015},
}
""",
    )
