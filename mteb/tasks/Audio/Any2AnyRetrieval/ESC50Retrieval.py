from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class ESC50A2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="ESC50A2TRetrieval",
        description=(
            "Retrieve the correct environmental sound label/description from an audio clip "
            "in the ESC-50 dataset of 50 environmental classes."
        ),
        reference="https://github.com/karolpiczak/ESC-50",
        dataset={
            "path": "mteb/esc50_a2t",
            "revision": "c4db6101f292632e50cb136ba4132e1515063d1b",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_1",
        date=("2015-01-01", "2015-12-31"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[""],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{piczak2015dataset,
  author = {Piczak, Karol J.},
  booktitle = {Proceedings of the 23rd {Annual ACM Conference} on {Multimedia}},
  date = {2015-10-13},
  doi = {10.1145/2733373.2806390},
  isbn = {978-1-4503-3459-4},
  location = {{Brisbane, Australia}},
  pages = {1015--1018},
  publisher = {{ACM Press}},
  title = {{ESC}: {Dataset} for {Environmental Sound Classification}},
  url = {http://dl.acm.org/citation.cfm?doid=2733373.2806390},
}
""",
    )


class ESC50T2ARetrieval(AbsTaskAny2AnyRetrieval):
    """Text-to-audio retrieval on ESC-50 class labels ↔ audio pairs."""

    metadata = TaskMetadata(
        name="ESC50T2ARetrieval",
        description=(
            "Retrieve the correct environmental sound clip for a given ESC-50 class label/description."
        ),
        reference="https://github.com/karolpiczak/ESC-50",
        dataset={
            "path": "mteb/esc50_t2a",
            "revision": "a3b1f7c174361184463d9f584b83035e43571563",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_1",
        date=("2015-01-01", "2015-12-31"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[""],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{piczak2015dataset,
  author = {Piczak, Karol J.},
  booktitle = {Proceedings of the 23rd {Annual ACM Conference} on {Multimedia}},
  date = {2015-10-13},
  doi = {10.1145/2733373.2806390},
  isbn = {978-1-4503-3459-4},
  location = {{Brisbane, Australia}},
  pages = {1015--1018},
  publisher = {{ACM Press}},
  title = {{ESC}: {Dataset} for {Environmental Sound Classification}},
  url = {http://dl.acm.org/citation.cfm?doid=2733373.2806390},
}
""",
    )
