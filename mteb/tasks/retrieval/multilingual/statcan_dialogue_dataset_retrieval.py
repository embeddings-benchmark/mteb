from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLITS = ["dev", "test"]

_LANGS = {
    # <iso_639_3>-<ISO_15924>
    "english": ["eng-Latn"],
    "french": ["fra-Latn"],
}


class StatcanDialogueDatasetRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="StatcanDialogueDatasetRetrieval",
        description="A Dataset for Retrieving Data Tables through Conversations with Genuine Intents, available in English and French.",
        dataset={
            "path": "mteb/StatcanDialogueDatasetRetrieval",
            "revision": "6f9747b22a93c6cf714945bca43576089fb80de3",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=_EVAL_SPLITS,
        eval_langs=_LANGS,
        main_score="recall_at_10",
        reference="https://mcgill-nlp.github.io/statcan-dialogue-dataset/",
        date=("2020-01-01", "2020-04-15"),
        domains=["Government", "Web", "Written"],
        task_subtypes=["Conversational retrieval"],
        license="https://huggingface.co/datasets/McGill-NLP/statcan-dialogue-dataset-retrieval/blob/main/LICENSE.md",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{lu-etal-2023-statcan,
  address = {Dubrovnik, Croatia},
  author = {Lu, Xing Han  and
Reddy, Siva  and
de Vries, Harm},
  booktitle = {Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics},
  month = may,
  pages = {2799--2829},
  publisher = {Association for Computational Linguistics},
  title = {The {S}tat{C}an Dialogue Dataset: Retrieving Data Tables through Conversations with Genuine Intents},
  url = {https://arxiv.org/abs/2304.01412},
  year = {2023},
}
""",
    )
