from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class LugandaSTS(AbsTaskSTS):
    """Semantic textual similarity evaluation for Luganda sentence pairs."""

    metadata = TaskMetadata(
        name="LugandaSTS",
        description=(
            "Luganda Semantic Textual Similarity benchmark adapted from the "
            "STS Benchmark. It evaluates sentence embedding models by measuring "
            "the correlation between cosine similarity scores and the adapted "
            "semantic similarity scores for pairs of Luganda sentences. The "
            "current evaluation split contains 300 sentence pairs."
        ),
        dataset={
            "path": "sagepond/sts",
            "revision": "0d52f8af50b78a35d7134e6da422d3d04ca70606",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        reference="https://huggingface.co/datasets/sagepond/sts",
        eval_splits=["test"],
        eval_langs=["lug-Latn"],
        main_score="spearman",
        domains=["Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        date=("2026-08-22", "2026-11-14"),
        sample_creation="human-translated and localized",
        adapted_from=["STSBenchmark"],
        is_public=True,
        bibtex_citation=r"""
@inproceedings{cer-etal-2017-semeval,
  address = {Vancouver, Canada},
  author = {Cer, Daniel  and
Diab, Mona  and
Agirre, Eneko  and
Lopez-Gazpio, I{\~n}igo  and
Specia, Lucia},
  booktitle = {Proceedings of the 11th International Workshop on Semantic Evaluation ({S}em{E}val-2017)},
  doi = {10.18653/v1/S17-2001},
  editor = {Bethard, Steven  and
Carpuat, Marine  and
Apidianaki, Marianna  and
Mohammad, Saif M.  and
Cer, Daniel  and
Jurgens, David},
  month = aug,
  pages = {1--14},
  publisher = {Association for Computational Linguistics},
  title = {{S}em{E}val-2017 Task 1: Semantic Textual Similarity Multilingual and Crosslingual Focused Evaluation},
  url = {https://aclanthology.org/S17-2001},
  year = {2017},
}
""",
    )

    column_names = ("sentence1", "sentence2")
    min_score = 0
    max_score = 5
