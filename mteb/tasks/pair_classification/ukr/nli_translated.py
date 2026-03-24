from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class UkrNLITranslatedStandford(AbsTaskPairClassification):
    input1_column_name = "premise"
    input2_column_name = "hypothesis"
    label_column_name = "labels"
    metadata = TaskMetadata(
        name="UkrNLITranslatedStandford",
        description="NLI classification task into 0 - entailment, 1 - neutral, 2 - contradiction for sentences by translating NLI English data.",
        reference="https://huggingface.co/datasets/ukr-detect/ukr-nli-dataset-translated-stanford",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ukr-Cyrl"],
        main_score="max_ap",
        dataset={
            "path": "mteb/UkrNLITranslatedStandford",
            "revision": "6e94ddfdcc9a0c73af1679bd3cff73fac97564d1",
        },
        date=("2015-01-01", "2015-12-31"),
        domains=["Constructed"],
        task_subtypes=[],
        license="openrail++",
        annotations_creators="derived",
        dialect=["Textual Entailment"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{dementieva-etal-2025-cross,
  address = {Abu Dhabi, UAE},
  author = {Dementieva, Daryna  and
Khylenko, Valeriia  and
Groh, Georg},
  booktitle = {Proceedings of the 31st International Conference on Computational Linguistics},
  editor = {Rambow, Owen  and
Wanner, Leo  and
Apidianaki, Marianna  and
Al-Khalifa, Hend  and
Eugenio, Barbara Di  and
Schockaert, Steven},
  month = jan,
  pages = {1451--1464},
  publisher = {Association for Computational Linguistics},
  title = {Cross-lingual Text Classification Transfer: The Case of {U}krainian},
  url = {https://aclanthology.org/2025.coling-main.97/},
  year = {2025},
}
""",
    )
