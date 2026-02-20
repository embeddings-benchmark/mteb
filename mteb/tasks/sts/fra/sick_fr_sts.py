from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class SickFrSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="SICKFr",
        dataset={
            "path": "Lajavaness/SICK-fr",
            "revision": "e077ab4cf4774a1e36d86d593b150422fafd8e8a",
        },
        description="SICK dataset french version",
        reference="https://huggingface.co/datasets/Lajavaness/SICK-fr",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["fra-Latn"],
        main_score="cosine_spearman",
        date=("2014-01-01", "2024-12-31"),
        domains=["Web", "Written"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@inproceedings{marelli-etal-2014-sick,
  address = {Reykjavik, Iceland},
  author = {Marelli, Marco  and
Menini, Stefano  and
Baroni, Marco  and
Bentivogli, Luisa  and
Bernardi, Raffaella  and
Zamparelli, Roberto},
  booktitle = {Proceedings of the Ninth International Conference on Language Resources and Evaluation ({LREC}'14)},
  month = may,
  pages = {216--223},
  publisher = {European Language Resources Association (ELRA)},
  title = {A {SICK} cure for the evaluation of compositional distributional semantic models},
  url = {http://www.lrec-conf.org/proceedings/lrec2014/pdf/363_Paper.pdf},
  year = {2014},
}
""",
    )

    min_score = 0
    max_score = 5

    def dataset_transform(self, num_proc: int = 1):
        self.dataset = self.dataset.rename_columns(
            {
                "sentence_A": "sentence1",
                "sentence_B": "sentence2",
                "relatedness_score": "score",
                "Unnamed: 0": "id",
            }
        )
