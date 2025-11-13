from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class DdiscoCohesionClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Ddisco",
        dataset={
            "path": "DDSC/ddisco",
            "revision": "514ab557579fcfba538a4078d6d647248a0e6eb7",
        },
        description="A Danish Discourse dataset with values for coherence and source (Wikipedia or Reddit)",
        reference="https://aclanthology.org/2022.lrec-1.260/",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["dan-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2022-06-25"),
        domains=["Non-fiction", "Social", "Written"],
        dialect=[],
        task_subtypes=["Discourse coherence"],
        license="cc-by-sa-3.0",
        annotations_creators="expert-annotated",
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{flansmose-mikkelsen-etal-2022-ddisco,
  address = {Marseille, France},
  author = {Flansmose Mikkelsen, Linea  and
Kinch, Oliver  and
Jess Pedersen, Anders  and
Lacroix, Oph{\'e}lie},
  booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  editor = {Calzolari, Nicoletta  and
B{\'e}chet, Fr{\'e}d{\'e}ric  and
Blache, Philippe  and
Choukri, Khalid  and
Cieri, Christopher  and
Declerck, Thierry  and
Goggi, Sara  and
Isahara, Hitoshi  and
Maegaard, Bente  and
Mariani, Joseph  and
Mazo, H{\'e}l{\`e}ne  and
Odijk, Jan  and
Piperidis, Stelios},
  month = jun,
  pages = {2440--2445},
  publisher = {European Language Resources Association},
  title = {{DD}is{C}o: A Discourse Coherence Dataset for {D}anish},
  url = {https://aclanthology.org/2022.lrec-1.260},
  year = {2022},
}
""",
        superseded_by="Ddisco.v2",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"rating": "label"}).remove_columns(
            ["domain"]
        )


class DdiscoCohesionClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Ddisco.v2",
        dataset={
            "path": "mteb/ddisco_cohesion",
            "revision": "b5a05bdecdfc6efc14eebc8f7a86e0986edaf5ff",
        },
        description="A Danish Discourse dataset with values for coherence and source (Wikipedia or Reddit) This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://aclanthology.org/2022.lrec-1.260/",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["dan-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2022-06-25"),
        domains=["Non-fiction", "Social", "Written"],
        dialect=[],
        task_subtypes=["Discourse coherence"],
        license="cc-by-sa-3.0",
        annotations_creators="expert-annotated",
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{flansmose-mikkelsen-etal-2022-ddisco,
  address = {Marseille, France},
  author = {Flansmose Mikkelsen, Linea  and
Kinch, Oliver  and
Jess Pedersen, Anders  and
Lacroix, Oph{\'e}lie},
  booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  editor = {Calzolari, Nicoletta  and
B{\'e}chet, Fr{\'e}d{\'e}ric  and
Blache, Philippe  and
Choukri, Khalid  and
Cieri, Christopher  and
Declerck, Thierry  and
Goggi, Sara  and
Isahara, Hitoshi  and
Maegaard, Bente  and
Mariani, Joseph  and
Mazo, H{\'e}l{\`e}ne  and
Odijk, Jan  and
Piperidis, Stelios},
  month = jun,
  pages = {2440--2445},
  publisher = {European Language Resources Association},
  title = {{DD}is{C}o: A Discourse Coherence Dataset for {D}anish},
  url = {https://aclanthology.org/2022.lrec-1.260},
  year = {2022},
}
""",
        adapted_from=["DdiscoCohesionClassification"],
    )
