from datasets import load_dataset

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class DdiscoCohesionClassification(AbsTaskClassification):
    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = load_dataset(
            self.metadata_dict["hf_hub_name"],
            revision=self.metadata_dict.get("revision"),
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"rating": "label"}).remove_columns(
            ["domain"]
        )

    @property
    def metadata(self) -> TaskMetadata:
        return TaskMetadata(
            name="Ddisco",
            hf_hub_name="DDSC/ddisco",
            description="A Danish Discourse dataset with values for coherence and source (Wikipedia or Reddit)",
            reference="https://aclanthology.org/2022.lrec-1.260/",
            type="Classification",
            category="s2s",
            eval_splits=["test"],
            eval_langs=["da"],
            main_score="accuracy",
            revision="514ab557579fcfba538a4078d6d647248a0e6eb7",
            date=("2021-01-01", "2022-06-25"),
            form=["written"],
            domains=["Non-fiction", "Social"],
            dialect=[],
            task_subtypes=["Discourse coherence"],
            license="cc-by-sa-3.0",
            socioeconomic_status="high",
            annotations_creators="expert-annotated",
            text_creation="found",
            bibtex_citation="""
        @inproceedings{flansmose-mikkelsen-etal-2022-ddisco,
    title = "{DD}is{C}o: A Discourse Coherence Dataset for {D}anish",
    author = "Flansmose Mikkelsen, Linea  and
      Kinch, Oliver  and
      Jess Pedersen, Anders  and
      Lacroix, Oph{\'e}lie",
    editor = "Calzolari, Nicoletta  and
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
      Piperidis, Stelios",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.260",
    pages = "2440--2445",
    abstract = "To date, there has been no resource for studying discourse coherence on real-world Danish texts. Discourse coherence has mostly been approached with the assumption that incoherent texts can be represented by coherent texts in which sentences have been shuffled. However, incoherent real-world texts rarely resemble that. We thus present DDisCo, a dataset including text from the Danish Wikipedia and Reddit annotated for discourse coherence. We choose to annotate real-world texts instead of relying on artificially incoherent text for training and testing models. Then, we evaluate the performance of several methods, including neural networks, on the dataset.",
}
        """,
        )

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
