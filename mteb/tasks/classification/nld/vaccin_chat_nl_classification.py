from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class VaccinChatNLClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="VaccinChatNLClassification",
        description="VaccinChatNL is a Flemish Dutch FAQ dataset on the topic of COVID-19 vaccinations in Flanders.",
        reference="https://huggingface.co/datasets/clips/VaccinChatNL",
        dataset={
            "path": "clips/VaccinChatNL",
            "revision": "bd27d0058bea2ad52470d9072a3b5da6b97c1ac3",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2022-01-01", "2022-09-01"),
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="f1",
        domains=["Spoken", "Web"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{buhmann-etal-2022-domain,
  address = {Gyeongju, Republic of Korea},
  author = {Buhmann, Jeska and De Bruyn, Maxime and Lotfi, Ehsan and Daelemans, Walter},
  booktitle = {Proceedings of the 29th International Conference on Computational Linguistics},
  month = oct,
  pages = {3539--3549},
  publisher = {International Committee on Computational Linguistics},
  title = {Domain- and Task-Adaptation for {V}accin{C}hat{NL}, a {D}utch {COVID}-19 {FAQ} Answering Corpus and Classification Model},
  url = {https://aclanthology.org/2022.coling-1.312},
  year = {2022},
}
""",
        prompt={
            "query": "Gegeven een gebruikersuiting als query, bepaal de gebruikersintenties"
        },
    )

    def dataset_transform(self):
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].rename_columns(
                {"sentence1": "text"}
            )
