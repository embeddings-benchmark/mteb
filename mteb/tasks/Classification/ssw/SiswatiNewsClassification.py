from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SiswatiNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SiswatiNewsClassification",
        description="Siswati News Classification Dataset",
        reference="https://huggingface.co/datasets/dsfsi/za-isizulu-siswati-news",
        dataset={
            "path": "isaacchung/siswati-news",
            "revision": "f5502326c4e48adc99b18b1582f68b8fb5e7ec30",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["ssw-Latn"],
        main_score="accuracy",
        date=("2022-08-01", "2022-08-01"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{Madodonga_Marivate_Adendorff_2023, title={Izindaba-Tindzaba: Machine learning news categorisation for Long and Short Text for isiZulu and Siswati}, volume={4}, url={https://upjournals.up.ac.za/index.php/dhasa/article/view/4449}, DOI={10.55492/dhasa.v4i01.4449}, author={Madodonga, Andani and Marivate, Vukosi and Adendorff, Matthew}, year={2023}, month={Jan.} }
        """,
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"title": "text"})
