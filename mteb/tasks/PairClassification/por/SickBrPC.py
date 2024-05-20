from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

N_SAMPLES = 1000


class SickBrPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SICK-BR-PC",
        dataset={
            "path": "eduagarcia/sick-br",
            "revision": "0cdfb1d51ef339011c067688a3b75b82f927c097",
        },
        description="SICK-BR is a Portuguese inference corpus, human translated from SICK",
        reference="https://linux.ime.usp.br/~thalen/SICK_PT.pdf",
        type="PairClassification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="ap",
        date=("2018-01-01", "2018-09-01"),  # rough estimate
        form=["written"],
        domains=["Web"],
        task_subtypes=["Textual Entailment"],
        license="unknown",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="human-translated and localized",
        bibtex_citation="""
@inproceedings{real18,
  author="Real, Livy
    and Rodrigues, Ana
    and Vieira e Silva, Andressa
    and Albiero, Beatriz
    and Thalenberg, Bruna
    and Guide, Bruno
    and Silva, Cindy
    and de Oliveira Lima, Guilherme
    and C{\^a}mara, Igor C. S.
    and Stanojevi{\'{c}}, Milo{\v{s}}
    and Souza, Rodrigo
    and de Paiva, Valeria"
  year ="2018",
  title="SICK-BR: A Portuguese Corpus for Inference",
  booktitle="Computational Processing of the Portuguese Language. PROPOR 2018.",
  doi ="10.1007/978-3-319-99722-3_31",
  isbn="978-3-319-99722-3"
}
        """,
        n_samples={"test": N_SAMPLES},
        avg_character_length={"test": 54.89},
    )

    def dataset_transform(self):
        _dataset = {}

        # Do not process the subsets we won't use
        self.dataset.pop("train")
        self.dataset.pop("validation")

        self.dataset = self.stratified_subsampling(
            self.dataset,
            seed=self.seed,
            splits=self.metadata.eval_splits,
            label="entailment_label",
            n_samples=N_SAMPLES,
        )

        for split in self.metadata.eval_splits:
            print(self.dataset[split]["entailment_label"])
            # keep labels 0=entailment and 2=contradiction, and map them as 1 and 0 for binary classification
            hf_dataset = self.dataset[split].filter(
                lambda x: x["entailment_label"] in [0, 2]
            )
            hf_dataset = hf_dataset.map(
                lambda example: {"label": 0 if example["entailment_label"] == 2 else 1}
            )
            _dataset[split] = [
                {
                    "sent1": hf_dataset["sentence_A"],
                    "sent2": hf_dataset["sentence_B"],
                    "labels": hf_dataset["label"],
                }
            ]
        self.dataset = _dataset
