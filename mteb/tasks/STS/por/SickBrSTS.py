from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS

N_SAMPLES = 1000


class SickBrSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="SICK-BR-STS",
        dataset={
            "path": "eduagarcia/sick-br",
            "revision": "0cdfb1d51ef339011c067688a3b75b82f927c097",
        },
        description="SICK-BR is a Portuguese inference corpus, human translated from SICK",
        reference="https://linux.ime.usp.br/~thalen/SICK_PT.pdf",
        type="STS",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="cosine_spearman",
        date=("2018-01-01", "2018-09-01"),  # rough estimate
        domains=["Web", "Written"],
        task_subtypes=["Textual Entailment"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="human-translated and localized",
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
    and Camara, Igor C. S.
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
        descriptive_stats={
            "n_samples": {"test": N_SAMPLES},
            "avg_character_length": {"test": 54.89},
        },
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 1
        metadata_dict["max_score"] = 5
        return metadata_dict

    def dataset_transform(self):
        for split in self.dataset:
            self.dataset.update(
                {
                    split: self.dataset[split].train_test_split(
                        test_size=N_SAMPLES, seed=self.seed, label="entailment_label"
                    )["test"]
                }
            )

        self.dataset = self.dataset.rename_columns(
            {
                "sentence_A": "sentence1",
                "sentence_B": "sentence2",
                "relatedness_score": "score",
                "pair_ID": "id",
            }
        )
