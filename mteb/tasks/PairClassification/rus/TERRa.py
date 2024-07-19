from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class TERRa(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="TERRa",
        dataset={
            "path": "ai-forever/terra-pairclassification",
            "revision": "7b58f24536063837d644aab9a023c62199b2a612",
        },
        description="Textual Entailment Recognition for Russian. This task requires to recognize, given two text fragments, "
        "whether the meaning of one text is entailed (can be inferred) from the other text.",
        reference="https://arxiv.org/pdf/2010.15925",
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["rus-Cyrl"],
        main_score="max_ap",
        date=("2000-01-01", "2018-01-01"),
        domains=["News", "Web", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{shavrina2020russiansuperglue,
        title={RussianSuperGLUE: A Russian Language Understanding Evaluation Benchmark},
        author={Shavrina, Tatiana 
                    and Fenogenova, Alena 
                    and Emelyanov, Anton 
                    and Shevelev, Denis 
                    and Artemova, Ekaterina 
                    and Malykh, Valentin 
                    and Mikhailov, Vladislav 
                    and Tikhonova, Maria 
                    and Chertok, Andrey 
                    and Evlampiev, Andrey},
        journal={arXiv preprint arXiv:2010.15925},
        year={2020}
        }""",
        descriptive_stats={
            "n_samples": {"dev": 307},
            "avg_character_length": {"dev": 138.2},
        },
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")
