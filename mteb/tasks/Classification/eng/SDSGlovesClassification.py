from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SDSGlovesClassification(AbsTaskClassification):
    superseded_by = "SDSGlovesClassification.v2"
    metadata = TaskMetadata(
        name="SDSGlovesClassification",
        description="ChemTEB evaluates the performance of text embedding models on chemical domain data.",
        reference="https://arxiv.org/abs/2412.00532",
        dataset={
            "path": "BASF-AI/SDSGlovesClassification",
            "revision": "c723236c5ec417d79512e6104aca9d2cd88168f6",
        },
        type="Classification",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-06-01", "2024-11-30"),
        domains=["Chemistry"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@article{kasmaee2024chemteb,
  author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
  journal = {arXiv preprint arXiv:2412.00532},
  title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
  year = {2024},
}

@inproceedings{pereira2020msds,
  author = {Pereira, Eliseu},
  booktitle = {15th Doctoral Symposium},
  pages = {42},
  title = {MSDS-OPP: Operator Procedures Prediction in Material Safety Data Sheets},
  year = {2020},
}
""",
    )


class SDSGlovesClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SDSGlovesClassification.v2",
        description="""ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)""",
        reference="https://arxiv.org/abs/2412.00532",
        dataset={
            "path": "mteb/sds_gloves",
            "revision": "09b09ee755ada02c68ad835971e22b1959d79448",
        },
        type="Classification",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-06-01", "2024-11-30"),
        domains=["Chemistry"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@article{kasmaee2024chemteb,
  author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
  journal = {arXiv preprint arXiv:2412.00532},
  title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
  year = {2024},
}

@inproceedings{pereira2020msds,
  author = {Pereira, Eliseu},
  booktitle = {15th Doctoral Symposium},
  pages = {42},
  title = {MSDS-OPP: Operator Procedures Prediction in Material Safety Data Sheets},
  year = {2020},
}
""",
        adapted_from=["SDSGlovesClassification"],
    )
