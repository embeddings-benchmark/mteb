from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class R2MEDBiologyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="R2MEDBiologyRetrieval",
        dataset={
            "path": "mteb/R2MEDBiologyRetrieval",
            "revision": "5784d4f7ca11b0467bbce42ffd48fa82c101a40c",
        },
        reference="https://huggingface.co/datasets/R2MED/Biology",
        description="Biology retrieval dataset.",
        type="Retrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2011-01-01", "2024-06-30"),
        domains=["Medical"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=r"""
@inproceedings{zhang2026r2medbenchmarkreasoningdrivenmedical,
  author = {Zhang, Xiangxu and Li, Lei and Liu, Zheng and Zhou, Xiao},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/2026.emnlp-main.2138},
  pages = {48600–48637},
  publisher = {Association for Computational Linguistics},
  title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  url = {https://doi.org/10.18653/2026.emnlp-main.2138},
  year = {2026},
}
""",
    )


class R2MEDBioinformaticsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="R2MEDBioinformaticsRetrieval",
        dataset={
            "path": "mteb/R2MEDBioinformaticsRetrieval",
            "revision": "cd73a5e8eeaef9b8dfbcc7de2fb4dd858acfe93f",
        },
        reference="https://huggingface.co/datasets/R2MED/Bioinformatics",
        description="Bioinformatics retrieval dataset.",
        type="Retrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2011-01-01", "2025-05-31"),
        domains=["Medical"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=r"""
@inproceedings{zhang2026r2medbenchmarkreasoningdrivenmedical,
  author = {Zhang, Xiangxu and Li, Lei and Liu, Zheng and Zhou, Xiao},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/2026.emnlp-main.2138},
  pages = {48600–48637},
  publisher = {Association for Computational Linguistics},
  title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  url = {https://doi.org/10.18653/2026.emnlp-main.2138},
  year = {2026},
}
""",
    )


class R2MEDMedicalSciencesRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="R2MEDMedicalSciencesRetrieval",
        dataset={
            "path": "mteb/R2MEDMedicalSciencesRetrieval",
            "revision": "acc8238076e63c0de24fb41de597c2cb1adef8db",
        },
        reference="https://huggingface.co/datasets/R2MED/Medical-Sciences",
        description="Medical-Sciences retrieval dataset.",
        type="Retrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        domains=["Medical"],
        date=("2011-01-01", "2025-05-31"),
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=r"""
@inproceedings{zhang2026r2medbenchmarkreasoningdrivenmedical,
  author = {Zhang, Xiangxu and Li, Lei and Liu, Zheng and Zhou, Xiao},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/2026.emnlp-main.2138},
  pages = {48600–48637},
  publisher = {Association for Computational Linguistics},
  title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  url = {https://doi.org/10.18653/2026.emnlp-main.2138},
  year = {2026},
}
""",
    )


class R2MEDMedXpertQAExamRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="R2MEDMedXpertQAExamRetrieval",
        dataset={
            "path": "mteb/R2MEDMedXpertQAExamRetrieval",
            "revision": "8cd897ce426f9d7ad7a62aa4aeccd7b04477eb5a",
        },
        reference="https://huggingface.co/datasets/R2MED/MedXpertQA-Exam",
        description="MedXpertQA-Exam retrieval dataset.",
        type="Retrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("1995-01-01", "2025-01-31"),
        domains=["Medical"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=r"""
@inproceedings{zhang2026r2medbenchmarkreasoningdrivenmedical,
  author = {Zhang, Xiangxu and Li, Lei and Liu, Zheng and Zhou, Xiao},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/2026.emnlp-main.2138},
  pages = {48600–48637},
  publisher = {Association for Computational Linguistics},
  title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  url = {https://doi.org/10.18653/2026.emnlp-main.2138},
  year = {2026},
}
""",
    )


class R2MEDMedQADiagRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="R2MEDMedQADiagRetrieval",
        dataset={
            "path": "mteb/R2MEDMedQADiagRetrieval",
            "revision": "2a5aa3eeb2413762152587b0e40bfc78905ed8af",
        },
        reference="https://huggingface.co/datasets/R2MED/MedQA-Diag",
        description="MedQA-Diag retrieval dataset.",
        type="Retrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("1992-01-01", "2020-09-30"),
        domains=["Medical"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=r"""
@inproceedings{zhang2026r2medbenchmarkreasoningdrivenmedical,
  author = {Zhang, Xiangxu and Li, Lei and Liu, Zheng and Zhou, Xiao},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/2026.emnlp-main.2138},
  pages = {48600–48637},
  publisher = {Association for Computational Linguistics},
  title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  url = {https://doi.org/10.18653/2026.emnlp-main.2138},
  year = {2026},
}
""",
    )


class R2MEDPMCTreatmentRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="R2MEDPMCTreatmentRetrieval",
        dataset={
            "path": "mteb/R2MEDPMCTreatmentRetrieval",
            "revision": "c46cb9cb83bf94a5bce9cc70b1bc98b15cebfe91",
        },
        reference="https://huggingface.co/datasets/R2MED/PMC-Treatment",
        description="PMC-Treatment retrieval dataset.",
        type="Retrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2000-02-01", "2025-03-31"),
        domains=["Medical"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=r"""
@inproceedings{zhang2026r2medbenchmarkreasoningdrivenmedical,
  author = {Zhang, Xiangxu and Li, Lei and Liu, Zheng and Zhou, Xiao},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/2026.emnlp-main.2138},
  pages = {48600–48637},
  publisher = {Association for Computational Linguistics},
  title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  url = {https://doi.org/10.18653/2026.emnlp-main.2138},
  year = {2026},
}
""",
    )


class R2MEDPMCClinicalRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="R2MEDPMCClinicalRetrieval",
        dataset={
            "path": "mteb/R2MEDPMCClinicalRetrieval",
            "revision": "d97d0aa167d9c8d5fc87374a2db20b2e183d25f5",
        },
        reference="https://huggingface.co/datasets/R2MED/PMC-Clinical",
        description="PMC-Clinical retrieval dataset.",
        type="Retrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2000-02-01", "2023-04-30"),
        domains=["Medical"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=r"""
@inproceedings{zhang2026r2medbenchmarkreasoningdrivenmedical,
  author = {Zhang, Xiangxu and Li, Lei and Liu, Zheng and Zhou, Xiao},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/2026.emnlp-main.2138},
  pages = {48600–48637},
  publisher = {Association for Computational Linguistics},
  title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  url = {https://doi.org/10.18653/2026.emnlp-main.2138},
  year = {2026},
}
""",
    )


class R2MEDIIYiClinicalRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="R2MEDIIYiClinicalRetrieval",
        dataset={
            "path": "mteb/R2MEDIIYiClinicalRetrieval",
            "revision": "63494d49fe21b369198c4bc013f2a305070013c4",
        },
        reference="https://huggingface.co/datasets/R2MED/IIYi-Clinical",
        description="IIYi-Clinical retrieval dataset.",
        type="Retrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2002-07-01", "2025-05-31"),
        domains=["Medical"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=r"""
@inproceedings{zhang2026r2medbenchmarkreasoningdrivenmedical,
  author = {Zhang, Xiangxu and Li, Lei and Liu, Zheng and Zhou, Xiao},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/2026.emnlp-main.2138},
  pages = {48600–48637},
  publisher = {Association for Computational Linguistics},
  title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  url = {https://doi.org/10.18653/2026.emnlp-main.2138},
  year = {2026},
}
""",
    )
