from collections import defaultdict

import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def load_r2med_data(
    path: str,
    eval_splits: list,
    revision: str,
):
    eval_split = eval_splits[0]
    corpus = {eval_split: None}
    queries = {eval_split: None}
    relevant_docs = {eval_split: None}
    domain_corpus = datasets.load_dataset(
        path, name="corpus", split="corpus", revision=revision
    )
    domain_queries = datasets.load_dataset(
        path, name="query", split="query", revision=revision
    )
    domain_qrels = datasets.load_dataset(
        path, name="qrels", split="qrels", revision=revision
    )
    corpus[eval_split] = {e["id"]: {"text": e["text"]} for e in domain_corpus}
    queries[eval_split] = {e["id"]: e["text"] for e in domain_queries}
    relevant_docs[eval_split] = defaultdict(dict)
    for e in domain_qrels:
        qid = e["q_id"]
        pid = e["p_id"]
        relevant_docs[eval_split][qid][pid] = int(e["score"])

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)
    return corpus, queries, relevant_docs


class R2MEDBiologyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="R2MEDBiologyRetrieval",
        dataset={
            "path": "R2MED/Biology",
            "revision": "8b9fec2db9eda4b5742d03732213fbaee8169556",
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
@article{li2025r2med,
  author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
  journal = {arXiv preprint arXiv:2505.14558},
  title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  year = {2025},
}
""",
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_r2med_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True


class R2MEDBioinformaticsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="R2MEDBioinformaticsRetrieval",
        dataset={
            "path": "R2MED/Bioinformatics",
            "revision": "6021fce366892cbfd7837fa85a4128ea93315e18",
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
@article{li2025r2med,
  author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
  journal = {arXiv preprint arXiv:2505.14558},
  title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  year = {2025},
}
""",
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_r2med_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True


class R2MEDMedicalSciencesRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="R2MEDMedicalSciencesRetrieval",
        dataset={
            "path": "R2MED/Medical-Sciences",
            "revision": "1b48911514c80bf9182222d99752ad75e23b4b47",
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
@article{li2025r2med,
  author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
  journal = {arXiv preprint arXiv:2505.14558},
  title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  year = {2025},
}
""",
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_r2med_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True


class R2MEDMedXpertQAExamRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="R2MEDMedXpertQAExamRetrieval",
        dataset={
            "path": "R2MED/MedXpertQA-Exam",
            "revision": "b457ea43db9ae5db74c3a3e5be0a213d0f85ac3a",
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
@article{li2025r2med,
  author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
  journal = {arXiv preprint arXiv:2505.14558},
  title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  year = {2025},
}
""",
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_r2med_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True


class R2MEDMedQADiagRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="R2MEDMedQADiagRetrieval",
        dataset={
            "path": "R2MED/MedQA-Diag",
            "revision": "78b585990279cc01a493f876c1b0cf09557fba57",
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
@article{li2025r2med,
  author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
  journal = {arXiv preprint arXiv:2505.14558},
  title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  year = {2025},
}
""",
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_r2med_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True


class R2MEDPMCTreatmentRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="R2MEDPMCTreatmentRetrieval",
        dataset={
            "path": "R2MED/PMC-Treatment",
            "revision": "53c489a44a3664ba352c07550b72b4525a5968d5",
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
@article{li2025r2med,
  author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
  journal = {arXiv preprint arXiv:2505.14558},
  title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  year = {2025},
}
""",
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_r2med_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True


class R2MEDPMCClinicalRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="R2MEDPMCClinicalRetrieval",
        dataset={
            "path": "R2MED/PMC-Clinical",
            "revision": "812829522f7eaa407ef82b96717be85788a50f7e",
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
@article{li2025r2med,
  author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
  journal = {arXiv preprint arXiv:2505.14558},
  title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  year = {2025},
}
""",
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_r2med_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True


class R2MEDIIYiClinicalRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="R2MEDIIYiClinicalRetrieval",
        dataset={
            "path": "R2MED/IIYi-Clinical",
            "revision": "974abbc9bc281c3169180a6aa5d7586cfd2f5877",
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
@article{li2025r2med,
  author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
  journal = {arXiv preprint arXiv:2505.14558},
  title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  year = {2025},
}
""",
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_r2med_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True
