from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LegalBenchConsumerContractsQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LegalBenchConsumerContractsQA",
        description="The dataset includes questions and answers related to contracts.",
        reference="https://huggingface.co/datasets/nguha/legalbench/viewer/consumer_contracts_qa",
        dataset={
            "path": "mteb/legalbench_consumer_contracts_qa",
            "revision": "b23590301ec94e8087e2850b21d43d4956b1cca9",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Legal", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-4.0",
        annotations_creators="derived",
        dialect=None,
        sample_creation="found",
        bibtex_citation=r"""
@article{hendrycks2021cuad,
  author = {Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
  journal = {arXiv preprint arXiv:2103.06268},
  title = {Cuad: An expert-annotated nlp dataset for legal contract review},
  year = {2021},
}

@article{koreeda2021contractnli,
  author = {Koreeda, Yuta and Manning, Christopher D},
  journal = {arXiv preprint arXiv:2110.01799},
  title = {ContractNLI: A dataset for document-level natural language inference for contracts},
  year = {2021},
}
""",
    )
