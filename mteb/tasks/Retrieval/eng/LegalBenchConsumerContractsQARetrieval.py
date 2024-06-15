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
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=["written"],
        domains=["Legal"],
        task_subtypes=["Question answering"],
        license="CC BY-NC 4.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation="""@article{koreeda2021contractnli,
  title={ContractNLI: A dataset for document-level natural language inference for contracts},
  author={Koreeda, Yuta and Manning, Christopher D},
  journal={arXiv preprint arXiv:2110.01799},
  year={2021}
  }
 
  @article{hendrycks2021cuad,
  title={Cuad: An expert-annotated nlp dataset for legal contract review},
  author={Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
  journal={arXiv preprint arXiv:2103.06268},
  year={2021}
}""",
        n_samples=None,
        avg_character_length=None,
    )
