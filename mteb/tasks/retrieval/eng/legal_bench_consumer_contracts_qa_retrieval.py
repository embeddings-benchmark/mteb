from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


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
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2023-12-31"),
        domains=["Legal", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{hendrycks2021cuad,
  archiveprefix = {arXiv},
  author = {Dan Hendrycks and Collin Burns and Anya Chen and Spencer Ball},
  booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
  eprint = {2103.06268},
  title = {{CUAD}: An Expert-Annotated {NLP} Dataset for Legal Contract Review},
  url = {https://openreview.net/forum?id=7l1Ygs3Bamw},
  year = {2021},
}

@inproceedings{koreeda2021contractnli,
  address = {Punta Cana, Dominican Republic},
  author = {Koreeda, Yuta and Manning, Christopher},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2021},
  doi = {10.18653/v1/2021.findings-emnlp.164},
  editor = {Moens, Marie-Francine and Huang, Xuanjing and Specia, Lucia and Yih, Scott Wen-tau},
  month = {nov},
  pages = {1907--1919},
  publisher = {Association for Computational Linguistics},
  title = {{C}ontract{NLI}: A Dataset for Document-level Natural Language Inference for Contracts},
  url = {https://aclanthology.org/2021.findings-emnlp.164/},
  year = {2021},
}
""",
    )
