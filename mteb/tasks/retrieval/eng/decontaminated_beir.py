from __future__ import annotations

from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_DECONTAMINATED_BEIR_CITATION = r"""
@article{lighton2024decontaminated_beir,
  author = {Raphaël Stylianou and LightOn AI},
  title = {Decontaminated BEIR: Evaluating Dense and Late-Interaction Retrievers without Contamination},
  url = {https://huggingface.co/blog/lightonai/denseon-lateon#decontaminated-beir},
  year = {2024},
}

@inproceedings{thakur2021beir,
  author = {Nandan Thakur and Nils Reimers and Andreas R{\"u}ckl{\'e} and Abhishek Srivastava and Iryna Gurevych},
  booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  title = {{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
  url = {https://openreview.net/forum?id=wCu6T5xFjeJ},
  year = {2021},
}
"""


class ArguAnaDecontaminated(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="ArguAnaDecontaminated",
        description="Decontaminated version of ArguAna from the LightOn AI decontaminated BEIR suite, removing train-test contamination and data leakage.",
        reference="https://huggingface.co/datasets/lightonai/arguana-decontaminated",
        dataset={
            "path": "iamfortytwo/arguana-decontaminated",
            "revision": "137f25bbb59b60bb6807ecb11a0d7e66b92fdd11",
        },
        adapted_from=["ArguAna"],
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-11-01", "2024-11-30"),
        domains=["Social", "Web", "Written"],
        task_subtypes=["Discourse coherence"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_DECONTAMINATED_BEIR_CITATION,
        prompt={"query": "Given a claim, find documents that refute the claim"},
    )


class SciFactDecontaminated(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SciFactDecontaminated",
        description="Decontaminated version of SciFact from the LightOn AI decontaminated BEIR suite, verifying scientific claims using evidence from research literature while eliminating contamination.",
        reference="https://huggingface.co/datasets/lightonai/scifact-decontaminated",
        dataset={
            "path": "iamfortytwo/scifact-decontaminated",
            "revision": "ad05bded28ca603a0ed5e27231899dd72c1edb8e",
        },
        adapted_from=["SciFact"],
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-11-01", "2024-11-30"),
        domains=["Academic", "Medical", "Written"],
        task_subtypes=["Claim verification"],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_DECONTAMINATED_BEIR_CITATION,
        prompt={
            "query": "Given a scientific claim, retrieve documents that support or refute the claim"
        },
    )


class NFCorpusDecontaminated(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NFCorpusDecontaminated",
        description="Decontaminated version of NFCorpus from the LightOn AI decontaminated BEIR suite for medical information retrieval.",
        reference="https://huggingface.co/datasets/lightonai/nfcorpus-decontaminated",
        dataset={
            "path": "iamfortytwo/nfcorpus-decontaminated",
            "revision": "e6f37a2687336d5b7b83d9aede6e114e9d46dad9",
        },
        adapted_from=["NFCorpus"],
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-11-01", "2024-11-30"),
        domains=["Medical", "Academic", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_DECONTAMINATED_BEIR_CITATION,
        prompt={
            "query": "Given a question, retrieve relevant documents that best answer the question"
        },
    )


class FiQADecontaminated(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="FiQADecontaminated",
        description="Decontaminated version of FiQA 2018 from the LightOn AI decontaminated BEIR suite for financial opinion mining and QA.",
        reference="https://huggingface.co/datasets/lightonai/fiqa-decontaminated",
        dataset={
            "path": "iamfortytwo/fiqa-decontaminated",
            "revision": "4e693e028398575f83c17bc91515e81b13f534a2",
        },
        adapted_from=["FiQA2018"],
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-11-01", "2024-11-30"),
        domains=["Written", "Financial"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_DECONTAMINATED_BEIR_CITATION,
        prompt={
            "query": "Given a financial question, retrieve user replies that best answer the question"
        },
    )


class SciDocsDecontaminated(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SciDocsDecontaminated",
        description="Decontaminated version of SciDocs from the LightOn AI decontaminated BEIR suite for scientific document retrieval.",
        reference="https://huggingface.co/datasets/lightonai/scidocs-decontaminated",
        dataset={
            "path": "iamfortytwo/scidocs-decontaminated",
            "revision": "5b7a380cb7c8608e8c9bf6097acbc0eccf935de3",
        },
        adapted_from=["SCIDOCS"],
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-11-01", "2024-11-30"),
        domains=["Academic", "Written", "Non-fiction"],
        task_subtypes=["Scientific Reranking"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_DECONTAMINATED_BEIR_CITATION,
        prompt={
            "query": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper"
        },
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        """Drop the explicit negatives SCIDOCS ships in its qrels.

        SCIDOCS qrels carry score-0 rows for non-cited papers. Keeping them
        would count those documents as judged-relevant, so they are removed
        along with any query left without a positive judgement.
        """
        for subset in self.dataset:
            for split in self.dataset[subset]:
                split_data = self.dataset[subset][split]
                rel_docs = split_data["relevant_docs"]
                filtered_rel_docs = {
                    qid: {doc_id: score for doc_id, score in docs.items() if score > 0}
                    for qid, docs in rel_docs.items()
                }
                valid_qids = {
                    qid for qid, docs in filtered_rel_docs.items() if len(docs) > 0
                }
                split_data["relevant_docs"] = {
                    qid: filtered_rel_docs[qid] for qid in valid_qids
                }
                queries = split_data["queries"]
                indices = [
                    i for i, qid in enumerate(queries["id"]) if qid in valid_qids
                ]
                split_data["queries"] = queries.select(indices)


class TrecCOVIDDecontaminated(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TrecCOVIDDecontaminated",
        description="Decontaminated version of TREC-COVID from the LightOn AI decontaminated BEIR suite containing scientific articles on COVID-19.",
        reference="https://huggingface.co/datasets/lightonai/trec-covid-decontaminated",
        dataset={
            "path": "iamfortytwo/trec-covid-decontaminated",
            "revision": "0b99aeb603c9ad909564db1a80e380a2a78bc026",
        },
        adapted_from=["TRECCOVID"],
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-11-01", "2024-11-30"),
        domains=["Medical", "Academic", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_DECONTAMINATED_BEIR_CITATION,
        prompt={
            "query": "Given a query on COVID-19, retrieve documents that answer the query"
        },
    )


class Touche2020Decontaminated(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Touche2020Decontaminated",
        description="Decontaminated version of Touché 2020 from the LightOn AI decontaminated BEIR suite for controversial question argument retrieval.",
        reference="https://huggingface.co/datasets/lightonai/webis-touche2020-decontaminated",
        dataset={
            "path": "iamfortytwo/webis-touche2020-decontaminated",
            "revision": "bc4d287a9168dfb0ab2a94866cc17fc02c0dcb73",
        },
        adapted_from=["Touche2020"],
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-11-01", "2024-11-30"),
        domains=["Academic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_DECONTAMINATED_BEIR_CITATION,
        prompt={
            "query": "Given a question, retrieve detailed and persuasive arguments that answer the question"
        },
    )


class QuoraRetrievalDecontaminated(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="QuoraRetrievalDecontaminated",
        description="Decontaminated version of QuoraRetrieval from the LightOn AI decontaminated BEIR suite for duplicate question retrieval.",
        reference="https://huggingface.co/datasets/lightonai/quora-decontaminated",
        dataset={
            "path": "iamfortytwo/quora-decontaminated",
            "revision": "4d1fc7a24779fdd767c54e19e5606a843d3ecec1",
        },
        adapted_from=["QuoraRetrieval"],
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-11-01", "2024-11-30"),
        domains=["Written", "Web", "Blog"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_DECONTAMINATED_BEIR_CITATION,
        prompt={
            "query": "Given a question, retrieve questions that are semantically equivalent to the given question"
        },
    )
