"""NanoBEIR with rubric-calibrated preference (RCP) gains.

Each task adapts an existing mteb task (see `adapted_from`): the same queries and corpus, plus a
candidate pool (`top_ranked`) and continuous relevance gains in the qrels' `gain` column.
Scored by `ndcg_float_at_10`, which each task adds in `task_specific_scores` from the `gain`
column (`load_float_gains`, `ndcg_float_scores`). Metadata is copied from the original task
except for name, description, reference, dataset, eval_langs, main_score, annotations_creators,
citation and adapted_from.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import datasets

from mteb._evaluators.retrieval_metrics import ndcg_float_scores
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from mteb.types import RelevantDocumentsType

_CITATION = r"""@misc{schmidt2026rubriccalibratedpreferencescrossquerycalibration,
  title = {Rubric-Calibrated Preferences: Cross-Query Calibration of LLM Judgments via Item Response Theory},
  author = {Fabian David Schmidt and Donato Crisostomi and Carlos Lassance and Nils Reimers},
  year = {2026},
  eprint = {2609.35739},
  archivePrefix = {arXiv},
  primaryClass = {cs.IR},
  url = {https://arxiv.org/abs/2609.35739},
}
"""


def load_float_gains(
    metadata: TaskMetadata, hf_subset: str, split: str
) -> dict[str, dict[str, float]]:
    """Load the float `gain` column of a subset's qrels.

    The standard loader keeps only the integer `score`. Rows whose `gain` is null are skipped.
    """
    qrels = datasets.load_dataset(
        metadata.dataset["path"],
        f"{hf_subset}-qrels",
        split=split,
        revision=metadata.dataset["revision"],
    )
    gains: dict[str, dict[str, float]] = defaultdict(dict)
    for query_id, doc_id, gain in zip(
        qrels["query-id"], qrels["corpus-id"], qrels["gain"], strict=True
    ):
        if gain is not None:
            gains[str(query_id)][str(doc_id)] = float(gain)
    return dict(gains)


class NanoArguAnaRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoArguAnaRCPRetrieval",
        description="NanoArguAna is a smaller subset of ArguAna, a dataset for argument retrieval in debate contexts. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The human integer qrels are unchanged and still drive `ndcg_at_10`, which here reranks the fixed pool. The query argument's own copy in the corpus (21 of 50 queries) is removed from the candidates.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-nanobeir",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-nanobeir",
            "revision": "c0c43c5790a808cb0e8ee833bc7487dc93e88cd5",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs={"NanoArguAnaRetrieval": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2020-01-01", "2020-12-31"),
        domains=["Social", "Web", "Written"],
        task_subtypes=["Discourse coherence"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@inproceedings{wachsmuth2018retrieval,\n  author = {Wachsmuth, Henning and Syed, Shahbaz and Stein, Benno},\n  booktitle = {ACL},\n  title = {Retrieval of the Best Counterargument without Prior Topic Knowledge},\n  year = {2018},\n}\n",
        adapted_from=["NanoArguAnaRetrieval"],
        prompt={"query": "Given a claim, find documents that refute the claim"},
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        gains = load_float_gains(self.metadata, hf_subset, hf_split)
        return ndcg_float_scores(gains, results, self.k_values)


class NanoClimateFeverRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoClimateFeverRCPRetrieval",
        description="NanoClimateFever is a small version of the BEIR dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The human integer qrels are unchanged and still drive `ndcg_at_10`, which here reranks the fixed pool.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-nanobeir",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-nanobeir",
            "revision": "c0c43c5790a808cb0e8ee833bc7487dc93e88cd5",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs={"NanoClimateFeverRetrieval": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2020-01-01", "2020-12-31"),
        domains=["Non-fiction", "Academic", "News"],
        task_subtypes=["Claim verification"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@misc{diggelmann2021climatefever,\n  archiveprefix = {arXiv},\n  author = {Thomas Diggelmann and Jordan Boyd-Graber and Jannis Bulian and Massimiliano Ciaramita and Markus Leippold},\n  eprint = {2012.00614},\n  primaryclass = {cs.CL},\n  title = {CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims},\n  year = {2021},\n}\n",
        adapted_from=["NanoClimateFeverRetrieval"],
        prompt={
            "query": "Given a claim about climate change, retrieve documents that support or refute the claim"
        },
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        gains = load_float_gains(self.metadata, hf_subset, hf_split)
        return ndcg_float_scores(gains, results, self.k_values)


class NanoDBPediaRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoDBPediaRCPRetrieval",
        description="NanoDBPediaRetrieval is a small version of the standard test collection for entity search over the DBpedia knowledge base. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The human integer qrels are unchanged and still drive `ndcg_at_10`, which here reranks the fixed pool.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-nanobeir",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-nanobeir",
            "revision": "c0c43c5790a808cb0e8ee833bc7487dc93e88cd5",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs={"NanoDBPediaRetrieval": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2015-01-01", "2015-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@article{lehmann2015dbpedia,\n  author = {Lehmann, Jens and et al.},\n  journal = {Semantic Web},\n  title = {DBpedia: A large-scale, multilingual knowledge base extracted from Wikipedia},\n  year = {2015},\n}\n",
        adapted_from=["NanoDBPediaRetrieval"],
        prompt={
            "query": "Given a query, retrieve relevant entity descriptions from DBPedia"
        },
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        gains = load_float_gains(self.metadata, hf_subset, hf_split)
        return ndcg_float_scores(gains, results, self.k_values)


class NanoFEVERRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoFEVERRCPRetrieval",
        description="NanoFEVER is a smaller version of FEVER (Fact Extraction and VERification), which consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The human integer qrels are unchanged and still drive `ndcg_at_10`, which here reranks the fixed pool.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-nanobeir",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-nanobeir",
            "revision": "c0c43c5790a808cb0e8ee833bc7487dc93e88cd5",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs={"NanoFEVERRetrieval": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["Academic", "Encyclopaedic"],
        task_subtypes=["Claim verification"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@inproceedings{thorne-etal-2018-fever,\n  address = {New Orleans, Louisiana},\n  author = {Thorne, James  and\nVlachos, Andreas  and\nChristodoulopoulos, Christos  and\nMittal, Arpit},\n  booktitle = {Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},\n  doi = {10.18653/v1/N18-1074},\n  editor = {Walker, Marilyn  and\nJi, Heng  and\nStent, Amanda},\n  month = jun,\n  pages = {809--819},\n  publisher = {Association for Computational Linguistics},\n  title = {{FEVER}: a Large-scale Dataset for Fact Extraction and {VER}ification},\n  url = {https://aclanthology.org/N18-1074},\n  year = {2018},\n}\n",
        adapted_from=["NanoFEVERRetrieval"],
        prompt={
            "query": "Given a claim, retrieve documents that support or refute the claim"
        },
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        gains = load_float_gains(self.metadata, hf_subset, hf_split)
        return ndcg_float_scores(gains, results, self.k_values)


class NanoFiQA2018RCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoFiQA2018RCPRetrieval",
        description="NanoFiQA2018 is a smaller subset of the Financial Opinion Mining and Question Answering dataset. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The human integer qrels are unchanged and still drive `ndcg_at_10`, which here reranks the fixed pool.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-nanobeir",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-nanobeir",
            "revision": "c0c43c5790a808cb0e8ee833bc7487dc93e88cd5",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs={"NanoFiQA2018Retrieval": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["Academic", "Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@inproceedings{thakur2021beir,\n  archiveprefix = {arXiv},\n  author = {Nandan Thakur and Nils Reimers and Andreas R{\\\"u}ckl{\\'e} and Abhishek Srivastava and Iryna Gurevych},\n  booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},\n  eprint = {2104.08663},\n  title = {{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},\n  url = {https://openreview.net/forum?id=wCu6T5xFjeJ},\n  year = {2021},\n}\n",
        adapted_from=["NanoFiQA2018Retrieval"],
        prompt={
            "query": "Given a financial question, retrieve user replies that best answer the question"
        },
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        gains = load_float_gains(self.metadata, hf_subset, hf_split)
        return ndcg_float_scores(gains, results, self.k_values)


class NanoHotpotQARCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoHotpotQARCPRetrieval",
        description="NanoHotpotQARetrieval is a smaller subset of the HotpotQA dataset, which is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The human integer qrels are unchanged and still drive `ndcg_at_10`, which here reranks the fixed pool.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-nanobeir",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-nanobeir",
            "revision": "c0c43c5790a808cb0e8ee833bc7487dc93e88cd5",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs={"NanoHotpotQARetrieval": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@inproceedings{yang-etal-2018-hotpotqa,\n  address = {Brussels, Belgium},\n  author = {Yang, Zhilin  and\nQi, Peng  and\nZhang, Saizheng  and\nBengio, Yoshua  and\nCohen, William  and\nSalakhutdinov, Ruslan  and\nManning, Christopher D.},\n  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},\n  doi = {10.18653/v1/D18-1259},\n  editor = {Riloff, Ellen  and\nChiang, David  and\nHockenmaier, Julia  and\nTsujii, Jun{'}ichi},\n  month = oct # {-} # nov,\n  pages = {2369--2380},\n  publisher = {Association for Computational Linguistics},\n  title = {{H}otpot{QA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},\n  url = {https://aclanthology.org/D18-1259},\n  year = {2018},\n}\n",
        adapted_from=["NanoHotpotQARetrieval"],
        prompt={
            "query": "Given a multi-hop question, retrieve documents that can help answer the question"
        },
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        gains = load_float_gains(self.metadata, hf_subset, hf_split)
        return ndcg_float_scores(gains, results, self.k_values)


class NanoMSMARCORCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoMSMARCORCPRetrieval",
        description="NanoMSMARCORetrieval is a smaller subset of MS MARCO, a collection of datasets focused on deep learning in search. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The human integer qrels are unchanged and still drive `ndcg_at_10`, which here reranks the fixed pool.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-nanobeir",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-nanobeir",
            "revision": "c0c43c5790a808cb0e8ee833bc7487dc93e88cd5",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs={"NanoMSMARCORetrieval": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2016-01-01", "2016-12-31"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@misc{bajaj2018msmarcohumangenerated,\n  archiveprefix = {arXiv},\n  author = {Payal Bajaj and Daniel Campos and Nick Craswell and Li Deng and Jianfeng Gao and Xiaodong Liu and Rangan Majumder and Andrew McNamara and Bhaskar Mitra and Tri Nguyen and Mir Rosenberg and Xia Song and Alina Stoica and Saurabh Tiwary and Tong Wang},\n  eprint = {1611.09268},\n  primaryclass = {cs.CL},\n  title = {MS MARCO: A Human Generated MAchine Reading COmprehension Dataset},\n  url = {https://arxiv.org/abs/1611.09268},\n  year = {2018},\n}\n",
        adapted_from=["NanoMSMARCORetrieval"],
        prompt={
            "query": "Given a web search query, retrieve relevant passages that answer the query"
        },
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        gains = load_float_gains(self.metadata, hf_subset, hf_split)
        return ndcg_float_scores(gains, results, self.k_values)


class NanoNFCorpusRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoNFCorpusRCPRetrieval",
        description="NanoNFCorpus is a smaller subset of NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The human integer qrels are unchanged and still drive `ndcg_at_10`, which here reranks the fixed pool.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-nanobeir",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-nanobeir",
            "revision": "c0c43c5790a808cb0e8ee833bc7487dc93e88cd5",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs={"NanoNFCorpusRetrieval": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2016-01-01", "2016-12-31"),
        domains=["Medical", "Academic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@inproceedings{boteva2016,\n  author = {Boteva, Vera and Gholipour, Demian and Sokolov, Artem and Riezler, Stefan},\n  city = {Padova},\n  country = {Italy},\n  journal = {Proceedings of the 38th European Conference on Information Retrieval},\n  journal-abbrev = {ECIR},\n  title = {A Full-Text Learning to Rank Dataset for Medical Information Retrieval},\n  url = {http://www.cl.uni-heidelberg.de/~riezler/publications/papers/ECIR2016.pdf},\n  year = {2016},\n}\n",
        adapted_from=["NanoNFCorpusRetrieval"],
        prompt={
            "query": "Given a question, retrieve relevant documents that best answer the question"
        },
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        gains = load_float_gains(self.metadata, hf_subset, hf_split)
        return ndcg_float_scores(gains, results, self.k_values)


class NanoNQRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoNQRCPRetrieval",
        description="NanoNQ is a smaller subset of a dataset which contains questions from real users, and it requires QA systems to read and comprehend an entire Wikipedia article that may or may not contain the answer to the question. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The human integer qrels are unchanged and still drive `ndcg_at_10`, which here reranks the fixed pool.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-nanobeir",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-nanobeir",
            "revision": "c0c43c5790a808cb0e8ee833bc7487dc93e88cd5",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs={"NanoNQRetrieval": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2019-01-01", "2019-12-31"),
        domains=["Academic", "Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@article{47761,\n  author = {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh\nand Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee\nand Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le\nand Slav Petrov},\n  journal = {Transactions of the Association of Computational\nLinguistics},\n  title = {Natural Questions: a Benchmark for Question Answering Research},\n  year = {2019},\n}\n",
        adapted_from=["NanoNQRetrieval"],
        prompt={
            "query": "Given a question, retrieve Wikipedia passages that answer the question"
        },
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        gains = load_float_gains(self.metadata, hf_subset, hf_split)
        return ndcg_float_scores(gains, results, self.k_values)


class NanoQuoraRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoQuoraRCPRetrieval",
        description="NanoQuoraRetrieval is a smaller subset of the QuoraRetrieval dataset, which is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The human integer qrels are unchanged and still drive `ndcg_at_10`, which here reranks the fixed pool.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-nanobeir",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-nanobeir",
            "revision": "c0c43c5790a808cb0e8ee833bc7487dc93e88cd5",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs={"NanoQuoraRetrieval": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2017-01-01", "2017-12-31"),
        domains=["Social"],
        task_subtypes=["Duplicate Detection"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@misc{quora-question-pairs,\n  author = {DataCanary, hilfialkaff, Lili Jiang, Meg Risdal, Nikhil Dandekar, tomtung},\n  publisher = {Kaggle},\n  title = {Quora Question Pairs},\n  url = {https://kaggle.com/competitions/quora-question-pairs},\n  year = {2017},\n}\n",
        adapted_from=["NanoQuoraRetrieval"],
        prompt={
            "query": "Given a question, retrieve questions that are semantically equivalent to the given question"
        },
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        gains = load_float_gains(self.metadata, hf_subset, hf_split)
        return ndcg_float_scores(gains, results, self.k_values)


class NanoSCIDOCSRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoSCIDOCSRCPRetrieval",
        description="NanoFiQA2018 is a smaller subset of SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The human integer qrels are unchanged and still drive `ndcg_at_10`, which here reranks the fixed pool.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-nanobeir",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-nanobeir",
            "revision": "c0c43c5790a808cb0e8ee833bc7487dc93e88cd5",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs={"NanoSCIDOCSRetrieval": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2020-01-01", "2020-12-31"),
        domains=["Academic", "Written", "Non-fiction"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@inproceedings{specter2020cohan,\n  author = {Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},\n  booktitle = {ACL},\n  title = {SPECTER: Document-level Representation Learning using Citation-informed Transformers},\n  year = {2020},\n}\n",
        adapted_from=["NanoSCIDOCSRetrieval"],
        prompt={
            "query": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper"
        },
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        gains = load_float_gains(self.metadata, hf_subset, hf_split)
        return ndcg_float_scores(gains, results, self.k_values)


class NanoSciFactRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoSciFactRCPRetrieval",
        description="NanoSciFact is a smaller subset of SciFact, which verifies scientific claims using evidence from the research literature containing scientific paper abstracts. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The human integer qrels are unchanged and still drive `ndcg_at_10`, which here reranks the fixed pool.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-nanobeir",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-nanobeir",
            "revision": "c0c43c5790a808cb0e8ee833bc7487dc93e88cd5",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs={"NanoSciFactRetrieval": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["Academic", "Medical", "Written"],
        task_subtypes=["Claim verification"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@inproceedings{specter2020cohan,\n  author = {Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},\n  booktitle = {ACL},\n  title = {SPECTER: Document-level Representation Learning using Citation-informed Transformers},\n  year = {2020},\n}\n",
        adapted_from=["NanoSciFactRetrieval"],
        prompt={
            "query": "Given a scientific claim, retrieve documents that support or refute the claim"
        },
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        gains = load_float_gains(self.metadata, hf_subset, hf_split)
        return ndcg_float_scores(gains, results, self.k_values)


class NanoTouche2020RCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoTouche2020RCPRetrieval",
        description="NanoTouche2020 is a smaller subset of Touché Task 1: Argument Retrieval for Controversial Questions. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The human integer qrels are unchanged and still drive `ndcg_at_10`, which here reranks the fixed pool.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-nanobeir",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-nanobeir",
            "revision": "c0c43c5790a808cb0e8ee833bc7487dc93e88cd5",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs={"NanoTouche2020Retrieval": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2020-09-23", "2020-09-23"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@dataset{potthast_2022_6862281,\n  author = {Potthast, Martin and\nGienapp, Lukas and\nWachsmuth, Henning and\nHagen, Matthias and\nFröbe, Maik and\nBondarenko, Alexander and\nAjjour, Yamen and\nStein, Benno},\n  doi = {10.5281/zenodo.6862281},\n  month = jul,\n  publisher = {Zenodo},\n  title = {{Touché20-Argument-Retrieval-for-Controversial-\nQuestions}},\n  url = {https://doi.org/10.5281/zenodo.6862281},\n  year = {2022},\n}\n",
        adapted_from=["NanoTouche2020Retrieval"],
        prompt={
            "query": "Given a question, retrieve detailed and persuasive arguments that answer the question"
        },
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        gains = load_float_gains(self.metadata, hf_subset, hf_split)
        return ndcg_float_scores(gains, results, self.k_values)
