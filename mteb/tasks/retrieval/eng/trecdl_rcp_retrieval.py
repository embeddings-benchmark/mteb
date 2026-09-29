"""TREC-DL with rubric-calibrated preference (RCP) gains.

Each task adapts an existing mteb task (see `adapted_from`): the same queries and corpus, plus a
candidate pool (`top_ranked`) and continuous relevance gains in the qrels' `gain` column.
Scored by `ndcg_float_at_10` (see `AbsTaskRetrievalFloatGains`). Metadata is copied from the
original task except for name, description, reference, dataset, eval_langs, main_score,
annotations_creators, citation and adapted_from.
"""

from __future__ import annotations

from mteb.abstasks.retrieval_float_gains import AbsTaskRetrievalFloatGains
from mteb.abstasks.task_metadata import TaskMetadata

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


class TRECDL2019RCPRetrieval(AbsTaskRetrievalFloatGains):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="TRECDL2019RCPRetrieval",
        description="TREC Deep Learning Track 2019 passage ranking task. The task involves retrieving relevant passages from the MS MARCO collection given web search queries. Queries have multi-graded relevance judgments. Reranking over the judged TREC-DL candidate pool (198-585 documents per query), scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.6-27B-FP8), calibrated with a 2PL item-response model. The human NIST relevance grades (0-3) are unchanged and still drive `ndcg_at_10`, which here reranks the fixed pool. The corpus text is the text the judge saw: the upstream TREC-DL dataset's double-encoding (mojibake) is corrected.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-trecdl",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-trecdl",
            "revision": "9b78fbd9e86610a88f8d6c761974141b5d53ae8f",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={"trec_dl_2019": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2019-01-01", "2019-12-31"),
        domains=[
            "Encyclopaedic",
            "Academic",
            "Blog",
            "News",
            "Medical",
            "Government",
            "Reviews",
            "Non-fiction",
            "Social",
            "Web",
        ],
        task_subtypes=["Question answering"],
        license="msr-la-nc",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@misc{bajaj2018msmarcohumangenerated,\n  archiveprefix = {arXiv},\n  author = {Payal Bajaj and Daniel Campos and Nick Craswell and Li Deng and Jianfeng Gao and Xiaodong Liu and Rangan Majumder and Andrew McNamara and Bhaskar Mitra and Tri Nguyen and Mir Rosenberg and Xia Song and Alina Stoica and Saurabh Tiwary and Tong Wang},\n  eprint = {1611.09268},\n  primaryclass = {cs.CL},\n  title = {MS MARCO: A Human Generated MAchine Reading COmprehension Dataset},\n  url = {https://arxiv.org/abs/1611.09268},\n  year = {2018},\n}\n\n@inproceedings{craswell2020overview,\n  author = {Craswell, Nick and Mitra, Bhaskar and Yilmaz, Emine and Campos, Daniel and Voorhees, Ellen M},\n  booktitle = {Proceedings of the 28th Text REtrieval Conference (TREC 2019)},\n  organization = {NIST},\n  title = {Overview of the TREC 2019 deep learning track},\n  year = {2020},\n}\n",
        adapted_from=["TRECDL2019"],
        prompt={
            "query": "Given a web search query, retrieve relevant passages that answer the query"
        },
    )


class TRECDL2020RCPRetrieval(AbsTaskRetrievalFloatGains):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="TRECDL2020RCPRetrieval",
        description="TREC Deep Learning Track 2020 passage ranking task. The task involves retrieving relevant passages from the MS MARCO collection given web search queries. Queries have multi-graded relevance judgments. Reranking over the judged TREC-DL 2020 candidate pool (213-429 documents per query), scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.6-27B-FP8), calibrated with a 2PL item-response model. The human NIST relevance grades (0-3) are unchanged and still drive `ndcg_at_10`, which here reranks the fixed pool. The corpus text is the text the judge saw: the upstream TREC-DL dataset's double-encoding (mojibake) is corrected.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-trecdl",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-trecdl",
            "revision": "9b78fbd9e86610a88f8d6c761974141b5d53ae8f",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={"trec_dl_2020": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2020-01-01", "2020-12-31"),
        domains=[
            "Encyclopaedic",
            "Academic",
            "Blog",
            "News",
            "Medical",
            "Government",
            "Reviews",
            "Non-fiction",
            "Social",
            "Web",
        ],
        task_subtypes=["Question answering"],
        license="msr-la-nc",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@misc{bajaj2018msmarcohumangenerated,\n  archiveprefix = {arXiv},\n  author = {Payal Bajaj and Daniel Campos and Nick Craswell and Li Deng and Jianfeng Gao and Xiaodong Liu and Rangan Majumder and Andrew McNamara and Bhaskar Mitra and Tri Nguyen and Mir Rosenberg and Xia Song and Alina Stoica and Saurabh Tiwary and Tong Wang},\n  eprint = {1611.09268},\n  primaryclass = {cs.CL},\n  title = {MS MARCO: A Human Generated MAchine Reading COmprehension Dataset},\n  url = {https://arxiv.org/abs/1611.09268},\n  year = {2018},\n}\n\n@misc{craswell2021overviewtrec2020deep,\n  archiveprefix = {arXiv},\n  author = {Nick Craswell and Bhaskar Mitra and Emine Yilmaz and Daniel Campos},\n  eprint = {2102.07662},\n  primaryclass = {cs.IR},\n  title = {Overview of the TREC 2020 deep learning track},\n  url = {https://arxiv.org/abs/2102.07662},\n  year = {2021},\n}\n",
        adapted_from=["TRECDL2020"],
        prompt={
            "query": "Given a web search query, retrieve relevant passages that answer the query"
        },
    )
