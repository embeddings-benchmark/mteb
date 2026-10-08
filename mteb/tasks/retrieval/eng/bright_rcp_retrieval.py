"""BRIGHT with rubric-calibrated preference (RCP) gains.

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


class BrightAopsRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightAopsRCPRetrieval",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of similar Math Olympiad problems from Art of Problem Solving. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. BRIGHT's `excluded_ids` are removed from the candidates. The integer qrels are upstream `xlangai/BRIGHT@a75a0eb4` gold_ids (as in `BrightRetrieval`). These differ from the later-corrected qrels of the BRIGHT(v1.1) tasks on some queries, so `ndcg_at_10` is not comparable to BRIGHT(v1.1).",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-bright",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-bright",
            "revision": "3f42b638ae5c7b55faa93379c29825095de28085",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["standard"],
        eval_langs={"aops": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@article{su2024bright,\n  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},\n  journal = {arXiv preprint arXiv:2407.12883},\n  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},\n  year = {2024},\n}\n",
        adapted_from=["BrightRetrieval", "BrightAopsRetrieval"],
        prompt={
            "query": "Represent this Math problem for searching relevant examples: "
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


class BrightBiologyRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightBiologyRCPRetrieval",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Biology StackExchange answers. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. BRIGHT's `excluded_ids` are removed from the candidates. The integer qrels are upstream `xlangai/BRIGHT@a75a0eb4` gold_ids (as in `BrightRetrieval`). These differ from the later-corrected qrels of the BRIGHT(v1.1) tasks on some queries, so `ndcg_at_10` is not comparable to BRIGHT(v1.1).",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-bright",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-bright",
            "revision": "3f42b638ae5c7b55faa93379c29825095de28085",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["standard"],
        eval_langs={"biology": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@article{su2024bright,\n  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},\n  journal = {arXiv preprint arXiv:2407.12883},\n  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},\n  year = {2024},\n}\n",
        adapted_from=["BrightRetrieval", "BrightBiologyRetrieval"],
        prompt={
            "query": "Represent this biology post for searching relevant passages: "
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


class BrightEarthScienceRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightEarthScienceRCPRetrieval",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Earth Science StackExchange answers. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. BRIGHT's `excluded_ids` are removed from the candidates. The integer qrels are upstream `xlangai/BRIGHT@a75a0eb4` gold_ids (as in `BrightRetrieval`). These differ from the later-corrected qrels of the BRIGHT(v1.1) tasks on some queries, so `ndcg_at_10` is not comparable to BRIGHT(v1.1).",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-bright",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-bright",
            "revision": "3f42b638ae5c7b55faa93379c29825095de28085",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["standard"],
        eval_langs={"earth_science": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@article{su2024bright,\n  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},\n  journal = {arXiv preprint arXiv:2407.12883},\n  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},\n  year = {2024},\n}\n",
        adapted_from=["BrightRetrieval", "BrightEarthScienceRetrieval"],
        prompt={
            "query": "Represent this earth_science post for searching relevant passages: "
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


class BrightEconomicsRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightEconomicsRCPRetrieval",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Economics StackExchange answers. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. BRIGHT's `excluded_ids` are removed from the candidates. The integer qrels are upstream `xlangai/BRIGHT@a75a0eb4` gold_ids (as in `BrightRetrieval`). These differ from the later-corrected qrels of the BRIGHT(v1.1) tasks on some queries, so `ndcg_at_10` is not comparable to BRIGHT(v1.1).",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-bright",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-bright",
            "revision": "3f42b638ae5c7b55faa93379c29825095de28085",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["standard"],
        eval_langs={"economics": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@article{su2024bright,\n  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},\n  journal = {arXiv preprint arXiv:2407.12883},\n  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},\n  year = {2024},\n}\n",
        adapted_from=["BrightRetrieval", "BrightEconomicsRetrieval"],
        prompt={
            "query": "Represent this economics post for searching relevant passages: "
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


class BrightLeetcodeRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightLeetcodeRCPRetrieval",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of similar algorithmic problems based on shared solution techniques. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. BRIGHT's `excluded_ids` are removed from the candidates. The integer qrels are upstream `xlangai/BRIGHT@a75a0eb4` gold_ids (as in `BrightRetrieval`). These differ from the later-corrected qrels of the BRIGHT(v1.1) tasks on some queries, so `ndcg_at_10` is not comparable to BRIGHT(v1.1).",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-bright",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-bright",
            "revision": "3f42b638ae5c7b55faa93379c29825095de28085",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["standard"],
        eval_langs={"leetcode": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@article{su2024bright,\n  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},\n  journal = {arXiv preprint arXiv:2407.12883},\n  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},\n  year = {2024},\n}\n",
        adapted_from=["BrightRetrieval", "BrightLeetcodeRetrieval"],
        prompt={
            "query": "Represent this Coding problem for searching relevant examples: "
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


class BrightPonyRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightPonyRCPRetrieval",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of Pony programming language syntax documentation. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. BRIGHT's `excluded_ids` are removed from the candidates. The integer qrels are upstream `xlangai/BRIGHT@a75a0eb4` gold_ids (as in `BrightRetrieval`). These differ from the later-corrected qrels of the BRIGHT(v1.1) tasks on some queries, so `ndcg_at_10` is not comparable to BRIGHT(v1.1).",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-bright",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-bright",
            "revision": "3f42b638ae5c7b55faa93379c29825095de28085",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["standard"],
        eval_langs={"pony": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@article{su2024bright,\n  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},\n  journal = {arXiv preprint arXiv:2407.12883},\n  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},\n  year = {2024},\n}\n",
        adapted_from=["BrightRetrieval", "BrightPonyRetrieval"],
        prompt={
            "query": "Represent this Pony question for searching relevant passages: "
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


class BrightPsychologyRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightPsychologyRCPRetrieval",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Psychology StackExchange answers. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. BRIGHT's `excluded_ids` are removed from the candidates. The integer qrels are upstream `xlangai/BRIGHT@a75a0eb4` gold_ids (as in `BrightRetrieval`). These differ from the later-corrected qrels of the BRIGHT(v1.1) tasks on some queries, so `ndcg_at_10` is not comparable to BRIGHT(v1.1).",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-bright",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-bright",
            "revision": "3f42b638ae5c7b55faa93379c29825095de28085",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["standard"],
        eval_langs={"psychology": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@article{su2024bright,\n  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},\n  journal = {arXiv preprint arXiv:2407.12883},\n  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},\n  year = {2024},\n}\n",
        adapted_from=["BrightRetrieval", "BrightPsychologyRetrieval"],
        prompt={
            "query": "Represent this psychology post for searching relevant passages: "
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


class BrightRoboticsRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightRoboticsRCPRetrieval",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Robotics StackExchange answers. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. BRIGHT's `excluded_ids` are removed from the candidates. The integer qrels are upstream `xlangai/BRIGHT@a75a0eb4` gold_ids (as in `BrightRetrieval`). These differ from the later-corrected qrels of the BRIGHT(v1.1) tasks on some queries, so `ndcg_at_10` is not comparable to BRIGHT(v1.1).",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-bright",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-bright",
            "revision": "3f42b638ae5c7b55faa93379c29825095de28085",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["standard"],
        eval_langs={"robotics": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@article{su2024bright,\n  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},\n  journal = {arXiv preprint arXiv:2407.12883},\n  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},\n  year = {2024},\n}\n",
        adapted_from=["BrightRetrieval", "BrightRoboticsRetrieval"],
        prompt={
            "query": "Represent this robotics post for searching relevant passages: "
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


class BrightStackoverflowRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightStackoverflowRCPRetrieval",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Stack Overflow answers. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. BRIGHT's `excluded_ids` are removed from the candidates. The integer qrels are upstream `xlangai/BRIGHT@a75a0eb4` gold_ids (as in `BrightRetrieval`). These differ from the later-corrected qrels of the BRIGHT(v1.1) tasks on some queries, so `ndcg_at_10` is not comparable to BRIGHT(v1.1).",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-bright",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-bright",
            "revision": "3f42b638ae5c7b55faa93379c29825095de28085",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["standard"],
        eval_langs={"stackoverflow": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@article{su2024bright,\n  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},\n  journal = {arXiv preprint arXiv:2407.12883},\n  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},\n  year = {2024},\n}\n",
        adapted_from=["BrightRetrieval", "BrightStackoverflowRetrieval"],
        prompt={
            "query": "Represent this stackoverflow post for searching relevant passages: "
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


class BrightSustainableLivingRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightSustainableLivingRCPRetrieval",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Sustainable Living StackExchange answers. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. BRIGHT's `excluded_ids` are removed from the candidates. The integer qrels are upstream `xlangai/BRIGHT@a75a0eb4` gold_ids (as in `BrightRetrieval`). These differ from the later-corrected qrels of the BRIGHT(v1.1) tasks on some queries, so `ndcg_at_10` is not comparable to BRIGHT(v1.1).",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-bright",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-bright",
            "revision": "3f42b638ae5c7b55faa93379c29825095de28085",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["standard"],
        eval_langs={"sustainable_living": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@article{su2024bright,\n  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},\n  journal = {arXiv preprint arXiv:2407.12883},\n  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},\n  year = {2024},\n}\n",
        adapted_from=["BrightRetrieval", "BrightSustainableLivingRetrieval"],
        prompt={
            "query": "Represent this sustainable_living post for searching relevant passages: "
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


class BrightTheoremQAQuestionsRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightTheoremQAQuestionsRCPRetrieval",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of theorem definitions from ProofWiki given questions rephrased as real-world scenarios. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. BRIGHT's `excluded_ids` are removed from the candidates. The integer qrels are upstream `xlangai/BRIGHT@a75a0eb4` gold_ids (as in `BrightRetrieval`). These differ from the later-corrected qrels of the BRIGHT(v1.1) tasks on some queries, so `ndcg_at_10` is not comparable to BRIGHT(v1.1).",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-bright",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-bright",
            "revision": "3f42b638ae5c7b55faa93379c29825095de28085",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["standard"],
        eval_langs={"theoremqa_questions": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@article{su2024bright,\n  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},\n  journal = {arXiv preprint arXiv:2407.12883},\n  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},\n  year = {2024},\n}\n",
        adapted_from=["BrightRetrieval", "BrightTheoremQAQuestionsRetrieval"],
        prompt={
            "query": "Represent this Math problem for searching relevant examples: "
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


class BrightTheoremQATheoremsRCPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightTheoremQATheoremsRCPRetrieval",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of theorem definitions and proofs from ProofWiki. Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. BRIGHT's `excluded_ids` are removed from the candidates. The integer qrels are upstream `xlangai/BRIGHT@a75a0eb4` gold_ids (as in `BrightRetrieval`). These differ from the later-corrected qrels of the BRIGHT(v1.1) tasks on some queries, so `ndcg_at_10` is not comparable to BRIGHT(v1.1).",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-bright",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-bright",
            "revision": "3f42b638ae5c7b55faa93379c29825095de28085",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["standard"],
        eval_langs={"theoremqa_theorems": ["eng-Latn"]},
        main_score="ndcg_float_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION
        + "\n@article{su2024bright,\n  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},\n  journal = {arXiv preprint arXiv:2407.12883},\n  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},\n  year = {2024},\n}\n",
        adapted_from=["BrightRetrieval", "BrightTheoremQATheoremsRetrieval"],
        prompt={
            "query": "Represent this Math problem for searching relevant theorems: "
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
