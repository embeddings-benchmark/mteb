from __future__ import annotations

from collections import defaultdict

import datasets

from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def load_bright_data(
    path: str,
    domain: str,
    eval_splits: list,
    cache_dir: str | None = None,
    revision: str | None = None,
):
    eval_split = eval_splits[0]
    corpus_name = "documents" if eval_split == "standard" else "long_documents"
    gold_ids_field = "gold_ids" if eval_split == "standard" else "gold_ids_long"

    corpus = dict.fromkeys(eval_splits)
    queries = dict.fromkeys(eval_splits)
    relevant_docs = dict.fromkeys(eval_splits)
    top_ranked = dict.fromkeys(eval_splits)

    domain_corpus = datasets.load_dataset(
        path,
        corpus_name,
        split=domain,
        cache_dir=cache_dir,
        revision=revision,
    )
    examples = datasets.load_dataset(
        path,
        "examples",
        split=domain,
        cache_dir=cache_dir,
        revision=revision,
    )
    corpus[eval_split] = {e["id"]: {"text": e["content"]} for e in domain_corpus}
    queries[eval_split] = {e["id"]: e["query"] for e in examples}
    relevant_docs[eval_split] = defaultdict(dict)
    top_ranked[eval_split] = defaultdict(list)

    # Get all document IDs
    all_doc_ids = [e["id"] for e in domain_corpus]
    have_excluded_ids = False

    for e in examples:
        qid = e["id"]
        gold_ids = e[gold_ids_field]
        for gid in gold_ids:
            relevant_docs[eval_split][qid].update({gid: 1})

        # Create top_ranked: all documents except excluded_ids
        excluded_ids = e.get("excluded_ids", [])
        if excluded_ids and excluded_ids != ["N/A"]:
            excluded_set = set(excluded_ids)
            top_ranked[eval_split][qid] = [
                doc_id for doc_id in all_doc_ids if doc_id not in excluded_set
            ]
            have_excluded_ids = True
        else:
            # No exclusions, use all documents
            top_ranked[eval_split][qid] = all_doc_ids

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)
    if have_excluded_ids:
        top_ranked = datasets.DatasetDict(top_ranked)
    else:
        top_ranked = None
    return corpus, queries, relevant_docs, top_ranked


_BIBTEX_CITATION = r"""
@misc{su2024brightrealisticchallengingbenchmark,
  archiveprefix = {arXiv},
  author = {Hongjin Su and Howard Yen and Mengzhou Xia and Weijia Shi and Niklas Muennighoff and Han-yu Wang and Haisu Liu and Quan Shi and Zachary S. Siegel and Michael Tang and Ruoxi Sun and Jinsung Yoon and Sercan O. Arik and Danqi Chen and Tao Yu},
  eprint = {2407.12883},
  primaryclass = {cs.CL},
  title = {BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval},
  url = {https://arxiv.org/abs/2407.12883},
  year = {2024},
}
"""


class BrightBiologyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightBiologyRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29c9651a576c8aba4832d249807b181ecae",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Biology StackExchange answers.",
        type="Retrieval",
        prompt={
            "query": "Represent this biology post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="biology",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightEarthScienceRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightEarthScienceRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29c9651a576c8aba4832d249807b181ecae",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Earth Science StackExchange answers.",
        type="Retrieval",
        prompt={
            "query": "Represent this earth_science post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="earth_science",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightEconomicsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightEconomicsRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29c9651a576c8aba4832d249807b181ecae",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Economics StackExchange answers.",
        type="Retrieval",
        prompt={
            "query": "Represent this economics post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="economics",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightPsychologyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightPsychologyRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29c9651a576c8aba4832d249807b181ecae",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Psychology StackExchange answers.",
        type="Retrieval",
        prompt={
            "query": "Represent this psychology post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="psychology",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightRoboticsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightRoboticsRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29c9651a576c8aba4832d249807b181ecae",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Robotics StackExchange answers.",
        type="Retrieval",
        prompt={
            "query": "Represent this robotics post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="robotics",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightStackoverflowRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightStackoverflowRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29c9651a576c8aba4832d249807b181ecae",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Stack Overflow answers.",
        type="Retrieval",
        prompt={
            "query": "Represent this stackoverflow post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="stackoverflow",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightSustainableLivingRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightSustainableLivingRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29c9651a576c8aba4832d249807b181ecae",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Sustainable Living StackExchange answers.",
        type="Retrieval",
        prompt={
            "query": "Represent this sustainable_living post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="sustainable_living",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightPonyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightPonyRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29c9651a576c8aba4832d249807b181ecae",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of Pony programming language syntax documentation.",
        type="Retrieval",
        prompt={
            "query": "Represent this Pony question for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="pony",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightLeetcodeRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightLeetcodeRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29c9651a576c8aba4832d249807b181ecae",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of similar algorithmic problems based on shared solution techniques.",
        type="Retrieval",
        prompt={
            "query": "Represent this Coding problem for searching relevant examples: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="leetcode",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightAopsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightAopsRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29c9651a576c8aba4832d249807b181ecae",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of similar Math Olympiad problems from Art of Problem Solving.",
        type="Retrieval",
        prompt={
            "query": "Represent this Math problem for searching relevant examples: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="aops",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightTheoremQATheoremsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightTheoremQATheoremsRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29c9651a576c8aba4832d249807b181ecae",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of theorem definitions and proofs from ProofWiki.",
        type="Retrieval",
        prompt={
            "query": "Represent this Math problem for searching relevant theorems: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="theoremqa_theorems",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightTheoremQAQuestionsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightTheoremQAQuestionsRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29c9651a576c8aba4832d249807b181ecae",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of theorem definitions from ProofWiki given questions rephrased as real-world scenarios.",
        type="Retrieval",
        prompt={
            "query": "Represent this Math problem for searching relevant examples: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="theoremqa_questions",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightBiologyLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightBiologyLongRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29c9651a576c8aba4832d249807b181ecae",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Biology StackExchange answers with long documents.",
        type="Retrieval",
        prompt={
            "query": "Represent this biology post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["long"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="biology",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightEarthScienceLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightEarthScienceLongRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29c9651a576c8aba4832d249807b181ecae",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Earth Science StackExchange answers with long documents.",
        type="Retrieval",
        prompt={
            "query": "Represent this earth_science post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["long"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="earth_science",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightEconomicsLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightEconomicsLongRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29c9651a576c8aba4832d249807b181ecae",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Economics StackExchange answers with long documents.",
        type="Retrieval",
        prompt={
            "query": "Represent this economics post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["long"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="economics",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightPsychologyLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightPsychologyLongRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29c9651a576c8aba4832d249807b181ecae",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Psychology StackExchange answers with long documents.",
        type="Retrieval",
        prompt={
            "query": "Represent this psychology post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["long"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="psychology",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightRoboticsLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightRoboticsLongRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29c9651a576c8aba4832d249807b181ecae",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Robotics StackExchange answers with long documents.",
        type="Retrieval",
        prompt={
            "query": "Represent this robotics post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["long"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="robotics",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightStackoverflowLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightStackoverflowLongRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29c9651a576c8aba4832d249807b181ecae",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Stack Overflow answers with long documents.",
        type="Retrieval",
        prompt={
            "query": "Represent this stackoverflow post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["long"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="stackoverflow",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightSustainableLivingLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightSustainableLivingLongRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29c9651a576c8aba4832d249807b181ecae",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Sustainable Living StackExchange answers with long documents.",
        type="Retrieval",
        prompt={
            "query": "Represent this sustainable_living post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["long"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="sustainable_living",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightPonyLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightPonyLongRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "3066d29c9651a576c8aba4832d249807b181ecae",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of Pony programming language syntax documentation with long documents.",
        type="Retrieval",
        prompt={
            "query": "Represent this Pony question for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["long"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="pony",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True
