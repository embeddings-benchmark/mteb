"""
MTEB Task Implementation for ACM Composed Audio Retrieval (deep9539/ACM-processed).
"""

from __future__ import annotations

from typing import Any

import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class ACMComposedAudioRetrieval(AbsTaskRetrieval):
    """
    Composed Audio Retrieval task implementation for HuggingFace dataset 'deep9539/ACM-processed'.
    """

    metadata = TaskMetadata(
        name="ACMComposedAudioRetrieval",
        description="Composed audio retrieval task evaluating multimodal audio-text embeddings using source audio and text modification instructions.",
        reference="https://arxiv.org/abs/2603.02098",
        dataset={
            "path": "deep9539/ACM-processed",
            "revision": "bc16d75267123a7ac893eccec65693e1e30bec72",
        },
        type="Any2AnyRetrieval",
        category="at2a",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_5",
        date=("2020-01-01", "2026-03-01"),
        domains=["AudioScene", "Speech", "Music"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        is_beta=True,
        sample_creation="found",
        bibtex_citation=r"""
@article{huynh2026omniret,
  author = {Huynh, Chuong and Luong, Manh and Shrivastava},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  title = {Efficient and High-Fidelity Omni Modality Retrieval},
  year = {2026},
}
""",
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        split = kwargs.get("split", "test")
        repo_id = self.metadata.dataset["path"]
        revision = self.metadata.dataset.get("revision", "main")

        # Load candidates and queries datasets from deep9539/ACM-processed
        candidates_ds = datasets.load_dataset(
            repo_id,
            name="composed_audio_retrieval_candidates",
            split=split,
            revision=revision,
        )
        queries_ds = datasets.load_dataset(
            repo_id,
            name="composed_audio_retrieval_queries",
            split=split,
            revision=revision,
        )

        self.dataset = {"default": {}}

        # Process Corpus: select audio column and add string IDs
        corpus = candidates_ds.select_columns(["audio"])
        corpus = corpus.add_column("id", [str(x) for x in candidates_ds["audio_id"]])

        # Process Queries: select text instruction and source audio column
        queries = queries_ds.select_columns(["modified_text", "src_audio"])
        queries = queries.rename_column("src_audio", "audio")
        queries = queries.rename_column("modified_text", "text")
        queries = queries.add_column("id", [str(x) for x in queries_ds["sample_id"]])

        valid_query_ids = set(queries["id"])
        relevant_docs: dict[str, dict[str, int]] = {}
        for item in queries_ds:
            query_id = str(item["sample_id"])
            if query_id in valid_query_ids:
                tgt_id = str(item["tgt_audio_id"])
                relevant_docs[query_id] = {tgt_id: 1}

        self.dataset["default"][split] = RetrievalSplitData(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            top_ranked=None,
        )
        self.data_loaded = True
