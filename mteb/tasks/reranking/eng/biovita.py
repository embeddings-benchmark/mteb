from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_REFERENCE = "https://arxiv.org/abs/2603.23883"
_BIBTEX = r"""
@inproceedings{shinoda2026biovita,
  author = {Shinoda, Risa and Shiohara, Kaede and Inoue, Nakamasa and Saito, Kuniaki and Santo, Hiroaki and Okura, Fumio},
  booktitle = {CVPR},
  title = {BioVITA: Biological Dataset, Model, and Benchmark for Visual-Textual-Acoustic Alignment},
  year = {2026},
}
"""


class _BioVITAReranking(AbsTaskRetrieval):
    """Shared loader for the six BioVITA cross-modal reranking directions."""

    # The official evaluation reports Top-1/Top-5 accuracy; its script also
    # computes Top-10.
    k_values = (1, 5, 10)

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        """Compute BioVITA's official taxon-level Top-k accuracy.

        Each candidate taxon can contain multiple documents. Following the
        reference evaluation, we max-pool document similarities per taxon and
        rank the 100 candidate taxa. Standard MTEB metrics remain document-level.
        """
        split_data = self.dataset[hf_subset][hf_split]
        queries = split_data["queries"]
        corpus = split_data["corpus"]

        candidate_taxa = dict(
            zip(queries["id"], queries["candidate_taxa"], strict=True)
        )
        correct_taxon = dict(zip(queries["id"], queries["correct_taxon"], strict=True))
        doc_taxon = dict(zip(corpus["id"], corpus["taxon"], strict=True))

        hits = dict.fromkeys(self.k_values, 0)
        total = 0
        for query_id, doc_scores in results.items():
            taxa = candidate_taxa[query_id]
            rank_of_taxon = {taxon: rank for rank, taxon in enumerate(taxa)}
            best: dict[str, float] = {}
            for doc_id, score in doc_scores.items():
                taxon = doc_taxon.get(doc_id)
                if taxon is None or taxon not in rank_of_taxon:
                    continue
                if taxon not in best or score > best[taxon]:
                    best[taxon] = score
            # Ties keep the candidate order of the official CSV, matching the
            # index order `torch.topk` falls back on in the reference script.
            ranked = sorted(
                taxa,
                key=lambda taxon: (
                    -best.get(taxon, float("-inf")),
                    rank_of_taxon[taxon],
                ),
            )
            total += 1
            correct = correct_taxon[query_id]
            for k in self.k_values:
                if correct in ranked[:k]:
                    hits[k] += 1

        return {
            f"taxon_top_{k}_accuracy": hits[k] / max(1, total) for k in self.k_values
        }


class BioVITAA2TReranking(_BioVITAReranking):
    metadata = TaskMetadata(
        name="BioVITAA2TReranking",
        description="Given a wildlife audio recording, rerank candidate taxon names from the official candidate pool. Each query has 100 candidate taxa, which are ranked by similarity to their text representations. Performance is reported as taxon-level Top-1, Top-5, and Top-10 accuracy on the unseen species and unseen genus subsets.",
        reference=_REFERENCE,
        dataset={
            "path": "myang333/BioVITAA2TRetrieval",
            "revision": "e5cef5bd77ae85a62e54536a2f29de5a6cc0c32e",
        },
        type="Reranking",
        category="a2t",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs={"unseen_species": ["eng-Latn"], "unseen_genus": ["eng-Latn"]},
        main_score="taxon_top_1_accuracy",
        date=("1960-01-01", "2026-05-15"),
        domains=["Bioacoustics", "Nature", "Encyclopaedic"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        is_beta=True,
    )


class BioVITAT2AReranking(_BioVITAReranking):
    metadata = TaskMetadata(
        name="BioVITAT2AReranking",
        description="Given a taxon name, rerank candidate audio recordings by taxon relevance from the official candidate pool. Each query has 100 candidate taxa; a taxon is scored by the maximum similarity over its audio recordings. Performance is reported as taxon-level Top-1, Top-5, and Top-10 accuracy on the unseen species and unseen genus subsets.",
        reference=_REFERENCE,
        dataset={
            "path": "myang333/BioVITAT2ARetrieval",
            "revision": "0ea901959456c96cdc31659a82256a8a2386d498",
        },
        type="Reranking",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs={"unseen_species": ["eng-Latn"], "unseen_genus": ["eng-Latn"]},
        main_score="taxon_top_1_accuracy",
        date=("1960-01-01", "2026-05-15"),
        domains=["Bioacoustics", "Nature", "Encyclopaedic"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        is_beta=True,
    )


class BioVITAA2IReranking(_BioVITAReranking):
    metadata = TaskMetadata(
        name="BioVITAA2IReranking",
        description="Given a wildlife audio recording, rerank candidate images by taxon relevance from the official candidate pool. Each query has 100 candidate taxa; a taxon is scored by the maximum similarity over its images. Performance is reported as taxon-level Top-1, Top-5, and Top-10 accuracy on the unseen species and unseen genus subsets.",
        reference=_REFERENCE,
        dataset={
            "path": "myang333/BioVITAA2IRetrieval",
            "revision": "8a63b554297a556cc56175e73b8c7239f713e2c9",
        },
        type="Reranking",
        category="a2i",
        modalities=["audio", "image"],
        eval_splits=["test"],
        eval_langs={"unseen_species": ["eng-Latn"], "unseen_genus": ["eng-Latn"]},
        main_score="taxon_top_1_accuracy",
        date=("1960-01-01", "2026-05-15"),
        domains=["Bioacoustics", "Nature", "Encyclopaedic"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        is_beta=True,
    )


class BioVITAI2AReranking(_BioVITAReranking):
    metadata = TaskMetadata(
        name="BioVITAI2AReranking",
        description="Given a wildlife image, rerank candidate audio recordings by taxon relevance from the official candidate pool. Each query has 100 candidate taxa; a taxon is scored by the maximum similarity over its audio recordings. Performance is reported as taxon-level Top-1, Top-5, and Top-10 accuracy on the unseen species and unseen genus subsets.",
        reference=_REFERENCE,
        dataset={
            "path": "myang333/BioVITAI2ARetrieval",
            "revision": "2a7ef5884dc53e03854c824916eb1c45e946969c",
        },
        type="Reranking",
        category="i2a",
        modalities=["image", "audio"],
        eval_splits=["test"],
        eval_langs={"unseen_species": ["eng-Latn"], "unseen_genus": ["eng-Latn"]},
        main_score="taxon_top_1_accuracy",
        date=("1960-01-01", "2026-05-15"),
        domains=["Bioacoustics", "Nature", "Encyclopaedic"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        is_beta=True,
    )


class BioVITAI2TReranking(_BioVITAReranking):
    metadata = TaskMetadata(
        name="BioVITAI2TReranking",
        description="Given a wildlife image, rerank candidate taxon names from the official candidate pool. Each query has 100 candidate taxa, which are ranked by similarity to their text representations. Performance is reported as taxon-level Top-1, Top-5, and Top-10 accuracy on the unseen species and unseen genus subsets.",
        reference=_REFERENCE,
        dataset={
            "path": "myang333/BioVITAI2TRetrieval",
            "revision": "fa7f2ac1456130e7158958e4324f981bb5308668",
        },
        type="Reranking",
        category="i2t",
        modalities=["image", "text"],
        eval_splits=["test"],
        eval_langs={"unseen_species": ["eng-Latn"], "unseen_genus": ["eng-Latn"]},
        main_score="taxon_top_1_accuracy",
        date=("1960-01-01", "2026-05-15"),
        domains=["Nature", "Encyclopaedic"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        is_beta=True,
    )


class BioVITAT2IReranking(_BioVITAReranking):
    metadata = TaskMetadata(
        name="BioVITAT2IReranking",
        description="Given a taxon name, rerank candidate images by taxon relevance from the official candidate pool. Each query has 100 candidate taxa; a taxon is scored by the maximum similarity over its images. Performance is reported as taxon-level Top-1, Top-5, and Top-10 accuracy on the unseen species and unseen genus subsets.",
        reference=_REFERENCE,
        dataset={
            "path": "myang333/BioVITAT2IRetrieval",
            "revision": "fa59ba155cc59354c388e454c705a93bb9d984ab",
        },
        type="Reranking",
        category="t2i",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs={"unseen_species": ["eng-Latn"], "unseen_genus": ["eng-Latn"]},
        main_score="taxon_top_1_accuracy",
        date=("1960-01-01", "2026-05-15"),
        domains=["Nature", "Encyclopaedic"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        is_beta=True,
    )
