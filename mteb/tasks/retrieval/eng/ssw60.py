from __future__ import annotations

from typing import Any

from datasets import concatenate_datasets, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_REFERENCE = "https://github.com/visipedia/ssw60"
_BIBTEX = r"""
@inproceedings{van2022exploring,
  author = {Van Horn, Grant and Qian, Rui and Wilber, Kimberly and Adam, Hartwig and Mac Aodha, Oisin and Belongie, Serge},
  title = {Exploring Fine-Grained Audiovisual Categorization with the SSW60 Dataset},
  booktitle={European Conference on Computer Vision},
  pages={271--289},
  year={2022},
  organization={Springer}
}
"""

_DESCRIPTION = (
    "Sapsucker Woods 60 (SSW60) contains standalone audio of 60 bird species "
    "sourced from Macaulay Library, and static bird images sourced from "
    "iNaturalist and NABirds. This task evaluates the model's ability to "
    "perform cross-modal class-level retrieval between bird audio recordings "
    "and static bird images."
)


class SSW60A2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SSW60A2IRetrieval",
        description=_DESCRIPTION
        + " Queries are bird audio recordings (Macaulay Library) and the corpus "
        "contains the combined static bird images from both iNaturalist and NABirds.",
        reference=_REFERENCE,
        dataset={
            "path": "nik1995/ssw60_audio_image",
            "revision": "453efd1eb3d933565e23e657ab70502360acd632",
        },
        type="Any2AnyRetrieval",
        category="a2i",
        modalities=["audio", "image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-12-31"),
        domains=["AudioScene", "Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Retrieve relevant bird images that match the species in this audio recording."
        },
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            # Load standalone audio queries
            queries_ds = load_dataset(
                self.metadata.dataset["path"], name="audio", split=split
            )

            # Load static image corpus from both iNaturalist and NABirds
            inat_ds = load_dataset(
                self.metadata.dataset["path"], name="images_inat", split=split
            )
            nabirds_ds = load_dataset(
                self.metadata.dataset["path"], name="images_nabirds", split=split
            )
            corpus_ds = concatenate_datasets([inat_ds, nabirds_ds])

            queries = queries_ds.rename_column("asset_id", "id")
            corpus = corpus_ds.rename_column("asset_id", "id")

            # Map species label -> list of relevant corpus document IDs (images)
            label_to_image_ids = {}
            for row in corpus:
                label_to_image_ids.setdefault(row["label"], []).append(row["id"])

            # Construct qrels: for each query (audio), all matching species images are relevant
            qrels = {}
            for row in queries:
                query_id = row["id"]
                query_label = row["label"]
                relevant_images = label_to_image_ids.get(query_label, [])
                qrels[query_id] = dict.fromkeys(relevant_images, 1)

            queries = queries.select_columns(["id", "audio"])
            corpus = corpus.select_columns(["id", "image"])

            self.dataset["default"][split] = RetrievalSplitData(
                queries=queries, corpus=corpus, relevant_docs=qrels, top_ranked=None
            )
        self.data_loaded = True


class SSW60I2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SSW60I2ARetrieval",
        description=_DESCRIPTION
        + " Queries are static bird images (combined from iNaturalist and NABirds) "
        "and the corpus contains the standalone bird audio recordings from Macaulay Library.",
        reference=_REFERENCE,
        dataset={
            "path": "nik1995/ssw60_audio_image",
            "revision": "453efd1eb3d933565e23e657ab70502360acd632",
        },
        type="Any2AnyRetrieval",
        category="i2a",
        modalities=["image", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-12-31"),
        domains=["Scene", "AudioScene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Retrieve relevant species audio recordings that match the bird depicted in this image."
        },
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            # Load static bird images as queries (from iNaturalist and NABirds)
            inat_ds = load_dataset(
                self.metadata.dataset["path"], name="images_inat", split=split
            )
            nabirds_ds = load_dataset(
                self.metadata.dataset["path"], name="images_nabirds", split=split
            )
            queries_ds = concatenate_datasets([inat_ds, nabirds_ds])

            # Load standalone audio recordings as corpus
            corpus_ds = load_dataset(
                self.metadata.dataset["path"], name="audio", split=split
            )

            queries = queries_ds.rename_column("asset_id", "id")
            corpus = corpus_ds.rename_column("asset_id", "id")

            # Map species label -> list of relevant corpus document IDs (audios)
            label_to_audio_ids = {}
            for row in corpus:
                label_to_audio_ids.setdefault(row["label"], []).append(row["id"])

            # Construct qrels: for each query (image), all matching species audios are relevant
            qrels = {}
            for row in queries:
                query_id = row["id"]
                query_label = row["label"]
                relevant_audios = label_to_audio_ids.get(query_label, [])
                qrels[query_id] = dict.fromkeys(relevant_audios, 1)

            queries = queries.select_columns(["id", "image"])
            corpus = corpus.select_columns(["id", "audio"])

            self.dataset["default"][split] = RetrievalSplitData(
                queries=queries, corpus=corpus, relevant_docs=qrels, top_ranked=None
            )
        self.data_loaded = True
