from __future__ import annotations

from typing import Any

import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

CITATION = r"""
@inproceedings{harwath2015deep,
  title={Deep multimodal semantic embeddings for speech and images},
  author={Harwath, David and Glass, James},
  booktitle={2015 IEEE workshop on automatic speech recognition and understanding (ASRU)},
  pages={237--244},
  year={2015},
  organization={IEEE}
}
"""

_DESCRIPTION_A2IT = (
    "Audio to (Image, Text) retrieval for the Flickr Audio Image dataset. Queries "
    "are spoken renditions of Flickr8k captions, corpus items pair each image with "
    "its written text caption, so only models that use both the image and text "
    "signal exploit the full corpus."
)
_DESCRIPTION_IT2A = (
    "(Image, Text) to Audio retrieval for the Flickr Audio Image dataset, the "
    "reverse direction of FlickrAudioToImageTextRetrieval. Queries pair each image "
    "with its written text caption, corpus items are spoken renditions of Flickr8k "
    "captions."
)


class FlickrAudioToImageTextRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FlickrAudioToImageTextRetrieval",
        description=_DESCRIPTION_A2IT,
        reference="https://arxiv.org/pdf/1511.03690",
        dataset={
            "path": "deep9539/flickr-audio-image",
            "revision": "eccef427e2398ade609d40c1dcc9b5c30dce8641",
        },
        type="Retrieval",
        category="a2it",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2015-06-01", "2016-05-31"),
        domains=["Web", "AudioScene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio", "image", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        prompt={
            "query": "Find the (image, text caption) pair described by the spoken audio."
        },
        is_beta=True,
    )

    def load_data(self, **kwargs: Any):
        if self.data_loaded:
            return

        revision = self.metadata.dataset.get("revision")
        audio_ds = datasets.load_dataset(
            self.metadata.dataset["path"], "audio", split="train", revision=revision
        )
        images_ds = datasets.load_dataset(
            self.metadata.dataset["path"], "images", split="train", revision=revision
        )
        qrels_ds = datasets.load_dataset(
            self.metadata.dataset["path"], "qrels", split="train", revision=revision
        )

        # qrels_ds has 'image_id' and 'audio_id'
        # For A2IT: query is audio, corpus is (image, text)
        qrels = {}
        for row in qrels_ds:
            q_id = str(row["audio_id"])
            c_id = str(row["image_id"])
            if q_id not in qrels:
                qrels[q_id] = {}
            qrels[q_id][c_id] = 1

        self.dataset = {
            "default": {
                "train": {
                    "corpus": images_ds,
                    "queries": audio_ds,
                    "relevant_docs": qrels,
                    "top_ranked": None,
                }
            }
        }
        self.data_loaded = True


class FlickrImageTextToAudioRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FlickrImageTextToAudioRetrieval",
        description=_DESCRIPTION_IT2A,
        reference="https://arxiv.org/pdf/1511.03690",
        dataset={
            "path": "deep9539/flickr-audio-image",
            "revision": "eccef427e2398ade609d40c1dcc9b5c30dce8641",
        },
        type="Retrieval",
        category="it2a",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2015-06-01", "2016-05-31"),
        domains=["Web", "AudioScene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["image", "text", "audio"],
        sample_creation="found",
        bibtex_citation=CITATION,
        prompt={
            "query": "Find the spoken audio caption describing this (image, text) pair."
        },
        is_beta=True,
    )

    def load_data(self, **kwargs: Any):
        if self.data_loaded:
            return

        revision = self.metadata.dataset.get("revision")
        audio_ds = datasets.load_dataset(
            self.metadata.dataset["path"], "audio", split="train", revision=revision
        )
        images_ds = datasets.load_dataset(
            self.metadata.dataset["path"], "images", split="train", revision=revision
        )
        qrels_ds = datasets.load_dataset(
            self.metadata.dataset["path"], "qrels", split="train", revision=revision
        )

        # For IT2A: query is (image, text), corpus is audio
        qrels = {}
        for row in qrels_ds:
            q_id = str(row["image_id"])
            c_id = str(row["audio_id"])
            if q_id not in qrels:
                qrels[q_id] = {}
            qrels[q_id][c_id] = 1

        self.dataset = {
            "default": {
                "train": {
                    "corpus": audio_ds,
                    "queries": images_ds,
                    "relevant_docs": qrels,
                    "top_ranked": None,
                }
            }
        }
        self.data_loaded = True
