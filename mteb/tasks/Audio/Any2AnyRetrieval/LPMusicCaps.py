from collections import defaultdict
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class MusicCapsA2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="MusicCapsA2TRetrieval",
        description="Audio-to-Text retrieval task based on MusicCaps/MagnaTagATune subset with music clips, tags, and captions.",
        reference="https://huggingface.co/datasets/mulab-mir/lp-music-caps-magnatagatune-3k",
        dataset={
            "path": "mulab-mir/lp-music-caps-magnatagatune-3k",
            "revision": "bf0da3a8ec9dfd48d3f6dd0de0d6d2742e1b4c17",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["audio", "text"],
        eval_splits=["train", "test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_5",
        date=None,
        domains=["Music"],
        task_subtypes=["Music Caption Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{agostinelli2023musiccaps,
  author = {Agostinelli, Andrea and Brown, Curtis and Kim, Bo and Kong, Qiao and Chechik, Gal and Choi, Keunwoo},
  title = {MusicCaps: Benchmarking Automatic Music Captioning with Large-Scale Human-Annotated Music Language Pairs},
  booktitle = {Proceedings of the 24th International Society for Music Information Retrieval Conference (ISMIR 2023)},
  year = {2023},
  address = {Milan, Italy},
  url = {https://arxiv.org/abs/2306.05284}
}
""",
    )

    def load_data(self, **kwargs):
        if getattr(self, "data_loaded", False):
            return
        self.corpus = defaultdict(DatasetDict)
        self.queries = defaultdict(DatasetDict)
        self.relevant_docs = defaultdict(DatasetDict)
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self, id_col="track_id", text_col="texts", audio_col="audio"):
        """A2T: Query=audio, Corpus=texts."""
        queries_ = {"id": [], "modality": [], "audio": []}
        corpus_ = {"id": [], "modality": [], "text": []}
        relevant_docs_ = {}

        ds = load_dataset(self.metadata.dataset["path"], split="train") + load_dataset(self.metadata.dataset["path"], split="test")

        for row in tqdm(ds, total=len(ds)):
            track_id = str(row[id_col])
            texts = row[text_col]  # this is a list of strings (captions/descriptions)
            audio = row[audio_col]  # dict with keys 'array' and 'sampling_rate'

            # Add one query per track (audio query)
            queries_["id"].append(track_id)
            queries_["modality"].append("audio")
            queries_["audio"].append(audio)

            # Add one corpus entry per track (all texts concatenated, or use first, or all as separate docs)
            corpus_["id"].append(track_id)
            corpus_["modality"].append("text")
            corpus_["text"].append(" ".join(texts))  # Join all texts, or use texts[0] for just the main caption

            relevant_docs_[track_id] = {track_id: 1}

        # For this dataset, each track_id is paired with itself as relevant (one-to-one)
        self.corpus["all"] = DatasetDict({"all": Dataset.from_dict(corpus_)})
        self.queries["all"] = DatasetDict({"all": Dataset.from_dict(queries_)})
        self.relevant_docs["all"] = relevant_docs_


class MusicCapsT2ARetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="MusicCapsT2ARetrieval",
        description="Text-to-Audio retrieval task based on MusicCaps/MagnaTagATune subset with music clips, tags, and captions.",
        reference="https://huggingface.co/datasets/mulab-mir/lp-music-caps-magnatagatune-3k",
        dataset={
            "path": "mulab-mir/lp-music-caps-magnatagatune-3k",
            "revision": "bf0da3a8ec9dfd48d3f6dd0de0d6d2742e1b4c17",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["train", "test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_5",
        date=None,
        domains=["Music"],
        task_subtypes=["Music Audio Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{agostinelli2023musiccaps,
  author = {Agostinelli, Andrea and Brown, Curtis and Kim, Bo and Kong, Qiao and Chechik, Gal and Choi, Keunwoo},
  title = {MusicCaps: Benchmarking Automatic Music Captioning with Large-Scale Human-Annotated Music Language Pairs},
  booktitle = {Proceedings of the 24th International Society for Music Information Retrieval Conference (ISMIR 2023)},
  year = {2023},
  address = {Milan, Italy},
  url = {https://arxiv.org/abs/2306.05284}
}
""",
    )

    def load_data(self, **kwargs):
        if getattr(self, "data_loaded", False):
            return
        self.corpus = defaultdict(DatasetDict)
        self.queries = defaultdict(DatasetDict)
        self.relevant_docs = defaultdict(DatasetDict)
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self, id_col="track_id", text_col="texts", audio_col="audio"):
        """T2A: Query=text, Corpus=audio."""
        queries_ = {"id": [], "modality": [], "text": []}
        corpus_ = {"id": [], "modality": [], "audio": []}
        relevant_docs_ = {}

        ds = load_dataset(self.metadata.dataset["path"], split="train") + load_dataset(self.metadata.dataset["path"], split="test")

        for row in tqdm(ds, total=len(ds)):
            track_id = str(row[id_col])
            texts = row[text_col]  # list of texts
            audio = row[audio_col]

            # Use the main caption as the query; you could loop for all captions for denser retrieval, but this is simplest
            queries_["id"].append(track_id)
            queries_["modality"].append("text")
            queries_["text"].append(" ".join(texts))

            corpus_["id"].append(track_id)
            corpus_["modality"].append("audio")
            corpus_["audio"].append(audio)

            relevant_docs_[track_id] = {track_id: 1}

        self.corpus["all"] = DatasetDict({"all": Dataset.from_dict(corpus_)})
        self.queries["all"] = DatasetDict({"all": Dataset.from_dict(queries_)})
        self.relevant_docs["all"] = relevant_docs_
