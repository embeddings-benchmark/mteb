from __future__ import annotations

import datasets

from mteb.abstasks.Audio.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class FreeMusicArchiveRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="FreeMusicArchiveRetrieval",
        description="Retrieval task to identify songs based on 5-second clips with adversarial modifications such as pitch shifting, EQ balancing, and background noise to test copyright infringement detection capabilities",
        reference="https://huggingface.co/datasets/ryanleeme17/free-music-archive-retrieval",
        dataset={
            "path": "ryanleeme17/free-music-archive-retrieval",
            "revision": "main",
        },
        type="Retrieval",
        category="s2s",
        modalities=["audio", "audio"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2024-12-31"),
        domains=["Music"],
        task_subtypes=["Duplicate Detection"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{piczak2015dataset,
  author = {Piczak, Karol J.},
  booktitle = {Proceedings of the 23rd {Annual ACM Conference} on {Multimedia}},
  date = {2015-10-13},
  doi = {10.1145/2733373.2806390},
  isbn = {978-1-4503-3459-4},
  location = {{Brisbane, Australia}},
  pages = {1015--1018},
  publisher = {{ACM Press}},
  title = {{ESC}: {Dataset} for {Environmental Sound Classification}},
  url = {http://dl.acm.org/citation.cfm?doid=2733373.2806390},
}
""",
        # prompt={"query": "Retrieve the corresponding song from the audio clip."},
    )

    audio_column_name: str = "audio"  # Full songs for corpus
    text_column_name: str = "title"  # Song titles
    id_column_name: str = "pid"  # Passage/Document IDs

    # Override modality detection for audio-to-audio retrieval
    default_query_modality: str = "audio"
    default_corpus_modality: str = "audio"

    def dataset_transform(self):
        """Transform the dataset to handle the FMA retrieval structure."""
        # Skip transformation for now to avoid processing the entire dataset
        pass

    def load_data(self, **kwargs):
        """Load data with proper query/corpus separation for FMA dataset."""
        if self.data_loaded:
            return

        # First load the dataset and apply transforms
        self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])
        self.dataset_transform()

        # Extract corpus, queries, and qrels from the dataset
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}

        for split_config in kwargs.get(
            "eval_splits", self.metadata_dict["eval_splits"]
        ):
            split = split_config.split("[")[0]

            if split not in self.dataset:
                continue

            dataset_split = self.dataset[split]

            # Apply slice notation if present
            if "[" in split_config and "]" in split_config:
                slice_part = split_config.split("[")[1].split("]")[0]
                if ":" in slice_part:
                    # Handle slice notation like "train[:50]"
                    slice_parts = slice_part.split(":")
                    start = int(slice_parts[0]) if slice_parts[0] else None
                    end = int(slice_parts[1]) if slice_parts[1] else None
                    dataset_split = dataset_split.select(
                        range(start or 0, end or len(dataset_split))
                    )
                else:
                    # Handle single index like "train[50]"
                    index = int(slice_part)
                    dataset_split = dataset_split.select([index])

            # Build corpus from full songs (audio-to-audio retrieval)
            corpus = {}
            for item in dataset_split:
                pid = str(item["pid"])  # Ensure string format
                # Use the full song audio as the corpus item
                corpus[pid] = item["audio"]

            # Build queries from 5-second clips
            # We'll use the original q_audio clips by default
            queries = {}
            for item in dataset_split:
                qid = str(item["qid"])  # Ensure string format
                queries[qid] = item["q_audio"]

            # Build qrels - each query maps to its corresponding song
            qrels = {}
            for item in dataset_split:
                qid = str(item["qid"])  # Ensure string format
                pid = str(item["pid"])  # Ensure string format
                qrels[qid] = {pid: 1}

            self.corpus[split] = corpus
            self.queries[split] = queries
            self.relevant_docs[split] = qrels

        self.data_loaded = True
