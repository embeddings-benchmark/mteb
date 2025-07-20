from __future__ import annotations

from collections import defaultdict

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class ClothoA2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="ClothoA2TRetrieval",
        description="An audio captioning datasetst containing audio clips and their corresponding captions.",
        reference="https://github.com/audio-captioning/clotho-dataset",
        dataset={
            "path": "CLAPv2/Clotho",
            "revision": "b491ad6569dba180ca60a0e2d17a1d6a0d5d9f4a",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{drossos2019clothoaudiocaptioningdataset,
  archiveprefix = {arXiv},
  author = {Konstantinos Drossos and Samuel Lipping and Tuomas Virtanen},
  eprint = {1910.09387},
  primaryclass = {cs.SD},
  title = {Clotho: An Audio Captioning Dataset},
  url = {https://arxiv.org/abs/1910.09387},
  year = {2019},
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        ds = load_dataset(**self.metadata.dataset, split="test", keep_in_memory=False)

        queries_ = {"id": [], "modality": [], "audio": []}
        corpus_ = {"id": [], "modality": [], "text": []}
        relevant_docs_ = {"query-id": [], "corpus-id": [], "score": []}

        qid = {}
        did = {}

        for row in tqdm(ds, total=len(ds), desc="Loading Clotho AT2 Retrieval Data"):
            audio = row["audio"]
            texts = row["text"]
            index = row["index"]

            ## a2t
            query_id = f"q-{index}"
            if query_id not in qid:
                qid[query_id] = query_id
                queries_["id"].append(query_id)
                queries_["audio"].append(audio)
                queries_["modality"].append("audio")

            for i, text in enumerate(texts.split(".")):
                doc_id = f"d-{index}-{i}"
                if doc_id not in did:
                    did[doc_id] = doc_id
                    corpus_["id"].append(doc_id)
                    corpus_["text"].append(text)
                    corpus_["modality"].append("text")

                relevant_docs_["query-id"].append(query_id)
                relevant_docs_["corpus-id"].append(doc_id)
                relevant_docs_["score"].append(1)

        corpus = Dataset.from_dict(corpus_)
        queries = Dataset.from_dict(queries_)
        relevant_docs = Dataset.from_dict(relevant_docs_)

        qrels_dict = defaultdict(dict)

        df = relevant_docs.to_pandas()
        query_ids = df["query-id"].to_numpy()
        corpus_ids = df["corpus-id"].to_numpy()
        scores = df["score"].to_numpy()

        for q, c, s in zip(query_ids, corpus_ids, scores):
            qrels_dict[q][c] = int(s)

        self.corpus = DatasetDict({"test": corpus})
        self.queries = DatasetDict({"test": queries})
        self.relevant_docs = {}
        self.relevant_docs["test"] = qrels_dict

        self.data_loaded = True


class ClothoT2ARetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="ClothoT2ARetrieval",
        description="An audio captioning datasetst containing audio clips from the Freesound platform and their corresponding captions.",
        reference="https://github.com/audio-captioning/clotho-dataset",
        dataset={
            "path": "CLAPv2/Clotho",
            "revision": "b491ad6569dba180ca60a0e2d17a1d6a0d5d9f4a",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{drossos2019clothoaudiocaptioningdataset,
  archiveprefix = {arXiv},
  author = {Konstantinos Drossos and Samuel Lipping and Tuomas Virtanen},
  eprint = {1910.09387},
  primaryclass = {cs.SD},
  title = {Clotho: An Audio Captioning Dataset},
  url = {https://arxiv.org/abs/1910.09387},
  year = {2019},
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        ds = load_dataset(**self.metadata.dataset, split="test", keep_in_memory=False)

        queries_ = {"id": [], "modality": [], "text": []}
        corpus_ = {"id": [], "modality": [], "audio": []}
        relevant_docs_ = {"query-id": [], "corpus-id": [], "score": []}

        qid = {}
        did = {}

        for row in tqdm(ds, total=len(ds), desc="Loading Clotho T2A Retrieval Data"):
            audio = row["audio"]
            texts = row["text"]
            index = row["index"]

            ## t2a
            doc_id = f"d-{index}"
            did[doc_id] = doc_id
            corpus_["id"].append(doc_id)
            corpus_["audio"].append(audio)
            corpus_["modality"].append("audio")

            for i, text in enumerate(texts.split(".")):
                query_id = f"q-{index}-{i}"
                if query_id not in qid:
                    qid[query_id] = query_id
                    queries_["id"].append(query_id)
                    queries_["text"].append(text)
                    queries_["modality"].append("text")

                relevant_docs_["query-id"].append(query_id)
                relevant_docs_["corpus-id"].append(doc_id)
                relevant_docs_["score"].append(1)

        corpus = Dataset.from_dict(corpus_)
        queries = Dataset.from_dict(queries_)
        relevant_docs = Dataset.from_dict(relevant_docs_)

        qrels_dict = defaultdict(dict)

        df = relevant_docs.to_pandas()
        query_ids = df["query-id"].to_numpy()
        corpus_ids = df["corpus-id"].to_numpy()
        scores = df["score"].to_numpy()

        for q, c, s in zip(query_ids, corpus_ids, scores):
            qrels_dict[q][c] = int(s)

        self.corpus = DatasetDict({"test": corpus})
        self.queries = DatasetDict({"test": queries})
        self.relevant_docs = {}
        self.relevant_docs["test"] = qrels_dict

        self.data_loaded = True
