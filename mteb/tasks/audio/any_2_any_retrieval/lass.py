from collections import defaultdict

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LASSA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LASSA2TRetrieval",
        description=(
            "Language-Queried Audio Source Separation (LASS) dataset for audio-to-text retrieval. "
            "Retrieve text descriptions/captions for audio clips using natural language queries."
            "The original dataset is based on the AudioCaps dataset."
            "The source audio has been synthesized by mixing two audio with their labelled snr ratio as indicated in the dataset."
        ),
        reference="https://dcase.community/challenge2024/task-language-queried-audio-source-separation",
        dataset={
            "path": "mteb/lass-synth",
            "revision": "102ae116d8537dab63d01655a69cf2ffe7d813b5",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2025-11-15", "2025-11-15"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{liu2022separate,
  title = {Separate What You Describe: Language-Queried Audio Source Separation},
  author = {Liu, Xubo and Liu, Haohe and Kong, Qiuqiang and Mei, Xinhao and Zhao, Jinzheng and Huang, Qiushi and Plumbley, Mark D and Wang, Wenwu},
  booktitle = {INTERSPEEH},
  year = {2022}
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

        for row in tqdm(ds, total=len(ds), desc="Loading LASS A2T Retrieval Data"):
            audio = row["audio"]
            text = row["text"]
            source_id = row["source_id"]

            ## a2t
            query_id = f"q-{source_id}"
            if query_id not in qid:
                qid[query_id] = query_id
                queries_["id"].append(query_id)
                queries_["audio"].append(audio)
                queries_["modality"].append("audio")

            doc_id = f"d-{source_id}"
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


class LASST2ARetrieval(AbsTaskRetrieval):
    """Text-to-audio retrieval on LASS dataset."""

    metadata = TaskMetadata(
        name="LASST2ARetrieval",
        description=(
            "Language-Queried Audio Source Separation (LASS) dataset for text-to-audio retrieval. "
            "Retrieve audio clips corresponding to natural language text descriptions/captions."
            "The original dataset is based on the AudioCaps dataset."
            "The source audio has been synthesized by mixing two audio with their labelled snr ratio as indicated in the dataset."
        ),
        reference="https://dcase.community/challenge2024/task-language-queried-audio-source-separation",
        dataset={
            "path": "mteb/lass-synth",
            "revision": "102ae116d8537dab63d01655a69cf2ffe7d813b5",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2025-11-15", "2025-11-15"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{liu2022separate,
  title = {Separate What You Describe: Language-Queried Audio Source Separation},
  author = {Liu, Xubo and Liu, Haohe and Kong, Qiuqiang and Mei, Xinhao and Zhao, Jinzheng and Huang, Qiushi and Plumbley, Mark D and Wang, Wenwu},
  booktitle = {INTERSPEEH},
  year = {2022}
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

        for row in tqdm(ds, total=len(ds), desc="Loading LASS T2A Retrieval Data"):
            audio = row["audio"]
            text = row["text"]
            source_id = row["source_id"]

            ## t2a
            doc_id = f"d-{source_id}"
            if doc_id not in did:
                did[doc_id] = doc_id
                corpus_["id"].append(doc_id)
                corpus_["audio"].append(audio)
                corpus_["modality"].append("audio")

            query_id = f"q-{source_id}"
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
