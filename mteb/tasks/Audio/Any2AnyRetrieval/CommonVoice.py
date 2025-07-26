from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata
import datasets
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

class CommonVoiceA2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="CommonVoiceA2TRetrieval",
        description="Speech recordings with corresponding text transcriptions from CommonVoice dataset.",
        reference="https://commonvoice.mozilla.org/",
        dataset={
            "path": "mozilla-foundation/common_voice_17_0",
            "revision": "b10d53980ef166bc24ce3358471c1970d7e6b5ec",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["af"],
        main_score="cv_recall_at_5",
        date=("2020-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{ardila2019common,
  title={Common voice: A massively-multilingual speech corpus},
  author={Ardila, Rosana and Branson, Megan and Davis, Kelly and Henretty, Michael and Kohler, Michael and Meyer, Josh and Morais, Reuben and Saunders, Lindsay and Tyers, Francis M and Weber, Gregor},
  booktitle={Proceedings of the 12th Language Resources and Evaluation Conference},
  pages={4218--4222},
  year={2020}
}
""",
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(
                                             self.metadata.dataset["path"],
                                             name = self.metadata.eval_langs[0],
                                             revision=self.metadata.dataset.get("revision"),
                                             )  # type: ignore
        self.dataset_transform()
        self.data_loaded = True
        


    def dataset_transform(self, id_col="path", text_col="sentence", audio_col="audio"):
        """
        Transform Common Voice dataset to MTEB t2a retrieval format.
        Returns (corpus, queries, relevant_docs) as DatasetDicts.
        """

        queries_ = {"id": [], "modality": [], "audio": []}
        corpus_ = {"id": [], "modality": [], "text": []}
        relevant_docs_ = {"query-id": [], "corpus-id": [], "score": []}
        relevant_docs_ = {}
        self.corpus = DatasetDict()
        self.queries = DatasetDict()
        self.relevant_docs = DatasetDict()
        
        qid = set()
        did = set()
        for split in self.metadata.eval_splits:
            for row in tqdm(self.dataset[split], total=len(self.dataset[split])):
                # Use the "path" field as a unique id for both query and doc

                query_id = str(row[id_col])
                doc_id = str(row[id_col])
                text = row[text_col]
                audio = row[audio_col]

                if query_id not in qid:
                    qid.add(query_id)
                    queries_["id"].append(query_id)
                    queries_['audio'].append(audio)
                    queries_["modality"].append("audio")
                    

                if doc_id not in did:
                    did.add(doc_id)
                    corpus_["id"].append(doc_id)
                    corpus_["text"].append(text)
                    corpus_["modality"].append("text")

                if query_id not in relevant_docs_:
                    relevant_docs_[query_id] = {}
                relevant_docs_[query_id][doc_id] = 1
                
            #  dict[str, dict[str, int]],
            self.corpus[split] = Dataset.from_dict(corpus_)
            self.queries[split] = Dataset.from_dict(queries_)
            self.relevant_docs[split] = relevant_docs_ # Dataset.from_dict(relevant_docs_)




class CommonVoiceT2ARetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="CommonVoiceT2ARetrieval",
        description="Speech recordings with corresponding text transcriptions from CommonVoice dataset.",
        reference="https://commonvoice.mozilla.org/",
        dataset={
            "path": "mozilla-foundation/common_voice_17_0",
            "revision": "b10d53980ef166bc24ce3358471c1970d7e6b5ec",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["af"],
        main_score="cv_recall_at_5",
        date=("2020-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Retrieval"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{ardila2019common,
  title={Common voice: A massively-multilingual speech corpus},
  author={Ardila, Rosana and Branson, Megan and Davis, Kelly and Henretty, Michael and Kohler, Michael and Meyer, Josh and Morais, Reuben and Saunders, Lindsay and Tyers, Francis M and Weber, Gregor},
  booktitle={Proceedings of the 12th Language Resources and Evaluation Conference},
  pages={4218--4222},
  year={2020}
}
""",
    )
    
    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(
                                             self.metadata.dataset["path"],
                                             name = self.metadata.eval_langs[0],
                                             revision=self.metadata.dataset.get("revision"),
                                             )  # type: ignore
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self, id_col="path", text_col="sentence", audio_col="audio"):
        """
        For T2A: query=text, corpus=audio.
        """
        queries_ = {"id": [], "modality": [], "text": []}
        corpus_ = {"id": [], "modality": [], "audio": []}
        relevant_docs_ = {}

        self.corpus = DatasetDict()
        self.queries = DatasetDict()
        self.relevant_docs = DatasetDict()

        qid = set()
        did = set()
        for split in self.metadata.eval_splits:
            for row in tqdm(self.dataset[split], total=len(self.dataset[split])):
                query_id = str(row[id_col])
                doc_id = str(row[id_col])
                text = row[text_col]
                audio = row[audio_col]

                if query_id not in qid:
                    qid.add(query_id)
                    queries_["id"].append(query_id)
                    queries_["text"].append(text)
                    queries_["modality"].append("text")

                if doc_id not in did:
                    did.add(doc_id)
                    corpus_["id"].append(doc_id)
                    corpus_["audio"].append(audio)
                    corpus_["modality"].append("audio")

                if query_id not in relevant_docs_:
                    relevant_docs_[query_id] = {}
                relevant_docs_[query_id][doc_id] = 1

            self.corpus[split] = Dataset.from_dict(corpus_)
            self.queries[split] = Dataset.from_dict(queries_)
            self.relevant_docs[split] = relevant_docs_
