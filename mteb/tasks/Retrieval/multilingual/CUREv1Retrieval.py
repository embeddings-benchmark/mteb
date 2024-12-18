from __future__ import annotations

from enum import Enum

from datasets import DatasetDict, load_dataset

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ....abstasks.MultilingualTask import MultilingualTask

_LANGUAGES = {
    "en": ["eng-Latn", "eng-Latn"],
    "es": ["spa-Latn", "eng-Latn"],
    "fr": ["fra-Latn", "eng-Latn"],
}


class CUREv1Splits(str, Enum):
    all = "All"
    dentistry_and_oral_health = "Dentistry and Oral Health"
    dermatology = "Dermatology"
    gastroenterology = "Gastroenterology"
    genetics = "Genetics"
    neuroscience_and_neurology = "Neuroscience and Neurology"
    orthopedic_surgery = "Orthopedic Surgery"
    otorhinolaryngology = "Otorhinolaryngology"
    plastic_surgery = "Plastic Surgery"
    psychiatry_and_psychology = "Psychiatry and Psychology"
    pulmonology = "Pulmonology"

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._member_names_)


class CUREv1Retrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        dataset={
            "path": "clinia/CUREv1",
            "revision": "3bcf51c91e04d04a8a3329dfbe988b964c5cbe83",
        },
        name="CUREv1",
        description="Collection of query-passage pairs curated by medical professionals, across 10 disciplines and 3 cross-lingual settings.",
        type="Retrieval",
        modalities=["text"],
        category="s2p",
        reference="https://huggingface.co/datasets/clinia/CUREv1",
        eval_splits=CUREv1Splits.names(),
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-10-31"),
        domains=["Medical", "Academic", "Written"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",
        prompt={
            "query": "Given a question by a medical professional, retrieve relevant passages that best answer the question",
        },
    )

    def _load_corpus(self, split: str, cache_dir: str | None = None):
        ds = load_dataset(
            path=self.metadata_dict["dataset"]["path"],
            revision=self.metadata_dict["dataset"]["revision"],
            name="corpus",
            split=split,
            cache_dir=cache_dir,
        )

        corpus = {
            doc["_id"]: {"title": doc["title"], "text": doc["text"]} for doc in ds
        }

        return corpus

    def _load_qrels(self, split: str, cache_dir: str | None = None):
        ds = load_dataset(
            path=self.metadata_dict["dataset"]["path"],
            revision=self.metadata_dict["dataset"]["revision"],
            name="qrels",
            split=split,
            cache_dir=cache_dir,
        )

        qrels = {}

        for qrel in ds:
            query_id = qrel["query-id"]
            doc_id = qrel["corpus-id"]
            score = int(qrel["score"])
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = score

        return qrels

    def _load_queries(self, split: str, language: str, cache_dir: str | None = None):
        ds = load_dataset(
            path=self.metadata_dict["dataset"]["path"],
            revision=self.metadata_dict["dataset"]["revision"],
            name=f"queries-{language}",
            split=split,
            cache_dir=cache_dir,
        )

        queries = {query["_id"]: query["text"] for query in ds}

        return queries

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        eval_splits = kwargs.get("eval_splits", self.metadata.eval_splits)
        languages = kwargs.get("eval_langs", self.metadata.eval_langs)
        cache_dir = kwargs.get("cache_dir", None)

        # Iterate over splits and languages
        corpus = {
            language: {split: None for split in eval_splits} for language in languages
        }
        queries = {
            language: {split: None for split in eval_splits} for language in languages
        }
        relevant_docs = {
            language: {split: None for split in eval_splits} for language in languages
        }
        for split in eval_splits:
            # Since this is a cross-lingual dataset, the corpus and the relevant documents do not depend on the language
            split_corpus = self._load_corpus(split=split, cache_dir=cache_dir)
            split_qrels = self._load_qrels(split=split, cache_dir=cache_dir)

            # Queries depend on the language
            for language in languages:
                corpus[language][split] = split_corpus
                relevant_docs[language][split] = split_qrels

                queries[language][split] = self._load_queries(
                    split=split, language=language, cache_dir=cache_dir
                )

        # Convert into DatasetDict
        self.corpus = DatasetDict(corpus)
        self.queries = DatasetDict(queries)
        self.relevant_docs = DatasetDict(relevant_docs)

        self.data_loaded = True
