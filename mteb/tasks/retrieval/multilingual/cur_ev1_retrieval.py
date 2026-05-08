from enum import Enum

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

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


class CUREv1Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        dataset={
            "path": "clinia/CUREv1",
            "revision": "3bcf51c91e04d04a8a3329dfbe988b964c5cbe83",
        },
        name="CUREv1",
        description="Collection of query-passage pairs curated by medical professionals, across 10 disciplines and 3 cross-lingual settings.",
        type="Retrieval",
        modalities=["text"],
        category="t2t",
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

    def _load_corpus(self, split: str):
        ds = load_dataset(
            path=self.metadata.dataset["path"],
            revision=self.metadata.dataset["revision"],
            name="corpus",
            split=split,
        )

        corpus = {
            doc["_id"]: {"title": doc["title"], "text": doc["text"]} for doc in ds
        }

        return corpus

    def _load_qrels(
        self,
        split: str,
    ):
        ds = load_dataset(
            path=self.metadata.dataset["path"],
            revision=self.metadata.dataset["revision"],
            name="qrels",
            split=split,
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

    def _load_queries(
        self,
        split: str,
        language: str,
    ):
        ds = load_dataset(
            path=self.metadata.dataset["path"],
            revision=self.metadata.dataset["revision"],
            name=f"queries-{language}",
            split=split,
        )

        queries = {query["_id"]: query["text"] for query in ds}

        return queries

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        eval_splits = self.metadata.eval_splits
        languages = self.metadata.eval_langs

        self.dataset = {}

        for language in languages:
            self.dataset[language] = {}

        for split in eval_splits:
            # Since this is a cross-lingual dataset, the corpus and the relevant documents do not depend on the language
            split_corpus_dict = self._load_corpus(split=split)
            split_qrels = self._load_qrels(split=split)

            split_corpus_ds = Dataset.from_list(
                [
                    {"id": k, "text": v.get("text", ""), "title": v.get("title", "")}
                    for k, v in split_corpus_dict.items()
                ]
            )

            # Queries depend on the language
            for language in languages:
                queries_dict = self._load_queries(split=split, language=language)
                queries_ds = Dataset.from_list(
                    [{"id": k, "text": v} for k, v in queries_dict.items()]
                )

                self.dataset[language][split] = {
                    "corpus": split_corpus_ds,
                    "queries": queries_ds,
                    "relevant_docs": split_qrels,
                    "top_ranked": None,
                }

        self.data_loaded = True
