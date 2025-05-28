from __future__ import annotations

import datasets

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "default"

_LANGUAGES = {
    "ar": ["ara-Arab"],
    "bn": ["ben-Beng"],
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fa": ["fas-Arab"],
    "fi": ["fin-Latn"],
    "fr": ["fra-Latn"],
    "hi": ["hin-Deva"],
    "id": ["ind-Latn"],
    "ja": ["jpn-Jpan"],
    "ko": ["kor-Kore"],
    "ru": ["rus-Cyrl"],
    "sw": ["swa-Latn"],
    "te": ["tel-Telu"],
    "th": ["tha-Thai"],
    "yo": ["yor-Latn"],
    "zh": ["zho-Hans"],
}


def _load_miracl_data(
    path: str,
    langs: list,
    splits: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
):
    corpus = {lang: dict.fromkeys(splits) for lang in langs}
    queries = {lang: dict.fromkeys(splits) for lang in langs}
    relevant_docs = {lang: dict.fromkeys(splits) for lang in langs}

    split = _EVAL_SPLIT

    for lang in langs:
        # Load corpus data (Can be several millions for languages)
        corpus_identifier = f"corpus-{lang}"
        corpus_data = datasets.load_dataset(
            path,
            corpus_identifier,
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        corpus_data = corpus_data.map(
            lambda x: {
                "id": x["_id"],
                "text": x["text"],
                "modality": "text",
                "title": x["title"],
                "image_id": x["image_id"],
            },
            remove_columns=["_id"],
        )
        corpus[lang][split] = corpus_data[split]

        # Load queries data
        queries_identifier = f"queries-{lang}"
        queries_data = datasets.load_dataset(
            path,
            queries_identifier,
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        queries_data = queries_data.map(
            lambda x: {
                "id": x["_id"],
                "text": x["text"],
                "modality": "text",
                "image": None,
            },
            remove_columns=["_id"],
        )
        queries[lang][split] = queries_data[split]

        # Load relevant documents data
        qrels_identifier = f"qrels-{lang}"
        qrels_data = datasets.load_dataset(
            path,
            qrels_identifier,
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        relevant_docs[lang][split] = {}
        for row in qrels_data[split]:
            query_id = row["query-id"]
            doc_id = row["corpus-id"]
            score = row["score"]
            if query_id not in relevant_docs[lang][split]:
                relevant_docs[lang][split][query_id] = {}
            relevant_docs[lang][split][query_id][doc_id] = score

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)

    return corpus, queries, relevant_docs


class MIRACLVisionRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="MIRACLVisionRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "nvidia/miracl-vision",
            "revision": "309e1696433408fbd555959cf1da968f3814f8b6",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["default"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_5",
        date=("2025-03-01", "2025-06-01"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{osmulski2025miraclvisionlargemultilingualvisual,
      title={MIRACL-VISION: A Large, multilingual, visual document retrieval benchmark}, 
      author={Radek Osmulski and Gabriel de Souza P. Moreira and Ronay Ak and Mengyao Xu and Benedikt Schifferer and Even Oldridge},
      year={2025},
      eprint={2505.11651},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2505.11651}, 
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's query."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 30,
                    "num_queries": 228,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_miracl_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            langs=self.metadata_dict["eval_langs"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True
