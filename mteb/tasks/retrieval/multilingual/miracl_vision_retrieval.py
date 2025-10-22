import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

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
    splits: list[str],
    revision: str | None = None,
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
            revision=revision,
        )

        images_identifier = f"images-{lang}"
        images_data = datasets.load_dataset(
            path,
            images_identifier,
            revision=revision,
        )

        # For text data, it would look like this, just use _id column

        imgid2docid = {
            str(ex["image_id"]): str(ex["_id"])
            for ex in corpus_data[split]  # e.g. “train”, “validation”, etc.
        }

        images_data = images_data.map(
            lambda x: {
                "id": imgid2docid[str(x["file_name"])],
                "modality": "image",
            },
            remove_columns=["file_name"],
        )

        corpus[lang][split] = images_data[split]

        # Load queries data
        queries_identifier = f"queries-{lang}"
        queries_data = datasets.load_dataset(
            path,
            queries_identifier,
            revision=revision,
        )
        queries_data = queries_data.map(
            lambda x: {
                "id": str(x["_id"]),
                "text": x["text"],
                "modality": "text",
            },
            remove_columns=["_id"],
        )
        queries[lang][split] = queries_data[split]

        # Load relevant documents data
        qrels_identifier = f"qrels-{lang}"
        qrels_data = datasets.load_dataset(
            path,
            qrels_identifier,
            revision=revision,
        )
        relevant_docs[lang][split] = {}
        for row in qrels_data[split]:
            query_id = str(row["query-id"])
            doc_id = str(row["corpus-id"])
            score = row["score"]
            if query_id not in relevant_docs[lang][split]:
                relevant_docs[lang][split][query_id] = {}
            relevant_docs[lang][split][query_id][doc_id] = score

    return corpus, queries, relevant_docs


class MIRACLVisionRetrieval(AbsTaskRetrieval):
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
        sample_creation="created",
        bibtex_citation=r"""
@article{osmulski2025miraclvisionlargemultilingualvisual,
  author = {Radek Osmulski and Gabriel de Souza P. Moreira and Ronay Ak and Mengyao Xu and Benedikt Schifferer and Even Oldridge},
  eprint = {2505.11651},
  journal = {arxiv},
  title = {{MIRACL-VISION: A Large, multilingual, visual document retrieval benchmark}},
  url = {https://arxiv.org/abs/2505.11651},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's query."},
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_miracl_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            langs=self.hf_subsets,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True
