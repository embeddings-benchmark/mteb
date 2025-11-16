import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "test"

_LANGS = {
    # <iso_639_3>-<ISO_15924>
    "arabic": ["ara-Arab"],
    "chinese": ["zho-Hans"],
    "english": ["eng-Latn"],
    "french": ["fra-Latn"],
    "korean": ["kor-Hang"],
    "russian": ["rus-Cyrl"],
    "spanish": ["spa-Latn"],
    "vietnamese": ["vie-Latn"],
}


def _load_publichealthqa_data(
    path: str, langs: list, split: str, revision: str | None = None
):
    queries = {lang: {split: {}} for lang in langs}
    corpus = {lang: {split: {}} for lang in langs}
    relevant_docs = {lang: {split: {}} for lang in langs}

    for lang in langs:
        data = datasets.load_dataset(
            path,
            lang,
            split=split,
            revision=revision,
        )

        question_ids = {}
        answer_ids = {}

        for row in data:
            if row["question"] is not None and row["question"] not in question_ids:
                question_ids[row["question"]] = len(question_ids)
            if row["answer"] is not None and row["answer"] not in answer_ids:
                answer_ids[row["answer"]] = len(answer_ids)

        for row in data:
            if row["question"] is None or row["answer"] is None:
                # There are some questions and answers that are None in the original dataset, specifically in the Arabic subset.
                continue
            question = row["question"]
            answer = row["answer"]
            query_id = f"Q{question_ids[question]}"
            queries[lang][split][query_id] = question
            doc_id = f"D{answer_ids[answer]}"
            corpus[lang][split][doc_id] = {"text": answer}
            if query_id not in relevant_docs[lang][split]:
                relevant_docs[lang][split][query_id] = {}
            relevant_docs[lang][split][query_id][doc_id] = 1

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)

    return corpus, queries, relevant_docs


class PublicHealthQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PublicHealthQA",
        description="A multilingual dataset for public health question answering, based on FAQ sourced from CDC and WHO.",
        dataset={
            "path": "xhluca/publichealth-qa",
            "revision": "main",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        reference="https://huggingface.co/datasets/xhluca/publichealth-qa",
        date=("2020-01-01", "2020-04-15"),
        domains=["Medical", "Government", "Web", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{xing_han_lu_2024,
  author = { {Xing Han Lu} },
  doi = { 10.57967/hf/2247 },
  publisher = { Hugging Face },
  title = { publichealth-qa (Revision 3b67b6b) },
  url = { https://huggingface.co/datasets/xhluca/publichealth-qa },
  year = {2024},
}
""",
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_publichealthqa_data(
            path=self.metadata.dataset["path"],
            langs=self.hf_subsets,
            split=self.metadata.eval_splits[0],
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True
