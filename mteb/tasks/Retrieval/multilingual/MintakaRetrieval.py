from __future__ import annotations

import datasets
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import MultilingualTask
from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

_EVAL_SPLIT = "test"
_LANGS = {
    "ar": ["ara-Arab"],
    "de": ["deu-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "hi": ["hin-Deva"],
    "it": ["ita-Latn"],
    "ja": ["jpn-Hira"],
    "pt": ["por-Latn"],
}


def _load_mintaka_data(
    path: str, langs: list, split: str, cache_dir: str = None, revision: str = None
):
    queries = {lang: {split: {}} for lang in langs}
    corpus = {lang: {split: {}} for lang in langs}
    relevant_docs = {lang: {split: {}} for lang in langs}

    for lang in langs:
        data = datasets.load_dataset(
            path,
            lang,
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )
        question_ids = {
            question: _id for _id, question in enumerate(set(data["question"]))
        }
        answer_ids = {answer: _id for _id, answer in enumerate(set(data["answer"]))}

        for row in data:
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


class MintakaRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MintakaRetrieval",
        description="MintakaRetrieval",
        reference=None,
        dataset={
            "path": "jinaai/mintakaqa",
            "revision": "efa78cc2f74bbcd21eff2261f9e13aebe40b814e",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_mintaka_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.metadata.eval_langs,
            split=self.metadata_dict["eval_splits"][0],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True
