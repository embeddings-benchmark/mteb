from __future__ import annotations

import datasets

from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

_EVAL_SPLIT = "test"

_LANGUAGES = {
    "ara": ["ara-Arab"],
    "aze": ["aze-Latn"],
    "ben": ["ben-Beng"],
    "bul": ["bul-Cyrl"],
    "cat": ["cat-Latn"],
    "ces": ["ces-Latn"],
    "dan": ["dan-Latn"],
    "deu": ["deu-Latn"],
    "ell": ["ell-Grek"],
    "eng": ["eng-Latn"],
    "est": ["est-Latn"],
    "fas": ["fas-Arab"],
    "fin": ["fin-Latn"],
    "fra": ["fra-Latn"],
    "heb": ["heb-Hebr"],
    "hin": ["hin-Deva"],
    "hrv": ["hrv-Latn"],
    "hun": ["hun-Latn"],
    "ind": ["ind-Latn"],
    "isl": ["isl-Latn"],
    "ita": ["ita-Latn"],
    "jpn": ["jpn-Jpan"],
    "kat": ["kat-Geor"],
    "kaz": ["kaz-Cyrl"],
    "kor": ["kor-Kore"],
    "lav": ["lav-Latn"],
    "lit": ["lit-Latn"],
    "mar": ["mar-Deva"],
    "msa": ["msa-Latn"],
    "nld": ["nld-Latn"],
    "nor": ["nor-Latn"],
    "pol": ["pol-Latn"],
    "por": ["por-Latn"],
    "ron": ["ron-Latn"],
    "rus": ["rus-Cyrl"],
    "slk": ["slk-Latn"],
    "slv": ["slv-Latn"],
    "spa": ["spa-Latn"],
    "sqi": ["sqi-Latn"],
    "srp": ["srp-Cyrl"],
    "swe": ["swe-Latn"],
    "tgl": ["tgl-Latn"],
    "tha": ["tha-Thai"],
    "tur": ["tur-Latn"],
    "ukr": ["ukr-Cyrl"],
    "urd": ["urd-Arab"],
    "uzb": ["uzb-Latn"],
    "vie": ["vie-Latn"],
    "zho": ["zho-Hans"],
}


def _load_webfaq_data(
    path: str, langs: list, splits: str, cache_dir: str = None, revision: str = None
):
    corpus = {lang: {split: None for split in splits} for lang in langs}
    queries = {lang: {split: None for split in splits} for lang in langs}
    relevant_docs = {lang: {split: None for split in splits} for lang in langs}

    split = _EVAL_SPLIT

    for lang in langs:
        # Load corpus data (Can be several millions for languages)
        corpus_identifier = f"{lang}-corpus"
        corpus_data = datasets.load_dataset(
            path,
            corpus_identifier,
            cache_dir=cache_dir,
            revision=revision,
        )
        corpus[lang][split] = {}
        for row in corpus_data["corpus"]:
            corpus_id = row["_id"]
            title = row["title"]
            text = row["text"]
            corpus[lang][split][corpus_id] = {"title": title, "text": text}

        # Load queries data
        queries_identifier = f"{lang}-queries"
        queries_data = datasets.load_dataset(
            path,
            queries_identifier,
            cache_dir=cache_dir,
            revision=revision,
        )
        queries[lang][split] = {}
        for row in queries_data[split]:
            query_id = row["_id"]
            text = row["text"]
            queries[lang][split][query_id] = text

        # Load relevant documents data
        qrels_identifier = f"{lang}-qrels"
        qrels_data = datasets.load_dataset(
            path,
            qrels_identifier,
            cache_dir=cache_dir,
            revision=revision,
        )
        relevant_docs[lang][split] = {}
        for row in qrels_data[split]:
            query_id = row["query-id"]
            corpus_id = row["corpus-id"]
            score = row["score"]
            if query_id not in relevant_docs[lang][split]:
                relevant_docs[lang][split][query_id] = {}
            relevant_docs[lang][split][query_id][corpus_id] = int(score)

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)

    return corpus, queries, relevant_docs


class WebFAQRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WebFAQRetrieval",
        description="WebFAQ is a broad-coverage corpus of natural question-answer pairs in 75 languages, gathered from FAQ pages on the web.",
        reference="https://huggingface.co/PaDaS-Lab",
        dataset={
            "path": "PaDaS-Lab/webfaq-retrieval",
            "revision": "c3262adb1c32ac0c3ea8de6393a44366edaa62e1",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2022-09-01", "2024-10-01"),
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{dinzinger2025webfaq,
  archiveprefix = {arXiv},
  author = {Michael Dinzinger and Laura Caspari and Kanishka Ghosh Dastidar and Jelena Mitrović and Michael Granitzer},
  eprint = {2502.20936},
  primaryclass = {cs.CL},
  title = {WebFAQ: A Multilingual Collection of Natural Q&amp;A Datasets for Dense Retrieval},
  url = {https://arxiv.org/abs/2502.20936},
  year = {2025},
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_webfaq_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.hf_subsets,
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True
