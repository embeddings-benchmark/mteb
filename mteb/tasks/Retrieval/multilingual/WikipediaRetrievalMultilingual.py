from __future__ import annotations

from datasets import load_dataset, DatasetDict

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import MultilingualTask
from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

# _EVAL_LANGS = {
#     "bg": ["bul-Cyrl"],
#     "bn": ["ben-Beng"],
#     "cs": ["ces-Latn"],
#     # "da": ["dan-Latn"],
#     "de": ["deu-Latn"],
#     # "en": ["eng-Latn"],
#     # "fa": ["fas-Arab"],
#     # "fi": ["fin-Latn"],
#     # "hi": ["hin-Deva"],
#     "it": ["ita-Latn"],
#     "nl": ["nld-Latn"],
#     "pt": ["por-Latn"],
#     "ro": ["ron-Latn"],
#     "sr": ["srp-Cyrl"],
#     # "no": ["nor-Latn"],
#     # "sv": ["swe-Latn"]
# }

_EVAL_LANGS = {
    "bg": ["bul-Cyrl"],
    "bn": ["ben-Beng"],
}

# adapted from MIRACLRetrieval
def _load_data(
    path: str, langs: list, split: str, cache_dir: str = None, revision_queries: str = None, revision_corpus: str = None, revision_qrels: str = None
):
    queries = {lang: {split: {}} for lang in langs}
    corpus = {lang: {split: {}} for lang in langs}
    qrels = {lang: {split: {}} for lang in langs}

    for lang in langs:
        print(f"LOADING {lang}")
        queries_path = path
        corpus_path = path.replace("queries", "corpus")
        qrels_path = path.replace("queries", "qrels")
        queries_lang = load_dataset(
            queries_path,
            lang,
            split=split,
            cache_dir=cache_dir,
            revision=revision_queries,
        )
        # ).to_list()
        corpus_lang = load_dataset(
            corpus_path,
            lang,
            split=split,
            cache_dir=cache_dir,
            revision=revision_corpus,
        )
        # .to_list()
        qrels_lang = load_dataset(
            qrels_path,
            lang,
            split=split,
            cache_dir=cache_dir,
            revision=revision_qrels,
        )
        # .to_list()
        # breakpoint()
        # don't pass on titles for extra diffculty
        # corpus_lang_dict = {doc["_id"]: {"text": doc["text"]} for doc in corpus_lang}
        # queries_lang_dict = {query["_id"]: {"text": query["text"]} for query in queries_lang}
        # qrels_lang_dict = {qrel["query-id"]: {qrel["corpus-id"]: qrel["score"]} for qrel in qrels_lang}
        corpus_lang_dict = {doc["_id"]: {"text": doc["text"], "title": doc["title"]} for doc in corpus_lang}
        queries_lang_dict = {query["_id"]: {"text": query["text"]} for query in queries_lang}
        # qrels_lang_dict = {qrel["query-id"]: {qrel["corpus-id"]: qrel["score"]} for qrel in qrels_lang}
        qrels_lang_dict = {}
        for qrel in qrels_lang:
            if qrel["score"] == 0.5:
                continue
            # score = 0 if qrel["score"] == 0.5 else qrel["score"]
            # score = int(score)
            score = int(qrel["score"])
            qrels_lang_dict[qrel["query-id"]] = {qrel["corpus-id"]: score}
        # breakpoint()
        corpus[lang][split] = corpus_lang_dict
        queries[lang][split] = queries_lang_dict
        qrels[lang][split] = qrels_lang_dict
        # corpus[lang][split] = corpus_lang
        # queries[lang][split] = queries_lang
        # qrels[lang][split] = qrels_lang

    # corpus = DatasetDict(corpus)
    # queries = DatasetDict(queries)
    # qrels = DatasetDict(qrels)

    return corpus, queries, qrels

class WikipediaRetrievalMultilingual(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WikipediaRetrievalMultilingual",
        description="The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.",
        reference="https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-pt",
        dataset={
            "path": "ellamind/wikipedia-2023-11-retrieval-multilingual-queries",
            "revision": "7389b05119b949bdc032c6cb07c6f51d7adc076a", # avoid validation error
            "revision_queries": "7389b05119b949bdc032c6cb07c6f51d7adc076a",
            "revision_corpus": "9b66888cd2cd65b60dd9d0a79da3fc497ad38eb2",
            "revision_qrels": "c6aa67a562dd670eb0bd37a72f2e705b6a4d93e1"
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2023-11-01", "2024-05-15"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Question answering", "Article retrieval"],
        license="cc-by-sa-3.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="LM-generated and verified",
        bibtex_citation="",
        n_samples={"test": 1500},
        avg_character_length={"test": 452},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.hf_subsets,
            split=self.metadata_dict["eval_splits"][0],
            cache_dir=kwargs.get("cache_dir", None),
            revision_queries=self.metadata_dict["dataset"]["revision_queries"],
            revision_corpus=self.metadata_dict["dataset"]["revision_corpus"],
            revision_qrels=self.metadata_dict["dataset"]["revision_qrels"],
        )


        self.data_loaded = True