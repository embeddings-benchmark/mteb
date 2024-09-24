from __future__ import annotations

import datasets

from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

_EVAL_SPLIT = "dev"

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
    path: str, langs: list, splits: str, cache_dir: str = None, revision: str = None
):
    corpus = {lang: {split: None for split in splits} for lang in langs}
    queries = {lang: {split: None for split in splits} for lang in langs}
    relevant_docs = {lang: {split: None for split in splits} for lang in langs}

    split = _EVAL_SPLIT

    for lang in langs:
        # subsampled langs: th,en,de,fr,es,ru,ja,fa,ar,fi,ko,id,te,hi,zh
        if lang in [
            "th",
            "en",
            "de",
            "fr",
            "es",
            "ru",
            "ja",
            "fa",
            "ar",
            "fi",
            "ko",
            "id",
            "te",
            "hi",
            "zh",
        ]:
            # load the fast miracle dataset
            # Load corpus data
            print(lang)
            corpus_identifier = f"corpus-{lang}"
            corpus_data = datasets.load_dataset(
                path,
                corpus_identifier,
                cache_dir=cache_dir,
                revision=revision,
                trust_remote_code=True,
            )
            corpus[lang][split] = {}
            for row in corpus_data["corpus"]:
                docid = row["_id"]
                doc_title = row["title"]
                doc_text = row["text"]
                corpus[lang][split][docid] = {"title": doc_title, "text": doc_text}

            # Load queries data
            queries_identifier = f"queries-{lang}"
            queries_data = datasets.load_dataset(
                path,
                queries_identifier,
                cache_dir=cache_dir,
                revision=revision,
                trust_remote_code=True,
            )
            queries[lang][split] = {}
            for row in queries_data["queries"]:
                query_id = row["_id"]
                query_text = row["text"]
                queries[lang][split][query_id] = query_text

            # Load relevant documents data
            qrels_identifier = f"{lang}"
            qrels_data = datasets.load_dataset(
                path,
                qrels_identifier,
                cache_dir=cache_dir,
                revision=revision,
                trust_remote_code=True,
            )
            relevant_docs[lang][split] = {}
            for row in qrels_data[split]:
                query_id = row["query-id"]
                doc_id = row["corpus-id"]
                score = row["score"]
                if query_id not in relevant_docs[lang][split]:
                    relevant_docs[lang][split][query_id] = {}
                relevant_docs[lang][split][query_id][doc_id] = score

        else:
            corpus_identifier = f"corpus-{lang}"
            corpus_data = datasets.load_dataset(
                "miracl/mmteb-miracl",
                corpus_identifier,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            corpus[lang][split] = {}
            for row in corpus_data["corpus"]:
                docid = row["docid"]
                doc_title = row["title"]
                doc_text = row["text"]
                corpus[lang][split][docid] = {"title": doc_title, "text": doc_text}

            # Load queries data
            queries_identifier = f"queries-{lang}"
            queries_data = datasets.load_dataset(
                "miracl/mmteb-miracl",
                queries_identifier,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            queries[lang][split] = {}
            for row in queries_data["queries"]:
                query_id = row["query_id"]
                query_text = row["query"]
                queries[lang][split][query_id] = query_text

            # Load relevant documents data
            qrels_identifier = f"{lang}"
            qrels_data = datasets.load_dataset(
                "miracl/mmteb-miracl",
                qrels_identifier,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            relevant_docs[lang][split] = {}
            for row in qrels_data[split]:
                query_id = row["query_id"]
                doc_id = row["docid"]
                score = row["score"]
                if query_id not in relevant_docs[lang][split]:
                    relevant_docs[lang][split][query_id] = {}
                relevant_docs[lang][split][query_id][doc_id] = score

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)

    return corpus, queries, relevant_docs


class MIRACLRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MIRACLRetrieval-Fast",
        description="MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages.",
        reference="http://miracl.ai",
        dataset={
            "path": "mteb/miracl-fast",
            "revision": "74532329bc23a24f3a30d5b27317638db5b5ba74",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2022-06-01", "2023-01-30"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""@article{10.1162/tacl_a_00595,
    author = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
    title = "{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {11},
    pages = {1114-1131},
    year = {2023},
    month = {09},
    abstract = "{MIRACL is a multilingual dataset for ad hoc retrieval across 18 languages that collectively encompass over three billion native speakers around the world. This resource is designed to support monolingual retrieval tasks, where the queries and the corpora are in the same language. In total, we have gathered over 726k high-quality relevance judgments for 78k queries over Wikipedia in these languages, where all annotations have been performed by native speakers hired by our team. MIRACL covers languages that are both typologically close as well as distant from 10 language families and 13 sub-families, associated with varying amounts of publicly available resources. Extensive automatic heuristic verification and manual assessments were performed during the annotation process to control data quality. In total, MIRACL represents an investment of around five person-years of human annotator effort. Our goal is to spur research on improving retrieval across a continuum of languages, thus enhancing information access capabilities for diverse populations around the world, particularly those that have traditionally been underserved. MIRACL is available at http://miracl.ai/.}",
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00595},
    url = {https://doi.org/10.1162/tacl\_a\_00595},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00595/2157340/tacl\_a\_00595.pdf},
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "dev": {
                    "average_document_length": 419.0049268922039,
                    "average_query_length": 37.46957385337667,
                    "num_documents": 2784108,
                    "num_queries": 11076,
                    "average_relevant_docs_per_query": 2.289007486472463,
                    "hf_subset_descriptive_stats": {
                        "ar": {
                            "average_document_length": 426.5111760859585,
                            "average_query_length": 29.584,
                            "num_documents": 431278,
                            "num_queries": 1000,
                            "average_relevant_docs_per_query": 1.9437154696132597,
                        },
                        "bn": {
                            "average_document_length": 383.2428136511194,
                            "average_query_length": 46.98053527980535,
                            "num_documents": 297265,
                            "num_queries": 411,
                            "average_relevant_docs_per_query": 2.099756690997567,
                        },
                        "de": {
                            "average_document_length": 513.6786163873892,
                            "average_query_length": 46.0,
                            "num_documents": 71494,
                            "num_queries": 305,
                            "average_relevant_docs_per_query": 2.6590163934426227,
                        },
                        "en": {
                            "average_document_length": 529.0259906785898,
                            "average_query_length": 40.247809762202756,
                            "num_documents": 179372,
                            "num_queries": 799,
                            "average_relevant_docs_per_query": 2.8385481852315393,
                        },
                        "es": {
                            "average_document_length": 535.6423647193866,
                            "average_query_length": 47.373456790123456,
                            "num_documents": 147231,
                            "num_queries": 648,
                            "average_relevant_docs_per_query": 4.609567901234568,
                        },
                        "fa": {
                            "average_document_length": 411.05307733635794,
                            "average_query_length": 41.1503164556962,
                            "num_documents": 134012,
                            "num_queries": 632,
                            "average_relevant_docs_per_query": 2.079113924050633,
                        },
                        "fi": {
                            "average_document_length": 461.00697711742,
                            "average_query_length": 38.646,
                            "num_documents": 236774,
                            "num_queries": 1000,
                            "average_relevant_docs_per_query": 1.9133858267716535,
                        },
                        "fr": {
                            "average_document_length": 460.33350970950846,
                            "average_query_length": 43.883381924198254,
                            "num_documents": 75596,
                            "num_queries": 343,
                            "average_relevant_docs_per_query": 2.131195335276968,
                        },
                        "hi": {
                            "average_document_length": 498.55261327346886,
                            "average_query_length": 53.34,
                            "num_documents": 63254,
                            "num_queries": 350,
                            "average_relevant_docs_per_query": 2.1485714285714286,
                        },
                        "id": {
                            "average_document_length": 493.96505208981864,
                            "average_query_length": 37.958333333333336,
                            "num_documents": 168651,
                            "num_queries": 960,
                            "average_relevant_docs_per_query": 3.171875,
                        },
                        "ja": {
                            "average_document_length": 206.1113071923557,
                            "average_query_length": 17.71395348837209,
                            "num_documents": 185864,
                            "num_queries": 860,
                            "average_relevant_docs_per_query": 2.0547147846332945,
                        },
                        "ko": {
                            "average_document_length": 257.7614288017319,
                            "average_query_length": 21.624413145539908,
                            "num_documents": 43421,
                            "num_queries": 213,
                            "average_relevant_docs_per_query": 2.5305164319248825,
                        },
                        "ru": {
                            "average_document_length": 474.7900288882676,
                            "average_query_length": 44.055,
                            "num_documents": 268275,
                            "num_queries": 1000,
                            "average_relevant_docs_per_query": 2.801916932907348,
                        },
                        "sw": {
                            "average_document_length": 228.71348655286377,
                            "average_query_length": 38.97095435684647,
                            "num_documents": 131924,
                            "num_queries": 482,
                            "average_relevant_docs_per_query": 1.887966804979253,
                        },
                        "te": {
                            "average_document_length": 601.5159486978657,
                            "average_query_length": 38.11231884057971,
                            "num_documents": 102140,
                            "num_queries": 828,
                            "average_relevant_docs_per_query": 1.0314769975786926,
                        },
                        "th": {
                            "average_document_length": 478.8321933034646,
                            "average_query_length": 42.87585266030014,
                            "num_documents": 116956,
                            "num_queries": 733,
                            "average_relevant_docs_per_query": 1.8267394270122783,
                        },
                        "yo": {
                            "average_document_length": 159.35250698366738,
                            "average_query_length": 37.6890756302521,
                            "num_documents": 49043,
                            "num_queries": 119,
                            "average_relevant_docs_per_query": 1.2100840336134453,
                        },
                        "zh": {
                            "average_document_length": 147.34962848524975,
                            "average_query_length": 10.867684478371501,
                            "num_documents": 81558,
                            "num_queries": 393,
                            "average_relevant_docs_per_query": 2.5292620865139948,
                        },
                    },
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_miracl_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.hf_subsets,
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True
