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
        # Load corpus data (Can be several millions for languages)
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
            docid = row["docid"]
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
            query_id = row["query_id"]
            query_text = row["query"]
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
        name="MIRACLRetrieval",
        description="MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages.",
        reference="http://miracl.ai",
        dataset={
            "path": "miracl/mmteb-miracl",
            "revision": "main",
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
                    "ar": {
                        "average_document_length": 318.6539598547405,
                        "average_query_length": 29.480662983425415,
                        "num_documents": 2061414,
                        "num_queries": 2896,
                        "average_relevant_docs_per_query": 1.953729281767956,
                    },
                    "bn": {
                        "average_document_length": 383.2428136511194,
                        "average_query_length": 46.98053527980535,
                        "num_documents": 297265,
                        "num_queries": 411,
                        "average_relevant_docs_per_query": 2.099756690997567,
                    },
                    "de": {
                        "average_document_length": 414.28004442393404,
                        "average_query_length": 46.0,
                        "num_documents": 15866222,
                        "num_queries": 305,
                        "average_relevant_docs_per_query": 2.6590163934426227,
                    },
                    "en": {
                        "average_document_length": 401.0042914921588,
                        "average_query_length": 40.247809762202756,
                        "num_documents": 32893221,
                        "num_queries": 799,
                        "average_relevant_docs_per_query": 2.911138923654568,
                    },
                    "es": {
                        "average_document_length": 403.71153493754986,
                        "average_query_length": 47.373456790123456,
                        "num_documents": 10373953,
                        "num_queries": 648,
                        "average_relevant_docs_per_query": 4.609567901234568,
                    },
                    "fa": {
                        "average_document_length": 262.6478385010321,
                        "average_query_length": 41.1503164556962,
                        "num_documents": 2207172,
                        "num_queries": 632,
                        "average_relevant_docs_per_query": 2.079113924050633,
                    },
                    "fi": {
                        "average_document_length": 359.87767671935734,
                        "average_query_length": 38.63493312352478,
                        "num_documents": 1883509,
                        "num_queries": 1271,
                        "average_relevant_docs_per_query": 1.925255704169945,
                    },
                    "fr": {
                        "average_document_length": 343.6283550271699,
                        "average_query_length": 43.883381924198254,
                        "num_documents": 14636953,
                        "num_queries": 343,
                        "average_relevant_docs_per_query": 2.131195335276968,
                    },
                    "hi": {
                        "average_document_length": 370.96196845914386,
                        "average_query_length": 53.34,
                        "num_documents": 506264,
                        "num_queries": 350,
                        "average_relevant_docs_per_query": 2.1485714285714286,
                    },
                    "id": {
                        "average_document_length": 350.2785651811673,
                        "average_query_length": 37.958333333333336,
                        "num_documents": 1446315,
                        "num_queries": 960,
                        "average_relevant_docs_per_query": 3.216666666666667,
                    },
                    "ja": {
                        "average_document_length": 145.8538220556965,
                        "average_query_length": 17.71395348837209,
                        "num_documents": 6953614,
                        "num_queries": 860,
                        "average_relevant_docs_per_query": 2.0813953488372094,
                    },
                    "ko": {
                        "average_document_length": 173.97649170809927,
                        "average_query_length": 21.624413145539908,
                        "num_documents": 1486752,
                        "num_queries": 213,
                        "average_relevant_docs_per_query": 2.568075117370892,
                    },
                    "ru": {
                        "average_document_length": 332.2475377512674,
                        "average_query_length": 44.13258785942492,
                        "num_documents": 9543918,
                        "num_queries": 1252,
                        "average_relevant_docs_per_query": 2.8434504792332267,
                    },
                    "sw": {
                        "average_document_length": 228.71348655286377,
                        "average_query_length": 38.97095435684647,
                        "num_documents": 131924,
                        "num_queries": 482,
                        "average_relevant_docs_per_query": 1.887966804979253,
                    },
                    "te": {
                        "average_document_length": 396.2108674545774,
                        "average_query_length": 38.11231884057971,
                        "num_documents": 518079,
                        "num_queries": 828,
                        "average_relevant_docs_per_query": 1.0314009661835748,
                    },
                    "th": {
                        "average_document_length": 356.8283496198581,
                        "average_query_length": 42.87585266030014,
                        "num_documents": 542166,
                        "num_queries": 733,
                        "average_relevant_docs_per_query": 1.8321964529331514,
                    },
                    "yo": {
                        "average_document_length": 159.35250698366738,
                        "average_query_length": 37.6890756302521,
                        "num_documents": 49043,
                        "num_queries": 119,
                        "average_relevant_docs_per_query": 1.2100840336134453,
                    },
                    "zh": {
                        "average_document_length": 119.9458931721347,
                        "average_query_length": 10.867684478371501,
                        "num_documents": 4934368,
                        "num_queries": 393,
                        "average_relevant_docs_per_query": 2.5292620865139948,
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


def _load_miracl_data_hard_negatives(
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
            # load the hard negatives miracle dataset
            # Load corpus data
            print(f"Loading data for {lang}")
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


class MIRACLRetrievalHardNegatives(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MIRACLRetrievalHardNegatives",
        description="MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.",
        reference="http://miracl.ai",
        dataset={
            "path": "mteb/miracl-hard-negatives",
            "revision": "95c8db7d4a6e9c1d8a60601afd63d553ae20a2eb",
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
                    "average_document_length": 417.6655323669399,
                    "average_query_length": 37.46957385337667,
                    "num_documents": 2449382,
                    "num_queries": 11076,
                    "average_relevant_docs_per_query": 2.3643011917659806,
                    "hf_subset_descriptive_stats": {
                        "ar": {
                            "average_document_length": 438.1872433017704,
                            "average_query_length": 29.584,
                            "num_documents": 192103,
                            "num_queries": 1000,
                            "average_relevant_docs_per_query": 1.982,
                        },
                        "bn": {
                            "average_document_length": 383.2428136511194,
                            "average_query_length": 46.98053527980535,
                            "num_documents": 297265,
                            "num_queries": 411,
                            "average_relevant_docs_per_query": 2.099756690997567,
                        },
                        "de": {
                            "average_document_length": 513.7796484139344,
                            "average_query_length": 46.0,
                            "num_documents": 71277,
                            "num_queries": 305,
                            "average_relevant_docs_per_query": 2.6590163934426227,
                        },
                        "en": {
                            "average_document_length": 529.2486406963214,
                            "average_query_length": 40.247809762202756,
                            "num_documents": 178768,
                            "num_queries": 799,
                            "average_relevant_docs_per_query": 2.911138923654568,
                        },
                        "es": {
                            "average_document_length": 535.8023645655877,
                            "average_query_length": 47.373456790123456,
                            "num_documents": 146750,
                            "num_queries": 648,
                            "average_relevant_docs_per_query": 4.609567901234568,
                        },
                        "fa": {
                            "average_document_length": 411.2648282882721,
                            "average_query_length": 41.1503164556962,
                            "num_documents": 133596,
                            "num_queries": 632,
                            "average_relevant_docs_per_query": 2.079113924050633,
                        },
                        "fi": {
                            "average_document_length": 462.9445310289844,
                            "average_query_length": 38.646,
                            "num_documents": 194415,
                            "num_queries": 1000,
                            "average_relevant_docs_per_query": 1.918,
                        },
                        "fr": {
                            "average_document_length": 460.40909271865917,
                            "average_query_length": 43.883381924198254,
                            "num_documents": 75357,
                            "num_queries": 343,
                            "average_relevant_docs_per_query": 2.131195335276968,
                        },
                        "hi": {
                            "average_document_length": 498.6759426632417,
                            "average_query_length": 53.34,
                            "num_documents": 63066,
                            "num_queries": 350,
                            "average_relevant_docs_per_query": 2.1485714285714286,
                        },
                        "id": {
                            "average_document_length": 494.1689807519638,
                            "average_query_length": 37.958333333333336,
                            "num_documents": 168173,
                            "num_queries": 960,
                            "average_relevant_docs_per_query": 3.216666666666667,
                        },
                        "ja": {
                            "average_document_length": 206.13654293407583,
                            "average_query_length": 17.71395348837209,
                            "num_documents": 185319,
                            "num_queries": 860,
                            "average_relevant_docs_per_query": 2.0813953488372094,
                        },
                        "ko": {
                            "average_document_length": 257.82646155267594,
                            "average_query_length": 21.624413145539908,
                            "num_documents": 43293,
                            "num_queries": 213,
                            "average_relevant_docs_per_query": 2.568075117370892,
                        },
                        "ru": {
                            "average_document_length": 476.0820349224605,
                            "average_query_length": 44.055,
                            "num_documents": 219114,
                            "num_queries": 1000,
                            "average_relevant_docs_per_query": 2.833,
                        },
                        "sw": {
                            "average_document_length": 228.71348655286377,
                            "average_query_length": 38.97095435684647,
                            "num_documents": 131924,
                            "num_queries": 482,
                            "average_relevant_docs_per_query": 1.887966804979253,
                        },
                        "te": {
                            "average_document_length": 601.7099283059209,
                            "average_query_length": 38.11231884057971,
                            "num_documents": 101961,
                            "num_queries": 828,
                            "average_relevant_docs_per_query": 1.0314009661835748,
                        },
                        "th": {
                            "average_document_length": 478.8818849711528,
                            "average_query_length": 42.87585266030014,
                            "num_documents": 116649,
                            "num_queries": 733,
                            "average_relevant_docs_per_query": 1.8321964529331514,
                        },
                        "yo": {
                            "average_document_length": 159.35250698366738,
                            "average_query_length": 37.6890756302521,
                            "num_documents": 49043,
                            "num_queries": 119,
                            "average_relevant_docs_per_query": 1.2100840336134453,
                        },
                        "zh": {
                            "average_document_length": 147.36211243527777,
                            "average_query_length": 10.867684478371501,
                            "num_documents": 81309,
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

        self.corpus, self.queries, self.relevant_docs = (
            _load_miracl_data_hard_negatives(
                path=self.metadata_dict["dataset"]["path"],
                langs=self.hf_subsets,
                splits=self.metadata_dict["eval_splits"],
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata_dict["dataset"]["revision"],
            )
        )

        self.data_loaded = True
