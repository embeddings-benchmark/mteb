from __future__ import annotations

from collections import defaultdict

from datasets import load_dataset

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGS = {
    "ar": ["ara-Arab"],
    "bn": ["ben-Beng"],
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "hi": ["hin-Deva"],
    "hu": ["hun-Latn"],
    "id": ["ind-Latn"],
    "it": ["ita-Latn"],
    "jp": ["jpn-Jpan"],
    "ko": ["kor-Hang"],
    "my": ["mya-Mymr"],
    "nl": ["nld-Latn"],
    "pl": ["pol-Latn"],
    "pt": ["por-Latn"],
    "ru": ["rus-Cyrl"],
    "th": ["tha-Thai"],
    "tr": ["tur-Latn"],
    "ur": ["urd-Arab"],
    "vi": ["vie-Latn"],
    "zh": ["zho-Hans"],
}


def get_langs(langs: list[str]) -> dict[str, list[str]]:
    return {lang: _LANGS[lang] for lang in langs}


COMMON_METADATA = {
    "description": "Retrieve associated pages according to questions or related text.",
    "reference": "https://arxiv.org/abs/2506.18902",
    "type": "DocumentUnderstanding",
    "category": "t2i",
    "eval_splits": ["test"],
    "main_score": "ndcg_at_5",
    "task_subtypes": ["Image Text Retrieval"],
    "dialect": [],
    "modalities": ["text", "image"],
    "bibtex_citation": r"""
@misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      title={jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      author={Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Sedigheh Eslami and Scott Martens and Bo Wang and Nan Wang and Han Xiao},
      year={2025},
      eprint={2506.18902},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.18902},
}
""",
    "prompt": {"query": "Find a screenshot that is relevant to the user's input."},
}


def _load_single_language(
    path: str,
    split: str,
    lang: str | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
):
    print(path, split, lang, cache_dir, revision)
    query_ds = load_dataset(
        path,
        data_dir=f"{lang}/queries" if lang else "queries",
        split=split,
        cache_dir=cache_dir,
        revision=revision,
    )
    query_ds = query_ds.map(
        lambda x: {
            "id": f"query-{split}-{x['query-id']}",
            "text": x["query"],
            "image": None,
            "modality": "text",
        },
        remove_columns=["query-id", "query"],
    )

    corpus_ds = load_dataset(
        path,
        data_dir=f"{lang}/corpus" if lang else "corpus",
        split=split,
        cache_dir=cache_dir,
        revision=revision,
    )
    corpus_ds = corpus_ds.map(
        lambda x: {
            "id": f"corpus-{split}-{x['corpus-id']}",
            "text": None,
            "modality": "image",
        },
        remove_columns=["corpus-id"],
    )

    qrels_ds = load_dataset(
        path,
        data_dir=f"{lang}/qrels" if lang else "qrels",
        split=split,
        cache_dir=cache_dir,
        revision=revision,
    )

    return query_ds, corpus_ds, qrels_ds


def _load_data(
    path: str,
    splits: str,
    langs: list | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
):
    if langs is None or len(langs) == 1:
        corpus = {}
        queries = {}
        relevant_docs = {}
        langs = ["default"]
    else:
        corpus = {lang: {} for lang in langs}
        queries = {lang: {} for lang in langs}
        relevant_docs = {lang: {} for lang in langs}

    for split in splits:
        for lang in langs:
            query_ds, corpus_ds, qrels_ds = _load_single_language(
                path=path,
                split=split,
                lang=lang,
                cache_dir=cache_dir,
                revision=revision,
            )

            if lang == "default":
                queries[split] = query_ds
                corpus[split] = corpus_ds
                relevant_docs[split] = defaultdict(dict)
                for row in qrels_ds:
                    qid = f"query-{split}-{row['query-id']}"
                    did = f"corpus-{split}-{row['corpus-id']}"
                    relevant_docs[split][qid][did] = int(row["score"])
            else:
                queries[lang][split] = query_ds

                corpus[lang][split] = corpus_ds

                relevant_docs[lang][split] = defaultdict(dict)
                for row in qrels_ds:
                    qid = f"query-{split}-{row['query-id']}"
                    did = f"corpus-{split}-{row['corpus-id']}"
                    relevant_docs[lang][split][qid][did] = int(row["score"])

    return corpus, queries, relevant_docs


def load_data(self, **kwargs):
    if self.data_loaded:
        return

    self.corpus, self.queries, self.relevant_docs = _load_data(
        path=self.metadata_dict["dataset"]["path"],
        splits=self.metadata_dict["eval_splits"],
        langs=self.metadata_dict["eval_langs"],
        cache_dir=kwargs.get("cache_dir", None),
        revision=self.metadata_dict["dataset"]["revision"],
    )

    self.data_loaded = True


class JinaVDRMedicalPrescriptionsRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRMedicalPrescriptionsRetrieval",
        dataset={
            "path": "jinaai/medical-prescriptions_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Medical"],
        license="not specified",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 100,
                    "num_queries": 100,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRStanfordSlideRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRStanfordSlideRetrieval",
        dataset={
            "path": "jinaai/stanford_slide_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Academic"],
        license="not specified",
        annotations_creators="human-annotated",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 994,
                    "num_queries": 13,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRDonutVQAISynHMPRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRDonutVQAISynHMPRetrieval",
        dataset={
            "path": "jinaai/donut_vqa_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Medical"],
        license="not specified",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 800,
                    "num_queries": 704,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRTableVQARetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRTableVQARetrieval",
        dataset={
            "path": "jinaai/table-vqa_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Academic"],
        license="not specified",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 385,
                    "num_queries": 992,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRChartQARetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRChartQARetrieval",
        dataset={
            "path": "jinaai/ChartQA_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="not specified",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 834,
                    "num_queries": 996,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRTQARetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRTQARetrieval",
        dataset={
            "path": "jinaai/tqa_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Academic"],
        license="cc-by-nc-3.0",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 393,
                    "num_queries": 981,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDROpenAINewsRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDROpenAINewsRetrieval",
        dataset={
            "path": "jinaai/openai-news_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["News", "Web"],
        license="not specified",
        annotations_creators="human-annotated",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 30,
                    "num_queries": 31,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDREuropeanaDeNewsRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDREuropeanaDeNewsRetrieval",
        dataset={
            "path": "jinaai/europeana-de-news_beir",
            "revision": "main",
        },
        eval_langs=["deu-Latn"],
        domains=["News"],
        license="cc0-1.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 137,
                    "num_queries": 379,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDREuropeanaEsNewsRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDREuropeanaEsNewsRetrieval",
        dataset={
            "path": "jinaai/europeana-es-news_beir",
            "revision": "main",
        },
        eval_langs=["spa-Latn"],
        domains=["News"],
        license="cc0-1.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 179,
                    "num_queries": 474,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDREuropeanaItScansRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDREuropeanaItScansRetrieval",
        dataset={
            "path": "jinaai/europeana-it-scans_beir",
            "revision": "main",
        },
        eval_langs=["ita-Latn"],
        domains=["News"],
        license="cc0-1.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 265,
                    "num_queries": 618,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDREuropeanaNlLegalRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDREuropeanaNlLegalRetrieval",
        dataset={
            "path": "jinaai/europeana-nl-legal_beir",
            "revision": "main",
        },
        eval_langs=["nld-Latn"],
        domains=["Legal"],
        license="cc0-1.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 244,
                    "num_queries": 198,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRHindiGovVQARetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRHindiGovVQARetrieval",
        dataset={
            "path": "jinaai/hindi-gov-vqa_beir",
            "revision": "main",
        },
        eval_langs=["hin-Deva"],
        domains=["Government"],
        license="not specified",
        annotations_creators="LM-generated",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 337,
                    "num_queries": 454,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRAutomobileCatelogRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRAutomobileCatelogRetrieval",
        dataset={
            "path": "jinaai/automobile_catalogue_jp_beir",
            "revision": "main",
        },
        eval_langs=["jpn-Jpan"],
        domains=["Engineering", "Web"],
        license="not specified",
        annotations_creators="LM-generated",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 15,
                    "num_queries": 45,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRBeveragesCatalogueRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRBeveragesCatalogueRetrieval",
        dataset={
            "path": "jinaai/beverages_catalogue_ru_beir",
            "revision": "main",
        },
        eval_langs=["rus-Cyrl"],
        domains=["Web"],
        license="not specified",
        annotations_creators="LM-generated",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 34,
                    "num_queries": 100,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRRamensBenchmarkRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRRamensBenchmarkRetrieval",
        dataset={
            "path": "jinaai/ramen_benchmark_jp_beir",
            "revision": "main",
        },
        eval_langs=["jpn-Jpan"],
        domains=["Web"],
        license="not specified",
        annotations_creators="LM-generated",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 10,
                    "num_queries": 29,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRJDocQARetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRJDocQARetrieval",
        dataset={
            "path": "jinaai/jdocqa_beir",
            "revision": "main",
        },
        eval_langs=["jpn-Jpan"],
        domains=["Web"],
        license="cc-by-4.0",
        jannotations_creators="LM-generated",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 758,
                    "num_queries": 744,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRHungarianDocQARetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRHungarianDocQARetrieval",
        dataset={
            "path": "jinaai/hungarian_doc_qa_beir",
            "revision": "main",
        },
        eval_langs=["hun-Latn"],
        domains=["Web"],
        license="cc-by-4.0",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 51,
                    "num_queries": 54,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRArabicChartQARetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRArabicChartQARetrieval",
        dataset={
            "path": "jinaai/arabic_chartqa_ar_beir",
            "revision": "main",
        },
        eval_langs=["ara-Arab"],
        domains=["Academic"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 342,
                    "num_queries": 745,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRArabicInfographicsVQARetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRArabicInfographicsVQARetrieval",
        dataset={
            "path": "jinaai/arabic_infographicsvqa_ar_beir",
            "revision": "main",
        },
        eval_langs=["ara-Arab"],
        domains=["Academic"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 40,
                    "num_queries": 120,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDROWIDChartsRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDROWIDChartsRetrieval",
        dataset={
            "path": "jinaai/owid_charts_en_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="cc-by-4.0",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 937,
                    "num_queries": 131,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRMPMQARetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRMPMQARetrieval",
        dataset={
            "path": "jinaai/mpmqa_small_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 782,
                    "num_queries": 154,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRJina2024YearlyBookRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRJina2024YearlyBookRetrieval",
        dataset={
            "path": "jinaai/jina_2024_yearly_book_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="apache-2.0",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 33,
                    "num_queries": 75,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRWikimediaCommonsMapsRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRWikimediaCommonsMapsRetrieval",
        dataset={
            "path": "jinaai/wikimedia-commons-maps_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="cc0-1.0",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 451,
                    "num_queries": 443,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRPlotQARetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRPlotQARetrieval",
        dataset={
            "path": "jinaai/plotqa_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="cc-by-4.0",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 986,
                    "num_queries": 610,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRMMTabRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRMMTabRetrieval",
        dataset={
            "path": "jinaai/MMTab_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 906,
                    "num_queries": 987,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRCharXivOCRRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRCharXivOCRRetrieval",
        dataset={
            "path": "jinaai/CharXiv-en_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 1000,
                    "num_queries": 999,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRStudentEnrollmentSyntheticRetrieval(
    MultilingualTask, AbsTaskAny2AnyRetrieval
):

    metadata = TaskMetadata(
        name="JinaVDRStudentEnrollmentSyntheticRetrieval",
        dataset={
            "path": "jinaai/student-enrollment_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Academic"],
        license="cc0-1.0",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 489,
                    "num_queries": 1000,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRGitHubReadmeRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRGitHubReadmeRetrieval",
        dataset={
            "path": "jinaai/github-readme-retrieval-multilingual_beir",
            "revision": "main",
        },
        eval_langs=get_langs(
            [
                "ar",
                "bn",
                "de",
                "en",
                "es",
                "fr",
                "hi",
                "id",
                "it",
                "jp",
                "ko",
                "nl",
                "pt",
                "ru",
                "th",
                "vi",
                "zh",
            ]
        ),
        domains=["Web"],
        license="multiple",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 16998,
                    "num_queries": 16953,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRTweetStockSyntheticsRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRTweetStockSyntheticsRetrieval",
        dataset={
            "path": "jinaai/tweet-stock-synthetic-retrieval_beir",
            "revision": "main",
        },
        eval_langs=get_langs(
            ["ar", "de", "en", "es", "fr", "hi", "hu", "jp", "ru", "zh"]
        ),
        domains=["Social"],
        license="not specified",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 10000,
                    "num_queries": 6277,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRAirbnbSyntheticRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRAirbnbSyntheticRetrieval",
        dataset={
            "path": "jinaai/airbnb-synthetic-retrieval_beir",
            "revision": "main",
        },
        eval_langs=get_langs(
            ["ar", "de", "en", "es", "fr", "hi", "hu", "jp", "ru", "zh"]
        ),
        domains=["Web"],
        license="cc0-1.0",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 10000,
                    "num_queries": 4952,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRShanghaiMasterPlanRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRShanghaiMasterPlanRetrieval",
        dataset={
            "path": "jinaai/shanghai_master_plan_beir",
            "revision": "main",
        },
        eval_langs=["zho-Hans"],
        domains=["Web"],
        license="not specified",
        annotations_creators="human-annotated",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 23,
                    "num_queries": 57,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRWikimediaCommonsDocumentsRetrieval(
    MultilingualTask, AbsTaskAny2AnyRetrieval
):

    metadata = TaskMetadata(
        name="JinaVDRWikimediaCommonsDocumentsRetrieval",
        dataset={
            "path": "jinaai/wikimedia-commons-documents-ml_beir",
            "revision": "main",
        },
        eval_langs=get_langs(
            [
                "ar",
                "bn",
                "de",
                "en",
                "es",
                "fr",
                "hi",
                "hu",
                "id",
                "it",
                "jp",
                "ko",
                "my",
                "nl",
                "pt",
                "ru",
                "th",
                "ur",
                "vi",
                "zh",
            ]
        ),
        domains=["Web"],
        license="multiple",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 15217,
                    "num_queries": 14060,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDREuropeanaFrNewsRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDREuropeanaFrNewsRetrieval",
        dataset={
            "path": "jinaai/europeana-fr-news_beir",
            "revision": "main",
        },
        eval_langs=["fra-Latn"],
        domains=["News"],
        license="cc0-1.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 145,
                    "num_queries": 237,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRDocQAHealthcareIndustryRetrieval(
    MultilingualTask, AbsTaskAny2AnyRetrieval
):

    metadata = TaskMetadata(
        name="JinaVDRDocQAHealthcareIndustryRetrieval",
        dataset={
            "path": "jinaai/docqa_healthcare_industry_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Medical"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 961,
                    "num_queries": 89,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRDocQAAI(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRDocQAAI",
        dataset={
            "path": "jinaai/docqa_artificial_intelligence_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 962,
                    "num_queries": 69,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRShiftProjectRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRShiftProjectRetrieval",
        dataset={
            "path": "jinaai/shiftproject_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 998,
                    "num_queries": 88,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRTatQARetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRTatQARetrieval",
        dataset={
            "path": "jinaai/tatqa_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 270,
                    "num_queries": 120,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRInfovqaRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRInfovqaRetrieval",
        dataset={
            "path": "jinaai/infovqa_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 500,
                    "num_queries": 362,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRDocVQARetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRDocVQARetrieval",
        dataset={
            "path": "jinaai/docvqa_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 499,
                    "num_queries": 38,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRDocQAGovReportRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRDocQAGovReportRetrieval",
        dataset={
            "path": "jinaai/docqa_gov_report_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Government"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 970,
                    "num_queries": 76,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRTabFQuadRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRTabFQuadRetrieval",
        dataset={
            "path": "jinaai/tabfquad_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Academic"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 70,
                    "num_queries": 125,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRDocQAEnergyRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRDocQAEnergyRetrieval",
        dataset={
            "path": "jinaai/docqa_energy_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 971,
                    "num_queries": 68,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRArxivQARetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):

    metadata = TaskMetadata(
        name="JinaVDRArxivQARetrieval",
        dataset={
            "path": "jinaai/arxivqa_beir",
            "revision": "main",
        },
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 499,
                    "num_queries": 29,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
        **COMMON_METADATA,
    )

    load_data = load_data
