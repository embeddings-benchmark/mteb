from collections import defaultdict

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

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
    "type": "DocumentUnderstanding",
    "category": "t2i",
    "eval_splits": ["test"],
    "main_score": "ndcg_at_5",
    "task_subtypes": ["Image Text Retrieval"],
    "dialect": [],
    "modalities": ["text", "image"],
    "bibtex_citation": r"""@misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
  archiveprefix = {arXiv},
  author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
  eprint = {2506.18902},
  primaryclass = {cs.AI},
  title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
  url = {https://arxiv.org/abs/2506.18902},
  year = {2025},
}""",
    "prompt": {"query": "Find a screenshot that is relevant to the user's input."},
}


def _load_single_language(
    path: str,
    split: str,
    lang: str | None = None,
    revision: str | None = None,
):
    query_ds = load_dataset(
        path,
        data_dir=f"{lang}/queries" if lang else "queries",
        split=split,
        revision=revision,
    )
    query_ds = query_ds.map(
        lambda x: {
            "id": f"query-{split}-{x['query-id']}",
            "text": x["query"],
            "modality": "text",
        },
        remove_columns=["query-id", "query"],
    )

    corpus_ds = load_dataset(
        path,
        data_dir=f"{lang}/corpus" if lang else "corpus",
        split=split,
        revision=revision,
    )
    corpus_ds = corpus_ds.map(
        lambda x: {
            "id": f"corpus-{split}-{x['corpus-id']}",
            "modality": "image",
        },
        remove_columns=["corpus-id"],
    )

    qrels_ds = load_dataset(
        path,
        data_dir=f"{lang}/qrels" if lang else "qrels",
        split=split,
        revision=revision,
    )

    return query_ds, corpus_ds, qrels_ds


def _load_data(
    path: str,
    splits: str,
    langs: list | None = None,
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


def load_data(self) -> None:
    if self.data_loaded:
        return

    self.corpus, self.queries, self.relevant_docs = _load_data(
        path=self.metadata.dataset["path"],
        splits=self.metadata.eval_splits,
        langs=self.metadata.eval_langs,
        revision=self.metadata.dataset["revision"],
    )

    self.data_loaded = True


class JinaVDRMedicalPrescriptionsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRMedicalPrescriptionsRetrieval",
        description="Retrieve medical prescriptions based on templated queries. Source dataset https://huggingface.co/datasets/Technoculture/medical-prescriptions",
        reference="https://huggingface.co/datasets/jinaai/medical-prescriptions_beir",
        dataset={
            "path": "jinaai/medical-prescriptions_beir",
            "revision": "f27559d1602523e1c6b66c83e68d337f7bb74fe2",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Medical"],
        license="not specified",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRStanfordSlideRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRStanfordSlideRetrieval",
        description="Retrieve scientific and engineering slides based on human annotated queries. Source dataset https://exhibits.stanford.edu/data/catalog/mv327tb8364",
        reference="https://huggingface.co/datasets/jinaai/stanford_slide_beir",
        dataset={
            "path": "jinaai/stanford_slide_beir",
            "revision": "6444c24c59dfb271bdc01e0a56292753e196fc98",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Academic"],
        license="not specified",
        annotations_creators="human-annotated",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRDonutVQAISynHMPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRDonutVQAISynHMPRetrieval",
        description="Retrieve medical records based on templated queries. Source dataset https://huggingface.co/datasets/warshakhan/donut_vqa_ISynHMP",
        reference="https://huggingface.co/datasets/jinaai/donut_vqa_beir",
        dataset={
            "path": "jinaai/donut_vqa_beir",
            "revision": "38e38a676202d3d8fd365b152ab7832207a7aa35",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Medical"],
        license="not specified",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRTableVQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRTableVQARetrieval",
        description="Retrieve scientific tables based on LLM generated queries. Source datasets https://huggingface.co/datasets/HuggingFaceM4/ChartQA or https://huggingface.co/datasets/cmarkea/aftdb",
        reference="https://huggingface.co/datasets/jinaai/table-vqa_beir",
        dataset={
            "path": "jinaai/table-vqa_beir",
            "revision": "d60d6d1311296fac106b5c399873539d3d155393",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Academic"],
        license="not specified",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRChartQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRChartQARetrieval",
        description="Retrieve charts based on LLM generated queries. Source datasets https://huggingface.co/datasets/HuggingFaceM4/ChartQA",
        reference="https://huggingface.co/datasets/jinaai/ChartQA_beir",
        dataset={
            "path": "jinaai/ChartQA_beir",
            "revision": "9d9f9fa99f1150b5af04348de90799a24138d46c",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="not specified",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRTQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRTQARetrieval",
        description="Retrieve textbook pages (images and text) based on LLM generated queries from the text. Source datasets https://prior.allenai.org/projects/tqa",
        reference="https://huggingface.co/datasets/jinaai/tqa_beir",
        dataset={
            "path": "jinaai/tqa_beir",
            "revision": "33b48ad357ceffac3488630b6b0f2c86a9386978",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Academic"],
        license="cc-by-nc-3.0",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDROpenAINewsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDROpenAINewsRetrieval",
        description="Retrieve news articles from the OpenAI news website based on human annotated queries. News taken from https://openai.com/news/",
        reference="https://huggingface.co/datasets/jinaai/openai-news_beir",
        dataset={
            "path": "jinaai/openai-news_beir",
            "revision": "2c2d1f9910abe9093aa6fa82a76ab73dca525cfd",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["News", "Web"],
        license="not specified",
        annotations_creators="human-annotated",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDREuropeanaDeNewsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDREuropeanaDeNewsRetrieval",
        description="Retrieve German news articles based on LLM generated queries. This dataset was created from records of the [Europeana](https://europeana.eu/) online collection by selecting scans of German news articles",
        reference="https://huggingface.co/datasets/jinaai/europeana-de-news_beir",
        dataset={
            "path": "jinaai/europeana-de-news_beir",
            "revision": "bf226830eac4d22a2389cdccafd254bf1bc1bc5f",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["deu-Latn"],
        domains=["News"],
        license="cc0-1.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDREuropeanaEsNewsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDREuropeanaEsNewsRetrieval",
        description="Retrieve Spanish news articles based on LLM generated queries. This dataset was created from records of the [Europeana](https://europeana.eu/) online collection by selecting scans of Spanish news articles",
        reference="https://huggingface.co/datasets/jinaai/europeana-es-news_beir",
        dataset={
            "path": "jinaai/europeana-es-news_beir",
            "revision": "724aa71a59e6870eccf3d046e08145c61d0620cb",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["spa-Latn"],
        domains=["News"],
        license="cc0-1.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDREuropeanaItScansRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDREuropeanaItScansRetrieval",
        description="Retrieve Italian historical articles based on LLM generated queries. This dataset was created from records of the [Europeana](https://europeana.eu/) online collection by selecting scans of Italian news articles",
        reference="https://huggingface.co/datasets/jinaai/europeana-it-scans_beir",
        dataset={
            "path": "jinaai/europeana-it-scans_beir",
            "revision": "8907ccacaa9c624218a2153598e57e444c76391e",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["ita-Latn"],
        domains=["News"],
        license="cc0-1.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDREuropeanaNlLegalRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDREuropeanaNlLegalRetrieval",
        description="Retrieve Dutch historical legal documents based on LLM generated queries.  This dataset was created from records of the [Europeana](https://europeana.eu/) online collection by selecting scans of Dutch news articles",
        reference="https://huggingface.co/datasets/jinaai/europeana-nl-legal_beir",
        dataset={
            "path": "jinaai/europeana-nl-legal_beir",
            "revision": "f71c665cc5d4d24ee6045717598d1480c5d63bbc",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["nld-Latn"],
        domains=["Legal"],
        license="cc0-1.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRHindiGovVQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRHindiGovVQARetrieval",
        description="Retrieve Hindi government documents based on LLM generated queries.",
        reference="https://huggingface.co/datasets/jinaai/hindi-gov-vqa_beir",
        dataset={
            "path": "jinaai/hindi-gov-vqa_beir",
            "revision": "a1b96978b1ad0c217a62600e0713ce40ea583cde",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["hin-Deva"],
        domains=["Government"],
        license="not specified",
        annotations_creators="LM-generated",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRAutomobileCatelogRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRAutomobileCatelogRetrieval",
        description="Retrieve automobile marketing documents based on LLM generated queries. Marketing document from Toyota Japanese website featuring [RAV4](https://toyota.jp/pages/contents/request/webcatalog/rav4/rav4_special1_202310.pdf) and [Corolla](https://toyota.jp/pages/contents/request/webcatalog/corolla/corolla_special1_202407.pdf). The `text_description` column contains OCR text extracted from the images using EasyOCR.",
        reference="https://huggingface.co/datasets/jinaai/automobile_catalogue_jp_beir",
        dataset={
            "path": "jinaai/automobile_catalogue_jp_beir",
            "revision": "b83ca039723e1c705dbb444147b1fa0cc6358d5f",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["jpn-Jpan"],
        domains=["Engineering", "Web"],
        license="not specified",
        annotations_creators="LM-generated",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRBeveragesCatalogueRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRBeveragesCatalogueRetrieval",
        description="Retrieve beverages marketing documents based on LLM generated queries. This dataset was self-curated by searching beverage catalogs on Google search and downloading PDFs.",
        reference="https://huggingface.co/datasets/jinaai/beverages_catalogue_ru_beir",
        dataset={
            "path": "jinaai/beverages_catalogue_ru_beir",
            "revision": "d1be95f14c1f8eedb0165303943cd5b69402e2b4",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["rus-Cyrl"],
        domains=["Web"],
        license="not specified",
        annotations_creators="LM-generated",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRRamensBenchmarkRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRRamensBenchmarkRetrieval",
        description="Retrieve ramen restaurant marketing documents based on LLM generated queries. Marketing document from Ramen [restaurants](https://www.city.niigata.lg.jp/kanko/kanko/oshirase/ramen.files/guidebook.pdf).",
        reference="https://huggingface.co/datasets/jinaai/ramen_benchmark_jp_beir",
        dataset={
            "path": "jinaai/ramen_benchmark_jp_beir",
            "revision": "ed0ca84e0d2441f9af2b6617ebcdbeefe8a65c1b",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["jpn-Jpan"],
        domains=["Web"],
        license="not specified",
        annotations_creators="LM-generated",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRJDocQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRJDocQARetrieval",
        description="Retrieve Japanese documents in various formats based on human annotated queries. Document Question answering from [JDocQAJP dataset](https://huggingface.co/datasets/jlli/JDocQA-nonbinary), test split.",
        reference="https://huggingface.co/datasets/jinaai/jdocqa_beir",
        dataset={
            "path": "jinaai/jdocqa_beir",
            "revision": "40a4c729550dfb560c479348775bcff99b6be91b",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["jpn-Jpan"],
        domains=["Web"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRHungarianDocQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRHungarianDocQARetrieval",
        description="Retrieve Hungarian documents in various formats based on human annotated queries. Document Question answering from [Hungurian doc qa dataset](https://huggingface.co/datasets/jlli/HungarianDocQA-OCR), test split.",
        reference="https://huggingface.co/datasets/jinaai/hungarian_doc_qa_beir",
        dataset={
            "path": "jinaai/hungarian_doc_qa_beir",
            "revision": "4179a258d99ed8e9cd1fdca76a74484e842412f5",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["hun-Latn"],
        domains=["Web"],
        license="cc-by-4.0",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRArabicChartQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRArabicChartQARetrieval",
        description="Retrieve Arabic charts based on queries. This dataset is derived from the [Arabic ChartQA dataset](https://huggingface.co/datasets/ahmedheakl/arabic_chartqa), reformatting the train split as a test split with modified field names such that it is compatible with the ViDoRe evaluation benchmark.",
        reference="https://huggingface.co/datasets/jinaai/arabic_chartqa_ar_beir",
        dataset={
            "path": "jinaai/arabic_chartqa_ar_beir",
            "revision": "13a71ebb8e17fd7d7303a41831ac0092b61ef7c1",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["ara-Arab"],
        domains=["Academic"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRArabicInfographicsVQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRArabicInfographicsVQARetrieval",
        description="Retrieve Arabic infographics based on queries. This dataset is derived from the [Arabic Infographics VQA dataset](https://huggingface.co/datasets/ahmedheakl/arabic_infographicsvqa), reformatting the train split as a test split with modified field names so it can be used in the ViDoRe evaluation benchmark.",
        reference="https://huggingface.co/datasets/jinaai/arabic_infographicsvqa_ar_beir",
        dataset={
            "path": "jinaai/arabic_infographicsvqa_ar_beir",
            "revision": "a78b0caf95636de35bb147db616181c8d3e5b9d3",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["ara-Arab"],
        domains=["Academic"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDROWIDChartsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDROWIDChartsRetrieval",
        description="Retrieve charts from the OWID dataset based on accompanied text snippets. We sampled a set of ~5k charts and articles from [Our World In Data](https://ourworldindata.org) to produce this evaluation set. This particular dataset is a subsample of 1000 random charts from the full dataset which can be found [here](https://huggingface.co/datasets/jjinaai/owid_charts).",
        reference="https://huggingface.co/datasets/jinaai/owid_charts_en_beir",
        dataset={
            "path": "jinaai/owid_charts_en_beir",
            "revision": "cac5a7f322b9baa473bb878ff6dbdda8a52840e9",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="cc-by-4.0",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRMPMQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRMPMQARetrieval",
        description="Retrieve product manuals based on human annotated queries. 155 questions and 782 document images cleaned from [jinaai/MPMQA](https://huggingface.co/datasets/jinaai/MPMQA), test set.",  # MPMQA not exists on HF
        reference="https://huggingface.co/datasets/jinaai/mpmqa_small_beir",
        dataset={
            "path": "jinaai/mpmqa_small_beir",
            "revision": "83deed2d9d7e16cb87aef80a419be16733cc954a",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRJina2024YearlyBookRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRJina2024YearlyBookRetrieval",
        description="Retrieve pages from the 2024 Jina yearbook based on human annotated questions. 75 human annotated questions created from digital version of Jina AI yearly book 2024, 166 pages in total. ",
        reference="https://huggingface.co/datasets/jinaai/jina_2024_yearly_book_beir",
        dataset={
            "path": "jinaai/jina_2024_yearly_book_beir",
            "revision": "79cd892d672b0b0f25229a0b57ba893ee6ac69c1",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="apache-2.0",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRWikimediaCommonsMapsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRWikimediaCommonsMapsRetrieval",
        description="Retrieve maps from Wikimedia Commons based on their description. It contains images of (mostly historic) maps which should be identified based on their description. We extracted those descriptions from [Wikimedia Commons](https://commons.wikimedia.org/). We have included the license type and a link (license_text) to the original Wikimedia Commons page for each extracted image.",
        reference="https://huggingface.co/datasets/jinaai/wikimedia-commons-maps_beir",
        dataset={
            "path": "jinaai/wikimedia-commons-maps_beir",
            "revision": "735c932678642e90909126f4d0948cc5fe1f406e",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="multiple",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRPlotQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRPlotQARetrieval",
        description="Retrieve plots from the PlotQA dataset based on LLM generated queries. Questions subsampled from [PlotQA](https://github.com/NiteshMethani/PlotQA) test set. It is following a subsample + LLM-based classification process, using LLM to verify the question quality, e.g. queries like `How many different coloured dotlines are there` will be filtered out.",
        reference="https://huggingface.co/datasets/jinaai/plotqa_beir",
        dataset={
            "path": "jinaai/plotqa_beir",
            "revision": "64a321b8bbba18ebe04a9099f4c3485e1c78b583",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="cc-by-4.0",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRMMTabRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRMMTabRetrieval",
        description="Retrieve tables from the MMTab dataset based on queries. This dataset is a copy of the original test split from MMTab, taking only items where an 'original_query' is present, and removing the 'input' and 'output' columns, as they are unnecessary for retrieval tasks.",
        reference="https://huggingface.co/datasets/jinaai/MMTab_beir",
        dataset={
            "path": "jinaai/MMTab_beir",
            "revision": "59e6a04a93a0eb082e2402717bb768d4b11795c7",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRCharXivOCRRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRCharXivOCRRetrieval",
        description="Retrieve charts from scientific papers based on human annotated queries. This dataset is derived from the [CharXiv dataset](https://huggingface.co/datasets/princeton-nlp/CharXiv), reformatting the test split with modified field names, so that it can be used in the ViDoRe benchmark.",
        reference="https://huggingface.co/datasets/jinaai/CharXiv-en_beir",
        dataset={
            "path": "jinaai/CharXiv-en_beir",
            "revision": "c38db7d063ee7d0c119eb41932e981943e37f702",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRStudentEnrollmentSyntheticRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRStudentEnrollmentSyntheticRetrieval",
        description="Retrieve student enrollment data based on templated queries. This dataset is created from the original Kaggle [Delaware Student Enrollment](https://www.kaggle.com/datasets/noeyislearning/delaware-student-enrollment) dataset. The charts are rendered and queries created using templates.",
        reference="https://huggingface.co/datasets/jinaai/student-enrollment_beir",
        dataset={
            "path": "jinaai/student-enrollment_beir",
            "revision": "80859af7fc43313b5e6e7bb1087b5c922f030ce1",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Academic"],
        license="cc0-1.0",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRGitHubReadmeRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRGitHubReadmeRetrieval",
        description=(
            "Retrieve GitHub readme files based their description. "
            "This dataset consists of rendered GitHub readmes in a variety of different languages, together with their accompanying descriptions as queries and their license in the `license_type` and `license_text` columns. "
            "This particular dataset is a subsample of 1000 random rows per language from the full dataset which can be found [here](https://huggingface.co/datasets/jinaai/github-readme-retrieval-ml-filtered)."
        ),
        reference="https://huggingface.co/datasets/jinaai/github-readme-retrieval-multilingual_beir",
        dataset={
            "path": "jinaai/github-readme-retrieval-multilingual_beir",
            "revision": "a7b17c2eca814c32b9af6a852a5d6d7b5e6b9165",
        },
        date=("2024-10-01", "2025-04-01"),
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
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRTweetStockSyntheticsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRTweetStockSyntheticsRetrieval",
        description="Retrieve rendered tables of stock prices based on templated queries. This dataset is created from the original Kaggle [Tweet Sentiment's Impact on Stock Returns](https://www.kaggle.com/datasets/thedevastator/tweet-sentiment-s-impact-on-stock-returns) dataset.",
        reference="https://huggingface.co/datasets/jinaai/tweet-stock-synthetic-retrieval_beir",
        dataset={
            "path": "jinaai/tweet-stock-synthetic-retrieval_beir",
            "revision": "955f2c8e171b3d9ff18c8b841cd814649209d4b0",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=get_langs(
            ["ar", "de", "en", "es", "fr", "hi", "hu", "jp", "ru", "zh"]
        ),
        domains=["Social"],
        license="not specified",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRAirbnbSyntheticRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRAirbnbSyntheticRetrieval",
        description="Retrieve rendered tables from Airbnb listings based on templated queries. This dataset is created from the original Kaggle [New York City Airbnb Open Data dataset](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data).",
        reference="https://huggingface.co/datasets/jinaai/airbnb-synthetic-retrieval_beir",
        dataset={
            "path": "jinaai/airbnb-synthetic-retrieval_beir",
            "revision": "14c4c816fff158d20719bebf414d495efeaedc20",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=get_langs(
            ["ar", "de", "en", "es", "fr", "hi", "hu", "jp", "ru", "zh"]
        ),
        domains=["Web"],
        license="cc0-1.0",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRShanghaiMasterPlanRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRShanghaiMasterPlanRetrieval",
        description="Retrieve pages from the Shanghai Master Plan based on human annotated queries. The master plan document is taken from [here](https://www.shanghai.gov.cn/newshanghai/xxgkfj/2035004.pdf).",
        reference="https://huggingface.co/datasets/jinaai/shanghai_master_plan_beir",
        dataset={
            "path": "jinaai/shanghai_master_plan_beir",
            "revision": "ba711c07aafbe43ef7970cf9429109fc6220c824",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["zho-Hans"],
        domains=["Web"],
        license="not specified",
        annotations_creators="human-annotated",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRWikimediaCommonsDocumentsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRWikimediaCommonsDocumentsRetrieval",
        description="Retrieve historical documents from Wikimedia Commons based on their description. Wikimedia Commons Documents. It contains images of (mostly historic) documents which should be identified based on their description. We extracted those descriptions from Wikimedia Commons. We have included the license type and a link (`license_text`) to the original Wikimedia Commons page for each extracted image.",
        reference="https://huggingface.co/datasets/jinaai/wikimedia-commons-documents-ml_beir",
        dataset={
            "path": "jinaai/wikimedia-commons-documents-ml_beir",
            "revision": "1307839f4deabc1dfa954ef6843ef4cf4fc038b8",
        },
        date=("2024-10-01", "2025-04-01"),
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
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDREuropeanaFrNewsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDREuropeanaFrNewsRetrieval",
        description="Retrieve French news articles from Europeana based on LLM generated queries. This dataset was created from records of the [Europeana online collection](https://europeana.eu) by selecting scans of French news articles.",
        reference="https://huggingface.co/datasets/jinaai/europeana-fr-news_beir",
        dataset={
            "path": "jinaai/europeana-fr-news_beir",
            "revision": "3abc89102ab1d64d02806ba612e7286d63624c01",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["fra-Latn"],
        domains=["News"],
        license="cc0-1.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRDocQAHealthcareIndustryRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRDocQAHealthcareIndustryRetrieval",
        description="Retrieve healthcare industry documents based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d). For more information regarding the filtering please read [our paper](https://arxiv.org/abs/2506.18902) or [this discussion on github](https://github.com/embeddings-benchmark/mteb/pull/2942#discussion_r2240711654).",
        reference="https://huggingface.co/datasets/jinaai/docqa_healthcare_industry_beir",
        dataset={
            "path": "jinaai/docqa_healthcare_industry_beir",
            "revision": "810989fee9624ef58b3522c20e00c55d9fc69002",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Medical"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        adapted_from=["VidoreDocVQARetrieval"],
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRDocQAAI(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRDocQAAI",
        description="Retrieve AI documents based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).",
        reference="https://huggingface.co/datasets/jinaai/docqa_artificial_intelligence_beir",
        dataset={
            "path": "jinaai/docqa_artificial_intelligence_beir",
            "revision": "9764d3c6b9b946b2b6302719e4a89bc99c83f975",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        adapted_from=["VidoreDocVQARetrieval"],
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRShiftProjectRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRShiftProjectRetrieval",
        description="Retrieve documents with graphs from the Shift Project based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).",
        reference="https://huggingface.co/datasets/jinaai/shiftproject_beir",
        dataset={
            "path": "jinaai/shiftproject_beir",
            "revision": "c97b12a93e714c7c3eebea80888ab83483803028",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        adapted_from=["VidoreShiftProjectRetrieval"],
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRTatQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRTatQARetrieval",
        description="Retrieve financial reports based on human annotated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).",
        reference="https://huggingface.co/datasets/jinaai/tatqa_beir",
        dataset={
            "path": "jinaai/tatqa_beir",
            "revision": "78b7f06bd45d8cead8a61ec83ed20d7eb3c0f82a",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        adapted_from=["VidoreTatdqaRetrieval"],
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRInfovqaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRInfovqaRetrieval",
        description="Retrieve infographics based on human annotated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).",
        reference="https://huggingface.co/datasets/jinaai/infovqa_beir",
        dataset={
            "path": "jinaai/infovqa_beir",
            "revision": "682247a4c07b5f9da329b2e29fb57c87efd26a3f",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        adapted_from=["VidoreInfoVQARetrieval"],
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRDocVQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRDocVQARetrieval",
        description="Retrieve industry documents based on human annotated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).",
        reference="https://huggingface.co/datasets/jinaai/docvqa_beir",
        dataset={
            "path": "jinaai/docvqa_beir",
            "revision": "d77d5d00a0047597a0ffc1ed25555078710e21b4",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        adapted_from=["VidoreDocVQARetrieval"],
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRDocQAGovReportRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRDocQAGovReportRetrieval",
        description="Retrieve government reports based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).",
        reference="https://huggingface.co/datasets/jinaai/docqa_gov_report_beir",
        dataset={
            "path": "jinaai/docqa_gov_report_beir",
            "revision": "76fd0c09bff018c2d503d4f50f0d3ddb68690af0",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Government"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        adapted_from=["VidoreDocVQARetrieval"],
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRTabFQuadRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRTabFQuadRetrieval",
        description="Retrieve tables from industry documents based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).",
        reference="https://huggingface.co/datasets/jinaai/tabfquad_beir",
        dataset={
            "path": "jinaai/tabfquad_beir",
            "revision": "42f9ba0b1f1dd0a6b82be6e9547367e2fb555e21",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Academic"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        adapted_from=["VidoreTabfquadRetrieval"],
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRDocQAEnergyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRDocQAEnergyRetrieval",
        description="Retrieve energy industry documents based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).",
        reference="https://huggingface.co/datasets/jinaai/docqa_energy_beir",
        dataset={
            "path": "jinaai/docqa_energy_beir",
            "revision": "25dac6859e7f5b7e0c309b6286534794b7d05a6c",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        adapted_from=["VidoreDocVQARetrieval"],
        **COMMON_METADATA,
    )

    load_data = load_data


class JinaVDRArxivQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRArxivQARetrieval",
        description="Retrieve figures from scientific papers from arXiv based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).",
        reference="https://huggingface.co/datasets/jinaai/arxivqa_beir",
        dataset={
            "path": "jinaai/arxivqa_beir",
            "revision": "d49798d601d4c53a1d15054acecf25f629f504f4",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        adapted_from=["VidoreArxivQARetrieval"],
        **COMMON_METADATA,
    )

    load_data = load_data
