from __future__ import annotations

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
    "bibtex_citation": r"""@inproceedings{günther2025jinaembeddingsv4universalembeddingsmultimodal,
  address = {Suzhuo, China},
  author = {G{\"u}nther, Michael and Sturua, Saba and Akram, Mohammad Kalim and Mohr, Isabelle and Ungureanu, Andrei and Wang, Bo and Eslami, Sedigheh and Martens, Scott and Werk, Maximilian and Wang, Nan and Xiao, Han},
  booktitle = {Proceedings of the 5th Workshop on Multilingual Representation Learning ({MRL} 2025)},
  doi = {10.18653/v1/2025.mrl-main.36},
  editor = {Adelani, David Ifeoluwa and Arnett, Catherine and Ataman, Duygu and Chang, Tyler A. and Gonen, Hila and Raja, Rahul and Schmidt, Fabian and Stap, David and Wang, Jiayi},
  isbn = {979-8-89176-345-6},
  month = {nov},
  pages = {531--550},
  publisher = {Association for Computational Linguistics},
  title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
  url = {https://aclanthology.org/2025.mrl-main.36/},
  year = {2025},
}""",
    "prompt": {"query": "Find a screenshot that is relevant to the user's input."},
}


class JinaVDRMedicalPrescriptionsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRMedicalPrescriptionsRetrieval",
        description="Retrieve medical prescriptions based on templated queries. Source dataset https://huggingface.co/datasets/Technoculture/medical-prescriptions",
        reference="https://huggingface.co/datasets/jinaai/medical-prescriptions_beir",
        dataset={
            "path": "mteb/JinaVDRMedicalPrescriptionsRetrieval",
            "revision": "29e840b69e957c993be41c892c7e11ed838403ea",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Medical"],
        license="not specified",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRStanfordSlideRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRStanfordSlideRetrieval",
        description="Retrieve scientific and engineering slides based on human annotated queries. Source dataset https://exhibits.stanford.edu/data/catalog/mv327tb8364",
        reference="https://huggingface.co/datasets/jinaai/stanford_slide_beir",
        dataset={
            "path": "mteb/JinaVDRStanfordSlideRetrieval",
            "revision": "b91ee039ab4b02e0fbcc3a80fd325a8af3bc0268",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Academic"],
        license="not specified",
        annotations_creators="human-annotated",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRDonutVQAISynHMPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRDonutVQAISynHMPRetrieval",
        description="Retrieve medical records based on templated queries. Source dataset https://huggingface.co/datasets/warshakhan/donut_vqa_ISynHMP",
        reference="https://huggingface.co/datasets/jinaai/donut_vqa_beir",
        dataset={
            "path": "mteb/JinaVDRDonutVQAISynHMPRetrieval",
            "revision": "90df24efdb052bb9c0e285f1e0e36b7c3d938a93",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Medical"],
        license="not specified",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRTableVQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRTableVQARetrieval",
        description="Retrieve scientific tables based on LLM generated queries. Source datasets https://huggingface.co/datasets/HuggingFaceM4/ChartQA or https://huggingface.co/datasets/cmarkea/aftdb",
        reference="https://huggingface.co/datasets/jinaai/table-vqa_beir",
        dataset={
            "path": "mteb/JinaVDRTableVQARetrieval",
            "revision": "83ec0f91b9b776f9505fd8f0a8a197fe7a95638d",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Academic"],
        license="not specified",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRChartQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRChartQARetrieval",
        description="Retrieve charts based on LLM generated queries. Source datasets https://huggingface.co/datasets/HuggingFaceM4/ChartQA",
        reference="https://huggingface.co/datasets/jinaai/ChartQA_beir",
        dataset={
            "path": "mteb/JinaVDRChartQARetrieval",
            "revision": "fb32bbc9494228d05e047e4340b7986ca3df0472",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="not specified",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRTQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRTQARetrieval",
        description="Retrieve textbook pages (images and text) based on LLM generated queries from the text. Source datasets https://prior.allenai.org/projects/tqa",
        reference="https://huggingface.co/datasets/jinaai/tqa_beir",
        dataset={
            "path": "mteb/JinaVDRTQARetrieval",
            "revision": "69a507868484c3b35c73f499fafd2f5d5e49a18c",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Academic"],
        license="cc-by-nc-3.0",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDROpenAINewsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDROpenAINewsRetrieval",
        description="Retrieve news articles from the OpenAI news website based on human annotated queries. News taken from https://openai.com/news/",
        reference="https://huggingface.co/datasets/jinaai/openai-news_beir",
        dataset={
            "path": "mteb/JinaVDROpenAINewsRetrieval",
            "revision": "8630517a15a458ecdd1512ff0f62d641bbeb7465",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["News", "Web"],
        license="not specified",
        annotations_creators="human-annotated",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDREuropeanaDeNewsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDREuropeanaDeNewsRetrieval",
        description="Retrieve German news articles based on LLM generated queries. This dataset was created from records of the [Europeana](https://europeana.eu/) online collection by selecting scans of German news articles",
        reference="https://huggingface.co/datasets/jinaai/europeana-de-news_beir",
        dataset={
            "path": "mteb/JinaVDREuropeanaDeNewsRetrieval",
            "revision": "29f7abde0e2be4ae907a09ad5e060a79fc391215",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["deu-Latn"],
        domains=["News"],
        license="cc0-1.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDREuropeanaEsNewsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDREuropeanaEsNewsRetrieval",
        description="Retrieve Spanish news articles based on LLM generated queries. This dataset was created from records of the [Europeana](https://europeana.eu/) online collection by selecting scans of Spanish news articles",
        reference="https://huggingface.co/datasets/jinaai/europeana-es-news_beir",
        dataset={
            "path": "mteb/JinaVDREuropeanaEsNewsRetrieval",
            "revision": "8bb49ac0c8900a50246f70ea60a312f55c824720",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["spa-Latn"],
        domains=["News"],
        license="cc0-1.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDREuropeanaItScansRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDREuropeanaItScansRetrieval",
        description="Retrieve Italian historical articles based on LLM generated queries. This dataset was created from records of the [Europeana](https://europeana.eu/) online collection by selecting scans of Italian news articles",
        reference="https://huggingface.co/datasets/jinaai/europeana-it-scans_beir",
        dataset={
            "path": "mteb/JinaVDREuropeanaItScansRetrieval",
            "revision": "19cf14066860dc3b7642aa5b18d4a68159c13585",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["ita-Latn"],
        domains=["News"],
        license="cc0-1.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDREuropeanaNlLegalRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDREuropeanaNlLegalRetrieval",
        description="Retrieve Dutch historical legal documents based on LLM generated queries.  This dataset was created from records of the [Europeana](https://europeana.eu/) online collection by selecting scans of Dutch news articles",
        reference="https://huggingface.co/datasets/jinaai/europeana-nl-legal_beir",
        dataset={
            "path": "mteb/JinaVDREuropeanaNlLegalRetrieval",
            "revision": "0b60de8d44f051c4f31d9b62fcbfe2f99ce2c1a8",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["nld-Latn"],
        domains=["Legal"],
        license="cc0-1.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRHindiGovVQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRHindiGovVQARetrieval",
        description="Retrieve Hindi government documents based on LLM generated queries.",
        reference="https://huggingface.co/datasets/jinaai/hindi-gov-vqa_beir",
        dataset={
            "path": "mteb/JinaVDRHindiGovVQARetrieval",
            "revision": "f0a4dc9308eafd7fdec67ad1647b06a510b0ffd2",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["hin-Deva"],
        domains=["Government"],
        license="not specified",
        annotations_creators="LM-generated",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRAutomobileCatelogRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRAutomobileCatelogRetrieval",
        description="Retrieve automobile marketing documents based on LLM generated queries. Marketing document from Toyota Japanese website featuring [RAV4](https://toyota.jp/pages/contents/request/webcatalog/rav4/rav4_special1_202310.pdf) and [Corolla](https://toyota.jp/pages/contents/request/webcatalog/corolla/corolla_special1_202407.pdf). The `text_description` column contains OCR text extracted from the images using EasyOCR.",
        reference="https://huggingface.co/datasets/jinaai/automobile_catalogue_jp_beir",
        dataset={
            "path": "mteb/JinaVDRAutomobileCatelogRetrieval",
            "revision": "26f80aa0329845562913f6c63ed276c40e577985",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["jpn-Jpan"],
        domains=["Engineering", "Web"],
        license="not specified",
        annotations_creators="LM-generated",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRBeveragesCatalogueRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRBeveragesCatalogueRetrieval",
        description="Retrieve beverages marketing documents based on LLM generated queries. This dataset was self-curated by searching beverage catalogs on Google search and downloading PDFs.",
        reference="https://huggingface.co/datasets/jinaai/beverages_catalogue_ru_beir",
        dataset={
            "path": "mteb/JinaVDRBeveragesCatalogueRetrieval",
            "revision": "d076250dfec7c26f9e461f365601bb78d0a44883",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["rus-Cyrl"],
        domains=["Web"],
        license="not specified",
        annotations_creators="LM-generated",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRRamensBenchmarkRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRRamensBenchmarkRetrieval",
        description="Retrieve ramen restaurant marketing documents based on LLM generated queries. Marketing document from Ramen [restaurants](https://www.city.niigata.lg.jp/kanko/kanko/oshirase/ramen.files/guidebook.pdf).",
        reference="https://huggingface.co/datasets/jinaai/ramen_benchmark_jp_beir",
        dataset={
            "path": "mteb/JinaVDRRamensBenchmarkRetrieval",
            "revision": "b97858ee9a057caf2ed6da6f105e5b92c20b6c67",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["jpn-Jpan"],
        domains=["Web"],
        license="not specified",
        annotations_creators="LM-generated",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRJDocQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRJDocQARetrieval",
        description="Retrieve Japanese documents in various formats based on human annotated queries. Document Question answering from [JDocQAJP dataset](https://huggingface.co/datasets/jlli/JDocQA-nonbinary), test split.",
        reference="https://huggingface.co/datasets/jinaai/jdocqa_beir",
        dataset={
            "path": "mteb/JinaVDRJDocQARetrieval",
            "revision": "683db9cabbfa44e71aa2772c868c7c0ca3722518",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["jpn-Jpan"],
        domains=["Web"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRHungarianDocQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRHungarianDocQARetrieval",
        description="Retrieve Hungarian documents in various formats based on human annotated queries. Document Question answering from [Hungurian doc qa dataset](https://huggingface.co/datasets/jlli/HungarianDocQA-OCR), test split.",
        reference="https://huggingface.co/datasets/jinaai/hungarian_doc_qa_beir",
        dataset={
            "path": "mteb/JinaVDRHungarianDocQARetrieval",
            "revision": "5ab283d150d2c4670f3d2338f03f95b8200bb5e6",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["hun-Latn"],
        domains=["Web"],
        license="cc-by-4.0",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRArabicChartQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRArabicChartQARetrieval",
        description="Retrieve Arabic charts based on queries. This dataset is derived from the [Arabic ChartQA dataset](https://huggingface.co/datasets/ahmedheakl/arabic_chartqa), reformatting the train split as a test split with modified field names such that it is compatible with the ViDoRe evaluation benchmark.",
        reference="https://huggingface.co/datasets/jinaai/arabic_chartqa_ar_beir",
        dataset={
            "path": "mteb/JinaVDRArabicChartQARetrieval",
            "revision": "453097afd6d4500f8ddfab8570a294c3d095fc24",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["ara-Arab"],
        domains=["Academic"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRArabicInfographicsVQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRArabicInfographicsVQARetrieval",
        description="Retrieve Arabic infographics based on queries. This dataset is derived from the [Arabic Infographics VQA dataset](https://huggingface.co/datasets/ahmedheakl/arabic_infographicsvqa), reformatting the train split as a test split with modified field names so it can be used in the ViDoRe evaluation benchmark.",
        reference="https://huggingface.co/datasets/jinaai/arabic_infographicsvqa_ar_beir",
        dataset={
            "path": "mteb/JinaVDRArabicInfographicsVQARetrieval",
            "revision": "0d247d4b2405e30d5af03f804036de271c360a3f",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["ara-Arab"],
        domains=["Academic"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDROWIDChartsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDROWIDChartsRetrieval",
        description="Retrieve charts from the OWID dataset based on accompanied text snippets. We sampled a set of ~5k charts and articles from [Our World In Data](https://ourworldindata.org) to produce this evaluation set. This particular dataset is a subsample of 1000 random charts from the full dataset which can be found [here](https://huggingface.co/datasets/jjinaai/owid_charts).",
        reference="https://huggingface.co/datasets/jinaai/owid_charts_en_beir",
        dataset={
            "path": "mteb/JinaVDROWIDChartsRetrieval",
            "revision": "24b5554134f9886e44c0b779602f6de2968a393e",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="cc-by-4.0",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRMPMQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRMPMQARetrieval",
        description="Retrieve product manuals based on human annotated queries. 155 questions and 782 document images cleaned from [jinaai/MPMQA](https://huggingface.co/datasets/jinaai/MPMQA), test set.",  # MPMQA not exists on HF
        reference="https://huggingface.co/datasets/jinaai/mpmqa_small_beir",
        dataset={
            "path": "mteb/JinaVDRMPMQARetrieval",
            "revision": "a0dd69f290617d5608b37048978b433268f0396e",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRJina2024YearlyBookRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRJina2024YearlyBookRetrieval",
        description="Retrieve pages from the 2024 Jina yearbook based on human annotated questions. 75 human annotated questions created from digital version of Jina AI yearly book 2024, 166 pages in total. ",
        reference="https://huggingface.co/datasets/jinaai/jina_2024_yearly_book_beir",
        dataset={
            "path": "mteb/JinaVDRJina2024YearlyBookRetrieval",
            "revision": "4ad99b503861f13f860af46ef6050d558bdac93b",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="apache-2.0",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRWikimediaCommonsMapsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRWikimediaCommonsMapsRetrieval",
        description="Retrieve maps from Wikimedia Commons based on their description. It contains images of (mostly historic) maps which should be identified based on their description. We extracted those descriptions from [Wikimedia Commons](https://commons.wikimedia.org/). We have included the license type and a link (license_text) to the original Wikimedia Commons page for each extracted image.",
        reference="https://huggingface.co/datasets/jinaai/wikimedia-commons-maps_beir",
        dataset={
            "path": "mteb/JinaVDRWikimediaCommonsMapsRetrieval",
            "revision": "887ce0f36755707c84a8dee30d5398762e43b4e4",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="multiple",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRPlotQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRPlotQARetrieval",
        description="Retrieve plots from the PlotQA dataset based on LLM generated queries. Questions subsampled from [PlotQA](https://github.com/NiteshMethani/PlotQA) test set. It is following a subsample + LLM-based classification process, using LLM to verify the question quality, e.g. queries like `How many different coloured dotlines are there` will be filtered out.",
        reference="https://huggingface.co/datasets/jinaai/plotqa_beir",
        dataset={
            "path": "mteb/JinaVDRPlotQARetrieval",
            "revision": "d5e1242a6b1c690f4abf2e36305c3b6d71725f0c",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="cc-by-4.0",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRMMTabRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRMMTabRetrieval",
        description="Retrieve tables from the MMTab dataset based on queries. This dataset is a copy of the original test split from MMTab, taking only items where an 'original_query' is present, and removing the 'input' and 'output' columns, as they are unnecessary for retrieval tasks.",
        reference="https://huggingface.co/datasets/jinaai/MMTab_beir",
        dataset={
            "path": "mteb/JinaVDRMMTabRetrieval",
            "revision": "ae7a5bd9a9a7d0958a5a5638469fa99fc6ed0726",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="mit",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRCharXivOCRRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRCharXivOCRRetrieval",
        description="Retrieve charts from scientific papers based on human annotated queries. This dataset is derived from the [CharXiv dataset](https://huggingface.co/datasets/princeton-nlp/CharXiv), reformatting the test split with modified field names, so that it can be used in the ViDoRe benchmark.",
        reference="https://huggingface.co/datasets/jinaai/CharXiv-en_beir",
        dataset={
            "path": "mteb/JinaVDRCharXivOCRRetrieval",
            "revision": "e3e32220c9ccc70a3361d42b6e9df7f29b22fa46",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Web"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRStudentEnrollmentSyntheticRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRStudentEnrollmentSyntheticRetrieval",
        description="Retrieve student enrollment data based on templated queries. This dataset is created from the original Kaggle [Delaware Student Enrollment](https://www.kaggle.com/datasets/noeyislearning/delaware-student-enrollment) dataset. The charts are rendered and queries created using templates.",
        reference="https://huggingface.co/datasets/jinaai/student-enrollment_beir",
        dataset={
            "path": "mteb/JinaVDRStudentEnrollmentSyntheticRetrieval",
            "revision": "c887d9c6ea33aca782d2bd2b090e4587d2fa0bdc",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["eng-Latn"],
        domains=["Academic"],
        license="cc0-1.0",
        annotations_creators="derived",
        sample_creation="found",
        **COMMON_METADATA,
    )


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
            "path": "mteb/JinaVDRGitHubReadmeRetrieval",
            "revision": "8cf4a3439738baafff179572a84b9857e3f39624",
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


class JinaVDRTweetStockSyntheticsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRTweetStockSyntheticsRetrieval",
        description="Retrieve rendered tables of stock prices based on templated queries. This dataset is created from the original Kaggle [Tweet Sentiment's Impact on Stock Returns](https://www.kaggle.com/datasets/thedevastator/tweet-sentiment-s-impact-on-stock-returns) dataset.",
        reference="https://huggingface.co/datasets/jinaai/tweet-stock-synthetic-retrieval_beir",
        dataset={
            "path": "mteb/JinaVDRTweetStockSyntheticsRetrieval",
            "revision": "8548a58c9c748e0c54d46b50d038bd1620808eec",
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


class JinaVDRAirbnbSyntheticRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRAirbnbSyntheticRetrieval",
        description="Retrieve rendered tables from Airbnb listings based on templated queries. This dataset is created from the original Kaggle [New York City Airbnb Open Data dataset](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data).",
        reference="https://huggingface.co/datasets/jinaai/airbnb-synthetic-retrieval_beir",
        dataset={
            "path": "mteb/JinaVDRAirbnbSyntheticRetrieval",
            "revision": "e9e8f8b59a94814e76f8ccbb46461151e1a18228",
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


class JinaVDRShanghaiMasterPlanRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRShanghaiMasterPlanRetrieval",
        description="Retrieve pages from the Shanghai Master Plan based on human annotated queries. The master plan document is taken from [here](https://www.shanghai.gov.cn/newshanghai/xxgkfj/2035004.pdf).",
        reference="https://huggingface.co/datasets/jinaai/shanghai_master_plan_beir",
        dataset={
            "path": "mteb/JinaVDRShanghaiMasterPlanRetrieval",
            "revision": "dff5daf575436bdae3733a2dc4ee754f8157f366",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["zho-Hans"],
        domains=["Web"],
        license="not specified",
        annotations_creators="human-annotated",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRWikimediaCommonsDocumentsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRWikimediaCommonsDocumentsRetrieval",
        description="Retrieve historical documents from Wikimedia Commons based on their description. Wikimedia Commons Documents. It contains images of (mostly historic) documents which should be identified based on their description. We extracted those descriptions from Wikimedia Commons. We have included the license type and a link (`license_text`) to the original Wikimedia Commons page for each extracted image.",
        reference="https://huggingface.co/datasets/jinaai/wikimedia-commons-documents-ml_beir",
        dataset={
            "path": "mteb/JinaVDRWikimediaCommonsDocumentsRetrieval",
            "revision": "f41e46aee7358cec5359fdfac4ec31ef1e56fa3d",
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


class JinaVDREuropeanaFrNewsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDREuropeanaFrNewsRetrieval",
        description="Retrieve French news articles from Europeana based on LLM generated queries. This dataset was created from records of the [Europeana online collection](https://europeana.eu) by selecting scans of French news articles.",
        reference="https://huggingface.co/datasets/jinaai/europeana-fr-news_beir",
        dataset={
            "path": "mteb/JinaVDREuropeanaFrNewsRetrieval",
            "revision": "d454c233c8f26a4db35b9619389dd874a8f828f0",
        },
        date=("2024-10-01", "2025-04-01"),
        eval_langs=["fra-Latn"],
        domains=["News"],
        license="cc0-1.0",
        annotations_creators="LM-generated",
        sample_creation="found",
        **COMMON_METADATA,
    )


class JinaVDRDocQAHealthcareIndustryRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRDocQAHealthcareIndustryRetrieval",
        description="Retrieve healthcare industry documents based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d). For more information regarding the filtering please read [our paper](https://arxiv.org/abs/2506.18902) or [this discussion on github](https://github.com/embeddings-benchmark/mteb/pull/2942#discussion_r2240711654).",
        reference="https://huggingface.co/datasets/jinaai/docqa_healthcare_industry_beir",
        dataset={
            "path": "mteb/JinaVDRDocQAHealthcareIndustryRetrieval",
            "revision": "a923b350b7a576b58ff5862ac09701b59165c6ed",
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


class JinaVDRDocQAAI(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRDocQAAI",
        description="Retrieve AI documents based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).",
        reference="https://huggingface.co/datasets/jinaai/docqa_artificial_intelligence_beir",
        dataset={
            "path": "mteb/JinaVDRDocQAAI",
            "revision": "b2753f7de6b95fd2f10853bf5b7038ce91e5b540",
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


class JinaVDRShiftProjectRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRShiftProjectRetrieval",
        description="Retrieve documents with graphs from the Shift Project based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).",
        reference="https://huggingface.co/datasets/jinaai/shiftproject_beir",
        dataset={
            "path": "mteb/JinaVDRShiftProjectRetrieval",
            "revision": "024ad897b29b7f672d7342f2c05a5a03d59e1924",
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


class JinaVDRTatQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRTatQARetrieval",
        description="Retrieve financial reports based on human annotated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).",
        reference="https://huggingface.co/datasets/jinaai/tatqa_beir",
        dataset={
            "path": "mteb/JinaVDRTatQARetrieval",
            "revision": "789f09e9342fbe79b11902412e8d7bd50369b11f",
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


class JinaVDRInfovqaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRInfovqaRetrieval",
        description="Retrieve infographics based on human annotated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).",
        reference="https://huggingface.co/datasets/jinaai/infovqa_beir",
        dataset={
            "path": "mteb/JinaVDRInfovqaRetrieval",
            "revision": "1d0259c673e23d5986a2f3738c182de377b40bbe",
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


class JinaVDRDocVQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRDocVQARetrieval",
        description="Retrieve industry documents based on human annotated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).",
        reference="https://huggingface.co/datasets/jinaai/docvqa_beir",
        dataset={
            "path": "mteb/JinaVDRDocVQARetrieval",
            "revision": "b6460c9174a963a489e78bf607e8656b7b95873b",
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


class JinaVDRDocQAGovReportRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRDocQAGovReportRetrieval",
        description="Retrieve government reports based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).",
        reference="https://huggingface.co/datasets/jinaai/docqa_gov_report_beir",
        dataset={
            "path": "mteb/JinaVDRDocQAGovReportRetrieval",
            "revision": "a80e6856ca795484e07e1620d2487e17aa6febc0",
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


class JinaVDRTabFQuadRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRTabFQuadRetrieval",
        description="Retrieve tables from industry documents based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).",
        reference="https://huggingface.co/datasets/jinaai/tabfquad_beir",
        dataset={
            "path": "mteb/JinaVDRTabFQuadRetrieval",
            "revision": "e500d13e30e33e6698a36b096f46cb68a5c53feb",
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


class JinaVDRDocQAEnergyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRDocQAEnergyRetrieval",
        description="Retrieve energy industry documents based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).",
        reference="https://huggingface.co/datasets/jinaai/docqa_energy_beir",
        dataset={
            "path": "mteb/JinaVDRDocQAEnergyRetrieval",
            "revision": "06bb0a8125d176a63a65515daebfaa9bdead5c6c",
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


class JinaVDRArxivQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JinaVDRArxivQARetrieval",
        description="Retrieve figures from scientific papers from arXiv based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).",
        reference="https://huggingface.co/datasets/jinaai/arxivqa_beir",
        dataset={
            "path": "mteb/JinaVDRArxivQARetrieval",
            "revision": "2c7fffa36a0d6acee56b30e1f59696eb906bd316",
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
