
# DocumentUnderstanding

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 66

#### JinaVDRAirbnbSyntheticRetrieval

Retrieve rendered tables from Airbnb listings based on templated queries. This dataset is created from the original Kaggle [New York City Airbnb Open Data dataset](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data).

**Dataset:** [`jinaai/airbnb-synthetic-retrieval_beir`](https://huggingface.co/datasets/jinaai/airbnb-synthetic-retrieval_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/airbnb-synthetic-retrieval_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | ara, deu, eng, fra, hin, ... (10) | Web | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRArabicChartQARetrieval

Retrieve Arabic charts based on queries. This dataset is derived from the [Arabic ChartQA dataset](https://huggingface.co/datasets/ahmedheakl/arabic_chartqa), reformatting the train split as a test split with modified field names such that it is compatible with the ViDoRe evaluation benchmark.

**Dataset:** [`jinaai/arabic_chartqa_ar_beir`](https://huggingface.co/datasets/jinaai/arabic_chartqa_ar_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/arabic_chartqa_ar_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | ara | Academic | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRArabicInfographicsVQARetrieval

Retrieve Arabic infographics based on queries. This dataset is derived from the [Arabic Infographics VQA dataset](https://huggingface.co/datasets/ahmedheakl/arabic_infographicsvqa), reformatting the train split as a test split with modified field names so it can be used in the ViDoRe evaluation benchmark.

**Dataset:** [`jinaai/arabic_infographicsvqa_ar_beir`](https://huggingface.co/datasets/jinaai/arabic_infographicsvqa_ar_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/arabic_infographicsvqa_ar_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | ara | Academic | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRArxivQARetrieval

Retrieve figures from scientific papers from arXiv based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).

**Dataset:** [`jinaai/arxivqa_beir`](https://huggingface.co/datasets/jinaai/arxivqa_beir) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/arxivqa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | LM-generated | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRAutomobileCatelogRetrieval

Retrieve automobile marketing documents based on LLM generated queries. Marketing document from Toyota Japanese website featuring [RAV4](https://toyota.jp/pages/contents/request/webcatalog/rav4/rav4_special1_202310.pdf) and [Corolla](https://toyota.jp/pages/contents/request/webcatalog/corolla/corolla_special1_202407.pdf). The `text_description` column contains OCR text extracted from the images using EasyOCR.

**Dataset:** [`jinaai/automobile_catalogue_jp_beir`](https://huggingface.co/datasets/jinaai/automobile_catalogue_jp_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/automobile_catalogue_jp_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | jpn | Engineering, Web | LM-generated | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRBeveragesCatalogueRetrieval

Retrieve beverages marketing documents based on LLM generated queries. This dataset was self-curated by searching beverage catalogs on Google search and downloading PDFs.

**Dataset:** [`jinaai/beverages_catalogue_ru_beir`](https://huggingface.co/datasets/jinaai/beverages_catalogue_ru_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/beverages_catalogue_ru_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | rus | Web | LM-generated | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRCharXivOCRRetrieval

Retrieve charts from scientific papers based on human annotated queries. This dataset is derived from the [CharXiv dataset](https://huggingface.co/datasets/princeton-nlp/CharXiv), reformatting the test split with modified field names, so that it can be used in the ViDoRe benchmark.

**Dataset:** [`jinaai/CharXiv-en_beir`](https://huggingface.co/datasets/jinaai/CharXiv-en_beir) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/CharXiv-en_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRChartQARetrieval

Retrieve charts based on LLM generated queries. Source datasets https://huggingface.co/datasets/HuggingFaceM4/ChartQA

**Dataset:** [`jinaai/ChartQA_beir`](https://huggingface.co/datasets/jinaai/ChartQA_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/ChartQA_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRDocQAAI

Retrieve AI documents based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).

**Dataset:** [`jinaai/docqa_artificial_intelligence_beir`](https://huggingface.co/datasets/jinaai/docqa_artificial_intelligence_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/docqa_artificial_intelligence_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRDocQAEnergyRetrieval

Retrieve energy industry documents based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).

**Dataset:** [`jinaai/docqa_energy_beir`](https://huggingface.co/datasets/jinaai/docqa_energy_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/docqa_energy_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRDocQAGovReportRetrieval

Retrieve government reports based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).

**Dataset:** [`jinaai/docqa_gov_report_beir`](https://huggingface.co/datasets/jinaai/docqa_gov_report_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/docqa_gov_report_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Government | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRDocQAHealthcareIndustryRetrieval

Retrieve healthcare industry documents based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d). For more information regarding the filtering please read [our paper](https://arxiv.org/abs/2506.18902) or [this discussion on github](https://github.com/embeddings-benchmark/mteb/pull/2942#discussion_r2240711654).

**Dataset:** [`jinaai/docqa_healthcare_industry_beir`](https://huggingface.co/datasets/jinaai/docqa_healthcare_industry_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/docqa_healthcare_industry_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Medical | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRDocVQARetrieval

Retrieve industry documents based on human annotated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).

**Dataset:** [`jinaai/docvqa_beir`](https://huggingface.co/datasets/jinaai/docvqa_beir) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/docvqa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | LM-generated | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRDonutVQAISynHMPRetrieval

Retrieve medical records based on templated queries. Source dataset https://huggingface.co/datasets/warshakhan/donut_vqa_ISynHMP

**Dataset:** [`jinaai/donut_vqa_beir`](https://huggingface.co/datasets/jinaai/donut_vqa_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/donut_vqa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Medical | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDREuropeanaDeNewsRetrieval

Retrieve German news articles based on LLM generated queries. This dataset was created from records of the [Europeana](https://europeana.eu/) online collection by selecting scans of German news articles

**Dataset:** [`jinaai/europeana-de-news_beir`](https://huggingface.co/datasets/jinaai/europeana-de-news_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/europeana-de-news_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | deu | News | LM-generated | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDREuropeanaEsNewsRetrieval

Retrieve Spanish news articles based on LLM generated queries. This dataset was created from records of the [Europeana](https://europeana.eu/) online collection by selecting scans of Spanish news articles

**Dataset:** [`jinaai/europeana-es-news_beir`](https://huggingface.co/datasets/jinaai/europeana-es-news_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/europeana-es-news_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | spa | News | LM-generated | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDREuropeanaFrNewsRetrieval

Retrieve French news articles from Europeana based on LLM generated queries. This dataset was created from records of the [Europeana online collection](https://europeana.eu) by selecting scans of French news articles.

**Dataset:** [`jinaai/europeana-fr-news_beir`](https://huggingface.co/datasets/jinaai/europeana-fr-news_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/europeana-fr-news_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | fra | News | LM-generated | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDREuropeanaItScansRetrieval

Retrieve Italian historical articles based on LLM generated queries. This dataset was created from records of the [Europeana](https://europeana.eu/) online collection by selecting scans of Italian news articles

**Dataset:** [`jinaai/europeana-it-scans_beir`](https://huggingface.co/datasets/jinaai/europeana-it-scans_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/europeana-it-scans_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | ita | News | LM-generated | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDREuropeanaNlLegalRetrieval

Retrieve Dutch historical legal documents based on LLM generated queries.  This dataset was created from records of the [Europeana](https://europeana.eu/) online collection by selecting scans of Dutch news articles

**Dataset:** [`jinaai/europeana-nl-legal_beir`](https://huggingface.co/datasets/jinaai/europeana-nl-legal_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/europeana-nl-legal_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | nld | Legal | LM-generated | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRGitHubReadmeRetrieval

Retrieve GitHub readme files based their description. This dataset consists of rendered GitHub readmes in a variety of different languages, together with their accompanying descriptions as queries and their license in the `license_type` and `license_text` columns. This particular dataset is a subsample of 1000 random rows per language from the full dataset which can be found [here](https://huggingface.co/datasets/jinaai/github-readme-retrieval-ml-filtered).

**Dataset:** [`jinaai/github-readme-retrieval-multilingual_beir`](https://huggingface.co/datasets/jinaai/github-readme-retrieval-multilingual_beir) • **License:** multiple • [Learn more →](https://huggingface.co/datasets/jinaai/github-readme-retrieval-multilingual_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | ara, ben, deu, eng, fra, ... (17) | Web | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRHindiGovVQARetrieval

Retrieve Hindi government documents based on LLM generated queries.

**Dataset:** [`jinaai/hindi-gov-vqa_beir`](https://huggingface.co/datasets/jinaai/hindi-gov-vqa_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/hindi-gov-vqa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | hin | Government | LM-generated | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRHungarianDocQARetrieval

Retrieve Hungarian documents in various formats based on human annotated queries. Document Question answering from [Hungurian doc qa dataset](https://huggingface.co/datasets/jlli/HungarianDocQA-OCR), test split.

**Dataset:** [`jinaai/hungarian_doc_qa_beir`](https://huggingface.co/datasets/jinaai/hungarian_doc_qa_beir) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/hungarian_doc_qa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | hun | Web | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRInfovqaRetrieval

Retrieve infographics based on human annotated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).

**Dataset:** [`jinaai/infovqa_beir`](https://huggingface.co/datasets/jinaai/infovqa_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/infovqa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRJDocQARetrieval

Retrieve Japanese documents in various formats based on human annotated queries. Document Question answering from [JDocQAJP dataset](https://huggingface.co/datasets/jlli/JDocQA-nonbinary), test split.

**Dataset:** [`jinaai/jdocqa_beir`](https://huggingface.co/datasets/jinaai/jdocqa_beir) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/jdocqa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | jpn | Web | LM-generated | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRJina2024YearlyBookRetrieval

Retrieve pages from the 2024 Jina yearbook based on human annotated questions. 75 human annotated questions created from digital version of Jina AI yearly book 2024, 166 pages in total.

**Dataset:** [`jinaai/jina_2024_yearly_book_beir`](https://huggingface.co/datasets/jinaai/jina_2024_yearly_book_beir) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/jinaai/jina_2024_yearly_book_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRMMTabRetrieval

Retrieve tables from the MMTab dataset based on queries. This dataset is a copy of the original test split from MMTab, taking only items where an 'original_query' is present, and removing the 'input' and 'output' columns, as they are unnecessary for retrieval tasks.

**Dataset:** [`jinaai/MMTab_beir`](https://huggingface.co/datasets/jinaai/MMTab_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/MMTab_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRMPMQARetrieval

Retrieve product manuals based on human annotated queries. 155 questions and 782 document images cleaned from [jinaai/MPMQA](https://huggingface.co/datasets/jinaai/MPMQA), test set.

**Dataset:** [`jinaai/mpmqa_small_beir`](https://huggingface.co/datasets/jinaai/mpmqa_small_beir) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/jinaai/mpmqa_small_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | human-annotated | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRMedicalPrescriptionsRetrieval

Retrieve medical prescriptions based on templated queries. Source dataset https://huggingface.co/datasets/Technoculture/medical-prescriptions

**Dataset:** [`jinaai/medical-prescriptions_beir`](https://huggingface.co/datasets/jinaai/medical-prescriptions_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/medical-prescriptions_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Medical | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDROWIDChartsRetrieval

Retrieve charts from the OWID dataset based on accompanied text snippets. We sampled a set of ~5k charts and articles from [Our World In Data](https://ourworldindata.org) to produce this evaluation set. This particular dataset is a subsample of 1000 random charts from the full dataset which can be found [here](https://huggingface.co/datasets/jjinaai/owid_charts).

**Dataset:** [`jinaai/owid_charts_en_beir`](https://huggingface.co/datasets/jinaai/owid_charts_en_beir) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/owid_charts_en_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDROpenAINewsRetrieval

Retrieve news articles from the OpenAI news website based on human annotated queries. News taken from https://openai.com/news/

**Dataset:** [`jinaai/openai-news_beir`](https://huggingface.co/datasets/jinaai/openai-news_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/openai-news_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | News, Web | human-annotated | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRPlotQARetrieval

Retrieve plots from the PlotQA dataset based on LLM generated queries. Questions subsampled from [PlotQA](https://github.com/NiteshMethani/PlotQA) test set. It is following a subsample + LLM-based classification process, using LLM to verify the question quality, e.g. queries like `How many different coloured dotlines are there` will be filtered out.

**Dataset:** [`jinaai/plotqa_beir`](https://huggingface.co/datasets/jinaai/plotqa_beir) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/plotqa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRRamensBenchmarkRetrieval

Retrieve ramen restaurant marketing documents based on LLM generated queries. Marketing document from Ramen [restaurants](https://www.city.niigata.lg.jp/kanko/kanko/oshirase/ramen.files/guidebook.pdf).

**Dataset:** [`jinaai/ramen_benchmark_jp_beir`](https://huggingface.co/datasets/jinaai/ramen_benchmark_jp_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/ramen_benchmark_jp_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | jpn | Web | LM-generated | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRShanghaiMasterPlanRetrieval

Retrieve pages from the Shanghai Master Plan based on human annotated queries. The master plan document is taken from [here](https://www.shanghai.gov.cn/newshanghai/xxgkfj/2035004.pdf).

**Dataset:** [`jinaai/shanghai_master_plan_beir`](https://huggingface.co/datasets/jinaai/shanghai_master_plan_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/shanghai_master_plan_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | zho | Web | human-annotated | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRShiftProjectRetrieval

Retrieve documents with graphs from the Shift Project based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).

**Dataset:** [`jinaai/shiftproject_beir`](https://huggingface.co/datasets/jinaai/shiftproject_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/shiftproject_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRStanfordSlideRetrieval

Retrieve scientific and engineering slides based on human annotated queries. Source dataset https://exhibits.stanford.edu/data/catalog/mv327tb8364

**Dataset:** [`jinaai/stanford_slide_beir`](https://huggingface.co/datasets/jinaai/stanford_slide_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/stanford_slide_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | human-annotated | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRStudentEnrollmentSyntheticRetrieval

Retrieve student enrollment data based on templated queries. This dataset is created from the original Kaggle [Delaware Student Enrollment](https://www.kaggle.com/datasets/noeyislearning/delaware-student-enrollment) dataset. The charts are rendered and queries created using templates.

**Dataset:** [`jinaai/student-enrollment_beir`](https://huggingface.co/datasets/jinaai/student-enrollment_beir) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jinaai/student-enrollment_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRTQARetrieval

Retrieve textbook pages (images and text) based on LLM generated queries from the text. Source datasets https://prior.allenai.org/projects/tqa

**Dataset:** [`jinaai/tqa_beir`](https://huggingface.co/datasets/jinaai/tqa_beir) • **License:** cc-by-nc-3.0 • [Learn more →](https://huggingface.co/datasets/jinaai/tqa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRTabFQuadRetrieval

Retrieve tables from industry documents based on LLM generated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).

**Dataset:** [`jinaai/tabfquad_beir`](https://huggingface.co/datasets/jinaai/tabfquad_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/tabfquad_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRTableVQARetrieval

Retrieve scientific tables based on LLM generated queries. Source datasets https://huggingface.co/datasets/HuggingFaceM4/ChartQA or https://huggingface.co/datasets/cmarkea/aftdb

**Dataset:** [`jinaai/table-vqa_beir`](https://huggingface.co/datasets/jinaai/table-vqa_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/table-vqa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRTatQARetrieval

Retrieve financial reports based on human annotated queries. This dataset is build upon the corresponding dataset from the [ViDoRe Benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d).

**Dataset:** [`jinaai/tatqa_beir`](https://huggingface.co/datasets/jinaai/tatqa_beir) • **License:** mit • [Learn more →](https://huggingface.co/datasets/jinaai/tatqa_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRTweetStockSyntheticsRetrieval

Retrieve rendered tables of stock prices based on templated queries. This dataset is created from the original Kaggle [Tweet Sentiment's Impact on Stock Returns](https://www.kaggle.com/datasets/thedevastator/tweet-sentiment-s-impact-on-stock-returns) dataset.

**Dataset:** [`jinaai/tweet-stock-synthetic-retrieval_beir`](https://huggingface.co/datasets/jinaai/tweet-stock-synthetic-retrieval_beir) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/tweet-stock-synthetic-retrieval_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | ara, deu, eng, fra, hin, ... (10) | Social | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRWikimediaCommonsDocumentsRetrieval

Retrieve historical documents from Wikimedia Commons based on their description. Wikimedia Commons Documents. It contains images of (mostly historic) documents which should be identified based on their description. We extracted those descriptions from Wikimedia Commons. We have included the license type and a link (`license_text`) to the original Wikimedia Commons page for each extracted image.

**Dataset:** [`jinaai/wikimedia-commons-documents-ml_beir`](https://huggingface.co/datasets/jinaai/wikimedia-commons-documents-ml_beir) • **License:** multiple • [Learn more →](https://huggingface.co/datasets/jinaai/wikimedia-commons-documents-ml_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | ara, ben, deu, eng, fra, ... (20) | Web | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### JinaVDRWikimediaCommonsMapsRetrieval

Retrieve maps from Wikimedia Commons based on their description. It contains images of (mostly historic) maps which should be identified based on their description. We extracted those descriptions from [Wikimedia Commons](https://commons.wikimedia.org/). We have included the license type and a link (license_text) to the original Wikimedia Commons page for each extracted image.

**Dataset:** [`jinaai/wikimedia-commons-maps_beir`](https://huggingface.co/datasets/jinaai/wikimedia-commons-maps_beir) • **License:** multiple • [Learn more →](https://huggingface.co/datasets/jinaai/wikimedia-commons-maps_beir)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Web | derived | found |



??? quote "Citation"


    ```bibtex
    @misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
      archiveprefix = {arXiv},
      author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
      eprint = {2506.18902},
      primaryclass = {cs.AI},
      title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
      url = {https://arxiv.org/abs/2506.18902},
      year = {2025},
    }
    ```




#### MIRACLVisionRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`nvidia/miracl-vision`](https://huggingface.co/datasets/nvidia/miracl-vision) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | ara, ben, deu, eng, fas, ... (18) | Encyclopaedic | derived | created |



??? quote "Citation"


    ```bibtex

    @article{osmulski2025miraclvisionlargemultilingualvisual,
      author = {Radek Osmulski and Gabriel de Souza P. Moreira and Ronay Ak and Mengyao Xu and Benedikt Schifferer and Even Oldridge},
      eprint = {2505.11651},
      journal = {arxiv},
      title = {{MIRACL-VISION: A Large, multilingual, visual document retrieval benchmark}},
      url = {https://arxiv.org/abs/2505.11651},
      year = {2025},
    }

    ```




#### Vidore2BioMedicalLecturesRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/biomedical_lectures_v2`](https://huggingface.co/datasets/vidore/biomedical_lectures_v2) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | deu, eng, fra, spa | Academic | derived | found |



??? quote "Citation"


    ```bibtex

    @article{mace2025vidorev2,
      author = {Macé, Quentin and Loison António and Faysse, Manuel},
      journal = {arXiv preprint arXiv:2505.17166},
      title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
      year = {2025},
    }

    ```




#### Vidore2ESGReportsHLRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/esg_reports_human_labeled_v2`](https://huggingface.co/datasets/vidore/esg_reports_human_labeled_v2) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



??? quote "Citation"


    ```bibtex

    @article{mace2025vidorev2,
      author = {Macé, Quentin and Loison António and Faysse, Manuel},
      journal = {arXiv preprint arXiv:2505.17166},
      title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
      year = {2025},
    }

    ```




#### Vidore2ESGReportsRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/esg_reports_v2`](https://huggingface.co/datasets/vidore/esg_reports_v2) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | deu, eng, fra, spa | Academic | derived | found |



??? quote "Citation"


    ```bibtex

    @article{mace2025vidorev2,
      author = {Macé, Quentin and Loison António and Faysse, Manuel},
      journal = {arXiv preprint arXiv:2505.17166},
      title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
      year = {2025},
    }

    ```




#### Vidore2EconomicsReportsRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/economics_reports_v2`](https://huggingface.co/datasets/vidore/economics_reports_v2) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | deu, eng, fra, spa | Academic | derived | found |



??? quote "Citation"


    ```bibtex

    @article{mace2025vidorev2,
      author = {Macé, Quentin and Loison António and Faysse, Manuel},
      journal = {arXiv preprint arXiv:2505.17166},
      title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
      year = {2025},
    }

    ```




#### Vidore3ComputerScienceRetrieval

Retrieve associated pages according to questions. This dataset, Computer Science, is a corpus of textbooks from the openstacks website, intended for long-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.

**Dataset:** [`vidore/vidore_v3_computer_science_mteb_format`](https://huggingface.co/datasets/vidore/vidore_v3_computer_science_mteb_format) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/blog/QuentinJG/introducing-vidore-v3)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | deu, eng, fra, ita, por, ... (6) | Engineering, Programming | derived | created and machine-translated |



??? quote "Citation"


    ```bibtex

    @misc{mace2025vidorev3,
      author = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
      day = {5},
      howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}},
      journal = {Hugging Face Blog},
      month = {November},
      publisher = {Hugging Face},
      title = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
      year = {2025},
    }

    ```




#### Vidore3EnergyRetrieval

Retrieve associated pages according to questions. This dataset, Energy Fr, is a corpus of reports on energy supply in europe, intended for complex-document understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish.

**Dataset:** [`vidore/vidore_v3_energy_mteb_format`](https://huggingface.co/datasets/vidore/vidore_v3_energy_mteb_format) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/blog/QuentinJG/introducing-vidore-v3)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | deu, eng, fra, ita, por, ... (6) | Academic, Chemistry, Engineering | derived | created and machine-translated |



??? quote "Citation"


    ```bibtex

    @misc{mace2025vidorev3,
      author = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
      day = {5},
      howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}},
      journal = {Hugging Face Blog},
      month = {November},
      publisher = {Hugging Face},
      title = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
      year = {2025},
    }

    ```




#### Vidore3FinanceEnRetrieval

Retrieve associated pages according to questions. This task, Finance - EN, is a corpus of reports from american banking companies, intended for long-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.

**Dataset:** [`vidore/vidore_v3_finance_en_mteb_format`](https://huggingface.co/datasets/vidore/vidore_v3_finance_en_mteb_format) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/blog/QuentinJG/introducing-vidore-v3)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | deu, eng, fra, ita, por, ... (6) | Financial | derived | created and machine-translated |



??? quote "Citation"


    ```bibtex

    @misc{mace2025vidorev3,
      author = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
      day = {5},
      howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}},
      journal = {Hugging Face Blog},
      month = {November},
      publisher = {Hugging Face},
      title = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
      year = {2025},
    }

    ```




#### Vidore3FinanceFrRetrieval

Retrieve associated pages according to questions. This task, Finance - FR, is a corpus of reports from french companies in the luxury domain, intended for long-document understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish.

**Dataset:** [`vidore/vidore_v3_finance_fr_mteb_format`](https://huggingface.co/datasets/vidore/vidore_v3_finance_fr_mteb_format) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/blog/QuentinJG/introducing-vidore-v3)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | deu, eng, fra, ita, por, ... (6) | Financial | derived | created and machine-translated |



??? quote "Citation"


    ```bibtex

    @misc{mace2025vidorev3,
      author = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
      day = {5},
      howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}},
      journal = {Hugging Face Blog},
      month = {November},
      publisher = {Hugging Face},
      title = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
      year = {2025},
    }

    ```




#### Vidore3HrRetrieval

Retrieve associated pages according to questions. This dataset, HR, is a corpus of reports released by the european union, intended for complex-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.

**Dataset:** [`vidore/vidore_v3_hr_mteb_format`](https://huggingface.co/datasets/vidore/vidore_v3_hr_mteb_format) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/blog/QuentinJG/introducing-vidore-v3)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | deu, eng, fra, ita, por, ... (6) | Social | derived | created and machine-translated |



??? quote "Citation"


    ```bibtex

    @misc{mace2025vidorev3,
      author = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
      day = {5},
      howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}},
      journal = {Hugging Face Blog},
      month = {November},
      publisher = {Hugging Face},
      title = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
      year = {2025},
    }

    ```




#### Vidore3IndustrialRetrieval

Retrieve associated pages according to questions. This dataset, Industrial reports, is a corpus of technical documents on military aircraft (fueling, mechanics...), intended for complex-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.

**Dataset:** [`vidore/vidore_v3_industrial_mteb_format`](https://huggingface.co/datasets/vidore/vidore_v3_industrial_mteb_format) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/blog/QuentinJG/introducing-vidore-v3)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | deu, eng, fra, ita, por, ... (6) | Engineering | derived | created and machine-translated |



??? quote "Citation"


    ```bibtex

    @misc{mace2025vidorev3,
      author = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
      day = {5},
      howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}},
      journal = {Hugging Face Blog},
      month = {November},
      publisher = {Hugging Face},
      title = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
      year = {2025},
    }

    ```




#### Vidore3PharmaceuticalsRetrieval

Retrieve associated pages according to questions. This dataset, Pharmaceutical, is a corpus of slides from the FDA, intended for long-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.

**Dataset:** [`vidore/vidore_v3_pharmaceuticals_mteb_format`](https://huggingface.co/datasets/vidore/vidore_v3_pharmaceuticals_mteb_format) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/blog/QuentinJG/introducing-vidore-v3)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | deu, eng, fra, ita, por, ... (6) | Medical | derived | created and machine-translated |



??? quote "Citation"


    ```bibtex

    @misc{mace2025vidorev3,
      author = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
      day = {5},
      howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}},
      journal = {Hugging Face Blog},
      month = {November},
      publisher = {Hugging Face},
      title = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
      year = {2025},
    }

    ```




#### Vidore3PhysicsRetrieval

Retrieve associated pages according to questions. This dataset, Physics, is a corpus of course slides on french bachelor level physics lectures, intended for complex visual understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish.

**Dataset:** [`vidore/vidore_v3_physics_mteb_format`](https://huggingface.co/datasets/vidore/vidore_v3_physics_mteb_format) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/blog/QuentinJG/introducing-vidore-v3)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_10 | deu, eng, fra, ita, por, ... (6) | Academic, Engineering | derived | created and machine-translated |



??? quote "Citation"


    ```bibtex

    @misc{mace2025vidorev3,
      author = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
      day = {5},
      howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}},
      journal = {Hugging Face Blog},
      month = {November},
      publisher = {Hugging Face},
      title = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
      year = {2025},
    }

    ```




#### VidoreArxivQARetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/arxivqa_test_subsampled_beir`](https://huggingface.co/datasets/vidore/arxivqa_test_subsampled_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



??? quote "Citation"


    ```bibtex

    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }

    ```




#### VidoreDocVQARetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/docvqa_test_subsampled_beir`](https://huggingface.co/datasets/vidore/docvqa_test_subsampled_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



??? quote "Citation"


    ```bibtex

    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }

    ```




#### VidoreInfoVQARetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/infovqa_test_subsampled_beir`](https://huggingface.co/datasets/vidore/infovqa_test_subsampled_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



??? quote "Citation"


    ```bibtex

    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }

    ```




#### VidoreShiftProjectRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/shiftproject_test_beir`](https://huggingface.co/datasets/vidore/shiftproject_test_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



??? quote "Citation"


    ```bibtex

    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }

    ```




#### VidoreSyntheticDocQAAIRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/syntheticDocQA_artificial_intelligence_test_beir`](https://huggingface.co/datasets/vidore/syntheticDocQA_artificial_intelligence_test_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



??? quote "Citation"


    ```bibtex

    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }

    ```




#### VidoreSyntheticDocQAEnergyRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/syntheticDocQA_energy_test_beir`](https://huggingface.co/datasets/vidore/syntheticDocQA_energy_test_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



??? quote "Citation"


    ```bibtex

    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }

    ```




#### VidoreSyntheticDocQAGovernmentReportsRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/syntheticDocQA_government_reports_test_beir`](https://huggingface.co/datasets/vidore/syntheticDocQA_government_reports_test_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



??? quote "Citation"


    ```bibtex

    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }

    ```




#### VidoreSyntheticDocQAHealthcareIndustryRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/syntheticDocQA_healthcare_industry_test_beir`](https://huggingface.co/datasets/vidore/syntheticDocQA_healthcare_industry_test_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



??? quote "Citation"


    ```bibtex

    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }

    ```




#### VidoreTabfquadRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/tabfquad_test_subsampled_beir`](https://huggingface.co/datasets/vidore/tabfquad_test_subsampled_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



??? quote "Citation"


    ```bibtex

    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }

    ```




#### VidoreTatdqaRetrieval

Retrieve associated pages according to questions.

**Dataset:** [`vidore/tatdqa_test_beir`](https://huggingface.co/datasets/vidore/tatdqa_test_beir) • **License:** mit • [Learn more →](https://arxiv.org/pdf/2407.01449)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to image (t2i) | ndcg_at_5 | eng | Academic | derived | found |



??? quote "Citation"


    ```bibtex

    @article{faysse2024colpali,
      author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
      journal = {arXiv preprint arXiv:2407.01449},
      title = {ColPali: Efficient Document Retrieval with Vision Language Models},
      year = {2024},
    }

    ```
