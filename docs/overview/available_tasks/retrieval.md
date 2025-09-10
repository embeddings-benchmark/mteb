
# Retrieval

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 326 

#### AILACasedocs

The task is to retrieve the case document that most closely matches or is most relevant to the scenario described in the provided query.

**Dataset:** [`mteb/AILA_casedocs`](https://huggingface.co/datasets/mteb/AILA_casedocs) • **License:** cc-by-4.0 • [Learn more →](https://zenodo.org/records/4063986)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Legal, Written | derived | found |



#### AILAStatutes

This dataset is structured for the task of identifying the most relevant statutes for a given situation.

**Dataset:** [`mteb/AILA_statutes`](https://huggingface.co/datasets/mteb/AILA_statutes) • **License:** cc-by-4.0 • [Learn more →](https://zenodo.org/records/4063986)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Legal, Written | derived | found |



#### ARCChallenge

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on ARC-Challenge.

**Dataset:** [`RAR-b/ARC-Challenge`](https://huggingface.co/datasets/RAR-b/ARC-Challenge) • **License:** cc-by-sa-4.0 • [Learn more →](https://allenai.org/data/arc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### AlloprofRetrieval

This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum curated by a large number of teachers to students on all subjects taught from in primary and secondary school

**Dataset:** [`lyon-nlp/alloprof`](https://huggingface.co/datasets/lyon-nlp/alloprof) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/antoinelb7/alloprof)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fra | Encyclopaedic, Written | human-annotated | found |



#### AlphaNLI

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on AlphaNLI.

**Dataset:** [`RAR-b/alphanli`](https://huggingface.co/datasets/RAR-b/alphanli) • **License:** cc-by-nc-4.0 • [Learn more →](https://leaderboard.allenai.org/anli/submissions/get-started)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### AppsRetrieval

The dataset is a collection of natural language queries and their corresponding code snippets. The task is to retrieve the most relevant code snippet for a given query.

**Dataset:** [`CoIR-Retrieval/apps`](https://huggingface.co/datasets/CoIR-Retrieval/apps) • **License:** mit • [Learn more →](https://arxiv.org/abs/2105.09938)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng, python | Programming, Written | derived | found |



#### ArguAna

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval

**Dataset:** [`mteb/arguana`](https://huggingface.co/datasets/mteb/arguana) • **License:** cc-by-sa-4.0 • [Learn more →](http://argumentation.bplaced.net/arguana/data)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical, Written | not specified | not specified |



#### ArguAna-Fa

ArguAna-Fa

**Dataset:** [`MCINext/arguana-fa`](https://huggingface.co/datasets/MCINext/arguana-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/arguana-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Blog | derived | found |



#### ArguAna-NL

ArguAna involves the task of retrieval of the best counterargument to an argument. ArguAna-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-arguana`](https://huggingface.co/datasets/clips/beir-nl-arguana) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-arguana)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### ArguAna-PL

ArguAna-PL

**Dataset:** [`mteb/ArguAna-PL`](https://huggingface.co/datasets/mteb/ArguAna-PL) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clarin-knext/arguana-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Medical, Written | not specified | not specified |



#### ArguAna-VN

A translated dataset from NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/arguana-vn`](https://huggingface.co/datasets/GreenNode/arguana-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://argumentation.bplaced.net/arguana/data)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Medical, Written | derived | machine-translated and LM verified |



#### AutoRAGRetrieval

This dataset enables the evaluation of Korean RAG performance across various domains—finance, public sector, healthcare, legal, and commerce—by providing publicly accessible documents, questions, and answers.

**Dataset:** [`yjoonjang/markers_bm`](https://huggingface.co/datasets/yjoonjang/markers_bm) • **License:** mit • [Learn more →](https://arxiv.org/abs/2410.20878)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | kor | Financial, Government, Legal, Medical, Social | human-annotated | created |



#### BIRCO-ArguAna

Retrieval task using the ArguAna dataset from BIRCO. This dataset contains 100 queries where both queries and passages are complex one-paragraph arguments about current affairs. The objective is to retrieve the counter-argument that directly refutes the query’s stance.

**Dataset:** [`mteb/BIRCO-ArguAna-Test`](https://huggingface.co/datasets/mteb/BIRCO-ArguAna-Test) • **License:** cc-by-4.0 • [Learn more →](https://github.com/BIRCO-benchmark/BIRCO)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Written | expert-annotated | found |



#### BIRCO-ClinicalTrial

Retrieval task using the Clinical-Trial dataset from BIRCO. This dataset contains 50 queries that are patient case reports. Each query has a candidate pool comprising 30-110 clinical trial descriptions. Relevance is graded (0, 1, 2), where 1 and 2 are considered relevant.

**Dataset:** [`mteb/BIRCO-ClinicalTrial-Test`](https://huggingface.co/datasets/mteb/BIRCO-ClinicalTrial-Test) • **License:** cc-by-4.0 • [Learn more →](https://github.com/BIRCO-benchmark/BIRCO)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | expert-annotated | found |



#### BIRCO-DorisMae

Retrieval task using the DORIS-MAE dataset from BIRCO. This dataset contains 60 queries that are complex research questions from computer scientists. Each query has a candidate pool of approximately 110 abstracts. Relevance is graded from 0 to 2 (scores of 1 and 2 are considered relevant).

**Dataset:** [`mteb/BIRCO-DorisMae-Test`](https://huggingface.co/datasets/mteb/BIRCO-DorisMae-Test) • **License:** cc-by-4.0 • [Learn more →](https://github.com/BIRCO-benchmark/BIRCO)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic | expert-annotated | found |



#### BIRCO-Relic

Retrieval task using the RELIC dataset from BIRCO. This dataset contains 100 queries which are excerpts from literary analyses with a missing quotation (indicated by [masked sentence(s)]). Each query has a candidate pool of 50 passages. The objective is to retrieve the passage that best completes the literary analysis.

**Dataset:** [`mteb/BIRCO-Relic-Test`](https://huggingface.co/datasets/mteb/BIRCO-Relic-Test) • **License:** cc-by-4.0 • [Learn more →](https://github.com/BIRCO-benchmark/BIRCO)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Fiction | expert-annotated | found |



#### BIRCO-WTB

Retrieval task using the WhatsThatBook dataset from BIRCO. This dataset contains 100 queries where each query is an ambiguous description of a book. Each query has a candidate pool of 50 book descriptions. The objective is to retrieve the correct book description.

**Dataset:** [`mteb/BIRCO-WTB-Test`](https://huggingface.co/datasets/mteb/BIRCO-WTB-Test) • **License:** cc-by-4.0 • [Learn more →](https://github.com/BIRCO-benchmark/BIRCO)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Fiction | expert-annotated | found |



#### BSARDRetrieval

The Belgian Statutory Article Retrieval Dataset (BSARD) is a French native dataset for studying legal information retrieval. BSARD consists of more than 22,600 statutory articles from Belgian law and about 1,100 legal questions posed by Belgian citizens and labeled by experienced jurists with relevant articles from the corpus.

**Dataset:** [`maastrichtlawtech/bsard`](https://huggingface.co/datasets/maastrichtlawtech/bsard) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/maastrichtlawtech/bsard)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_100 | fra | Legal, Spoken | expert-annotated | found |



#### BSARDRetrieval.v2

BSARD is a French native dataset for legal information retrieval. BSARDRetrieval.v2 covers multi-article queries, fixing issues (#2906) with the previous data loading. 

**Dataset:** [`maastrichtlawtech/bsard`](https://huggingface.co/datasets/maastrichtlawtech/bsard) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/maastrichtlawtech/bsard)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | recall_at_100 | fra | Legal, Spoken | expert-annotated | found |



#### BarExamQA

A benchmark for retrieving legal provisions that answer US bar exam questions.

**Dataset:** [`isaacus/mteb-barexam-qa`](https://huggingface.co/datasets/isaacus/mteb-barexam-qa) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/reglab/barexam_qa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Legal | expert-annotated | found |



#### BelebeleRetrieval

Belebele is a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants (including 115 distinct languages and their scripts)

**Dataset:** [`facebook/belebele`](https://huggingface.co/datasets/facebook/belebele) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2308.16884)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | acm, afr, als, amh, apc, ... (115) | News, Web, Written | expert-annotated | created |



#### BillSumCA

A benchmark for retrieving Californian bills based on their summaries.

**Dataset:** [`isaacus/mteb-BillSumCA`](https://huggingface.co/datasets/isaacus/mteb-BillSumCA) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/FiscalNote/billsum)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Government, Legal | expert-annotated | found |



#### BillSumUS

A benchmark for retrieving US federal bills based on their summaries.

**Dataset:** [`isaacus/mteb-BillSumUS`](https://huggingface.co/datasets/isaacus/mteb-BillSumUS) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/FiscalNote/billsum)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Government, Legal | expert-annotated | found |



#### BrightLongRetrieval

Bright retrieval dataset with long documents.

**Dataset:** [`xlangai/BRIGHT`](https://huggingface.co/datasets/xlangai/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Non-fiction, Written | derived | found |



#### BrightRetrieval

Bright retrieval dataset.

**Dataset:** [`xlangai/BRIGHT`](https://huggingface.co/datasets/xlangai/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Non-fiction, Written | derived | found |



#### BuiltBenchRetrieval

Retrieval of built asset entity type/class descriptions given a query describing an entity as represented in well-established industry classification systems such as Uniclass, IFC, etc.

**Dataset:** [`mteb/BuiltBenchRetrieval`](https://huggingface.co/datasets/mteb/BuiltBenchRetrieval) • **License:** cc-by-nd-4.0 • [Learn more →](https://arxiv.org/abs/2411.12056)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Engineering, Written | derived | created |



#### COIRCodeSearchNetRetrieval

The dataset is a collection of code snippets and their corresponding natural language queries. The task is to retrieve the most relevant code summary given a code snippet.

**Dataset:** [`CoIR-Retrieval/CodeSearchNet`](https://huggingface.co/datasets/CoIR-Retrieval/CodeSearchNet) • **License:** mit • [Learn more →](https://huggingface.co/datasets/code_search_net/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | go, java, javascript, php, python, ... (6) | Programming, Written | derived | found |



#### CQADupstack-Android-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Android-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Android-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-android-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Non-fiction, Programming, Web, Written | derived | machine-translated |



#### CQADupstack-English-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-English-PL`](https://huggingface.co/datasets/mteb/CQADupstack-English-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-english-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Written | derived | machine-translated |



#### CQADupstack-Gaming-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Gaming-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Gaming-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-gaming-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web, Written | derived | machine-translated |



#### CQADupstack-Gis-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Gis-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Gis-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-gis-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic, Non-fiction, Written | derived | machine-translated |



#### CQADupstack-Mathematica-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Mathematica-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Mathematica-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-mathematica-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic, Non-fiction, Written | derived | machine-translated |



#### CQADupstack-Physics-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Physics-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Physics-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-physics-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic, Non-fiction, Written | derived | machine-translated |



#### CQADupstack-Programmers-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Programmers-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Programmers-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-programmers-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Non-fiction, Programming, Written | derived | machine-translated |



#### CQADupstack-Stats-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Stats-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Stats-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-stats-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic, Non-fiction, Written | derived | machine-translated |



#### CQADupstack-Tex-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Tex-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Tex-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-tex-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Non-fiction, Written | derived | machine-translated |



#### CQADupstack-Unix-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Unix-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Unix-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-unix-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Programming, Web, Written | derived | machine-translated |



#### CQADupstack-Webmasters-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Webmasters-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Webmasters-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-webmasters-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web, Written | derived | machine-translated |



#### CQADupstack-Wordpress-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Wordpress-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Wordpress-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-wordpress-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Programming, Web, Written | derived | machine-translated |



#### CQADupstackAndroid-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackAndroid-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-android-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-android-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Non-fiction, Programming, Web, Written | derived | machine-translated and LM verified |



#### CQADupstackAndroidRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-android`](https://huggingface.co/datasets/mteb/cqadupstack-android) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Non-fiction, Programming, Web, Written | derived | found |



#### CQADupstackAndroidRetrieval-Fa

CQADupstackAndroidRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-android-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-android-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-android-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackEnglish-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackEnglishRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-english`](https://huggingface.co/datasets/mteb/cqadupstack-english) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Written | derived | found |



#### CQADupstackEnglishRetrieval-Fa

CQADupstackEnglishRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-english-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-english-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-english-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackGaming-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackGamingRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-gaming`](https://huggingface.co/datasets/mteb/cqadupstack-gaming) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Web, Written | derived | found |



#### CQADupstackGamingRetrieval-Fa

CQADupstackGamingRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-gaming-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-gaming-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-gaming-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackGis-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackGis-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-gis-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-gis-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Non-fiction, Written | derived | machine-translated and LM verified |



#### CQADupstackGisRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-gis`](https://huggingface.co/datasets/mteb/cqadupstack-gis) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Non-fiction, Written | derived | found |



#### CQADupstackGisRetrieval-Fa

CQADupstackGisRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-gis-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-gis-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-gis-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackMathematica-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackMathematica-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-mathematica-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-mathematica-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified |



#### CQADupstackMathematicaRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-mathematica`](https://huggingface.co/datasets/mteb/cqadupstack-mathematica) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Non-fiction, Written | derived | found |



#### CQADupstackMathematicaRetrieval-Fa

CQADupstackMathematicaRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-mathematica-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-mathematica-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-mathematica-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackPhysics-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackPhysics-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-physics-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-physics-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified |



#### CQADupstackPhysicsRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-physics`](https://huggingface.co/datasets/mteb/cqadupstack-physics) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Non-fiction, Written | derived | found |



#### CQADupstackPhysicsRetrieval-Fa

CQADupstackPhysicsRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-physics-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-physics-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-physics-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackProgrammers-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackProgrammers-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-programmers-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-programmers-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Non-fiction, Programming, Written | derived | machine-translated and LM verified |



#### CQADupstackProgrammersRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-programmers`](https://huggingface.co/datasets/mteb/cqadupstack-programmers) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Non-fiction, Programming, Written | derived | found |



#### CQADupstackProgrammersRetrieval-Fa

CQADupstackProgrammersRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-programmers-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-programmers-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-programmers-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackStats-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackStats-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-stats-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-stats-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified |



#### CQADupstackStatsRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-stats`](https://huggingface.co/datasets/mteb/cqadupstack-stats) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Non-fiction, Written | derived | found |



#### CQADupstackStatsRetrieval-Fa

CQADupstackStatsRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-stats-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-stats-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-stats-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackTex-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackTex-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-tex-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-tex-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Non-fiction, Written | derived | machine-translated and LM verified |



#### CQADupstackTexRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-tex`](https://huggingface.co/datasets/mteb/cqadupstack-tex) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Non-fiction, Written | derived | found |



#### CQADupstackTexRetrieval-Fa

CQADupstackTexRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-tex-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-tex-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-tex-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackUnix-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackUnix-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-unix-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-unix-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Programming, Web, Written | derived | machine-translated and LM verified |



#### CQADupstackUnixRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-unix`](https://huggingface.co/datasets/mteb/cqadupstack-unix) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Programming, Web, Written | derived | found |



#### CQADupstackUnixRetrieval-Fa

CQADupstackUnixRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-unix-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-unix-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-unix-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackWebmasters-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackWebmasters-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-webmasters-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-webmasters-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Web, Written | derived | machine-translated and LM verified |



#### CQADupstackWebmastersRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-webmasters`](https://huggingface.co/datasets/mteb/cqadupstack-webmasters) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Web, Written | derived | found |



#### CQADupstackWebmastersRetrieval-Fa

CQADupstackWebmastersRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-webmasters-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-webmasters-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-webmasters-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CQADupstackWordpress-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### CQADupstackWordpress-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-wordpress-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-wordpress-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Programming, Web, Written | derived | machine-translated and LM verified |



#### CQADupstackWordpressRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-wordpress`](https://huggingface.co/datasets/mteb/cqadupstack-wordpress) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Programming, Web, Written | derived | found |



#### CQADupstackWordpressRetrieval-Fa

CQADupstackWordpressRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-wordpress-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-wordpress-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-wordpress-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### CUREv1

Collection of query-passage pairs curated by medical professionals, across 10 disciplines and 3 cross-lingual settings.

**Dataset:** [`clinia/CUREv1`](https://huggingface.co/datasets/clinia/CUREv1) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/clinia/CUREv1)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng, fra, spa | Academic, Medical, Written | expert-annotated | created |



#### ChatDoctorRetrieval

A medical retrieval task based on ChatDoctor_HealthCareMagic dataset containing 112,000 real-world medical question-and-answer pairs. Each query is a medical question from patients (e.g., 'What are the symptoms of diabetes?'), and the corpus contains medical responses and healthcare information. The task is to retrieve the correct medical information that answers the patient's question. The dataset includes grammatical inconsistencies which help separate strong healthcare retrieval models from weak ones. Queries are patient medical questions while the corpus contains relevant medical responses, diagnoses, and treatment information from healthcare professionals.

**Dataset:** [`embedding-benchmark/ChatDoctor_HealthCareMagic`](https://huggingface.co/datasets/embedding-benchmark/ChatDoctor_HealthCareMagic) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/ChatDoctor_HealthCareMagic)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng | Medical | expert-annotated | found |



#### ChemHotpotQARetrieval

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/ChemHotpotQARetrieval`](https://huggingface.co/datasets/BASF-AI/ChemHotpotQARetrieval) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Chemistry | derived | found |



#### ChemNQRetrieval

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/ChemNQRetrieval`](https://huggingface.co/datasets/BASF-AI/ChemNQRetrieval) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Chemistry | derived | found |



#### ClimateFEVER

CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims (queries) regarding climate-change. The underlying corpus is the same as FVER.

**Dataset:** [`mteb/climate-fever`](https://huggingface.co/datasets/mteb/climate-fever) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



#### ClimateFEVER-Fa

ClimateFEVER-Fa

**Dataset:** [`MCINext/climate-fever-fa`](https://huggingface.co/datasets/MCINext/climate-fever-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/climate-fever-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### ClimateFEVER-NL

CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change. ClimateFEVER-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-climate-fever`](https://huggingface.co/datasets/clips/beir-nl-climate-fever) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-climate-fever)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Encyclopaedic, Written | derived | machine-translated and verified |



#### ClimateFEVER-VN

A translated dataset from CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/climate-fever-vn`](https://huggingface.co/datasets/GreenNode/climate-fever-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Encyclopaedic, Written | derived | machine-translated and LM verified |



#### ClimateFEVER.v2

CLIMATE-FEVER is a dataset following the FEVER methodology, containing 1,535 real-world climate change claims. This updated version addresses corpus mismatches and qrel inconsistencies in MTEB, restoring labels while refining corpus-query alignment for better accuracy. 

**Dataset:** [`mteb/climate-fever-v2`](https://huggingface.co/datasets/mteb/climate-fever-v2) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Written | human-annotated | found |



#### ClimateFEVERHardNegatives

CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/ClimateFEVER_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/ClimateFEVER_test_top_250_only_w_correct-v2) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



#### CmedqaRetrieval

Online medical consultation text. Used the CMedQAv2 as its underlying dataset.

**Dataset:** [`mteb/CmedqaRetrieval`](https://huggingface.co/datasets/mteb/CmedqaRetrieval) • **License:** not specified • [Learn more →](https://aclanthology.org/2022.emnlp-main.357.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | Medical, Written | not specified | not specified |



#### CodeEditSearchRetrieval

The dataset is a collection of unified diffs of code changes, paired with a short instruction that describes the change. The dataset is derived from the CommitPackFT dataset.

**Dataset:** [`cassanof/CodeEditSearch`](https://huggingface.co/datasets/cassanof/CodeEditSearch) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/cassanof/CodeEditSearch/viewer)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | c, c++, go, java, javascript, ... (13) | Programming, Written | derived | found |



#### CodeFeedbackMT

The dataset is a collection of user queries and assistant responses. The task is to retrieve the most relevant response for a given query.

**Dataset:** [`CoIR-Retrieval/codefeedback-mt`](https://huggingface.co/datasets/CoIR-Retrieval/codefeedback-mt) • **License:** mit • [Learn more →](https://arxiv.org/abs/2402.14658)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Programming, Written | derived | found |



#### CodeFeedbackST

The dataset is a collection of user queries and assistant responses. The task is to retrieve the most relevant response for a given query.

**Dataset:** [`CoIR-Retrieval/codefeedback-st`](https://huggingface.co/datasets/CoIR-Retrieval/codefeedback-st) • **License:** mit • [Learn more →](https://arxiv.org/abs/2407.02883)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Programming, Written | derived | found |



#### CodeSearchNetCCRetrieval

The dataset is a collection of code snippets. The task is to retrieve the most relevant code snippet for a given code snippet.

**Dataset:** [`CoIR-Retrieval/CodeSearchNet-ccr`](https://huggingface.co/datasets/CoIR-Retrieval/CodeSearchNet-ccr) • **License:** mit • [Learn more →](https://arxiv.org/abs/2407.02883)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | go, java, javascript, php, python, ... (6) | Programming, Written | derived | found |



#### CodeSearchNetRetrieval

The dataset is a collection of code snippets and their corresponding natural language queries. The task is to retrieve the most relevant code snippet for a given query.

**Dataset:** [`code-search-net/code_search_net`](https://huggingface.co/datasets/code-search-net/code_search_net) • **License:** mit • [Learn more →](https://huggingface.co/datasets/code_search_net/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | go, java, javascript, php, python, ... (6) | Programming, Written | derived | found |



#### CodeTransOceanContest

The dataset is a collection of code snippets and their corresponding natural language queries. The task is to retrieve the most relevant code snippet

**Dataset:** [`CoIR-Retrieval/codetrans-contest`](https://huggingface.co/datasets/CoIR-Retrieval/codetrans-contest) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2310.04951)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | c++, python | Programming, Written | derived | found |



#### CodeTransOceanDL

The dataset is a collection of equivalent Python Deep Learning code snippets written in different machine learning framework. The task is to retrieve the equivalent code snippet in another framework, given a query code snippet from one framework.

**Dataset:** [`CoIR-Retrieval/codetrans-dl`](https://huggingface.co/datasets/CoIR-Retrieval/codetrans-dl) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2310.04951)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | python | Programming, Written | derived | found |



#### CosQA

The dataset is a collection of natural language queries and their corresponding code snippets. The task is to retrieve the most relevant code snippet for a given query.

**Dataset:** [`CoIR-Retrieval/cosqa`](https://huggingface.co/datasets/CoIR-Retrieval/cosqa) • **License:** mit • [Learn more →](https://arxiv.org/abs/2105.13239)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng, python | Programming, Written | derived | found |



#### CovidRetrieval

COVID-19 news articles

**Dataset:** [`mteb/CovidRetrieval`](https://huggingface.co/datasets/mteb/CovidRetrieval) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2203.03367)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | Entertainment, Medical | human-annotated | not specified |



#### CrossLingualSemanticDiscriminationWMT19

Evaluate a multilingual embedding model based on its ability to discriminate against the original parallel pair against challenging distractors - spawning from WMT19 DE-FR test set

**Dataset:** [`Andrianos/clsd_wmt19_21`](https://huggingface.co/datasets/Andrianos/clsd_wmt19_21) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/Andrianos/clsd_wmt19_21)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_1 | deu, fra | News, Written | derived | LM-generated and verified |



#### CrossLingualSemanticDiscriminationWMT21

Evaluate a multilingual embedding model based on its ability to discriminate against the original parallel pair against challenging distractors - spawning from WMT21 DE-FR test set

**Dataset:** [`Andrianos/clsd_wmt19_21`](https://huggingface.co/datasets/Andrianos/clsd_wmt19_21) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/Andrianos/clsd_wmt19_21)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_1 | deu, fra | News, Written | derived | LM-generated and verified |



#### DBPedia

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base

**Dataset:** [`mteb/dbpedia`](https://huggingface.co/datasets/mteb/dbpedia) • **License:** mit • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### DBPedia-Fa

DBPedia-Fa

**Dataset:** [`MCINext/dbpedia-fa`](https://huggingface.co/datasets/MCINext/dbpedia-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/dbpedia-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Encyclopaedic | derived | found |



#### DBPedia-NL

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base. DBPedia-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-dbpedia-entity`](https://huggingface.co/datasets/clips/beir-nl-dbpedia-entity) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-dbpedia-entity)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Encyclopaedic, Written | derived | machine-translated and verified |



#### DBPedia-PL

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base

**Dataset:** [`mteb/DBPedia-PL`](https://huggingface.co/datasets/mteb/DBPedia-PL) • **License:** mit • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Encyclopaedic, Written | derived | machine-translated |



#### DBPedia-PLHardNegatives

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/DBPedia_PL_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/DBPedia_PL_test_top_250_only_w_correct-v2) • **License:** mit • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Encyclopaedic, Written | derived | machine-translated |



#### DBPedia-VN

A translated dataset from DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/dbpedia-vn`](https://huggingface.co/datasets/GreenNode/dbpedia-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Encyclopaedic, Written | derived | machine-translated and LM verified |



#### DBPediaHardNegatives

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/DBPedia_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/DBPedia_test_top_250_only_w_correct-v2) • **License:** mit • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### DS1000Retrieval

A code retrieval task based on 1,000 data science programming problems from DS-1000. Each query is a natural language description of a data science task (e.g., 'Create a scatter plot of column A vs column B with matplotlib'), and the corpus contains Python code implementations using libraries like pandas, numpy, matplotlib, scikit-learn, and scipy. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains Python function implementations focused on data science workflows.

**Dataset:** [`embedding-benchmark/DS1000`](https://huggingface.co/datasets/embedding-benchmark/DS1000) • **License:** mit • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/DS1000)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng, python | Programming | expert-annotated | found |



#### DanFEVER

A Danish dataset intended for misinformation research. It follows the same format as the English FEVER dataset.

**Dataset:** [`strombergnlp/danfever`](https://huggingface.co/datasets/strombergnlp/danfever) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2021.nodalida-main.47/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | dan | Encyclopaedic, Non-fiction, Spoken | human-annotated | found |



#### DanFeverRetrieval

A Danish dataset intended for misinformation research. It follows the same format as the English FEVER dataset. DanFeverRetrieval fixed an issue in DanFever where some corpus entries were incorrectly removed.

**Dataset:** [`strombergnlp/danfever`](https://huggingface.co/datasets/strombergnlp/danfever) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2021.nodalida-main.47/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | dan | Encyclopaedic, Non-fiction, Spoken | human-annotated | found |



#### DuRetrieval

A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine

**Dataset:** [`mteb/DuRetrieval`](https://huggingface.co/datasets/mteb/DuRetrieval) • **License:** not specified • [Learn more →](https://aclanthology.org/2022.emnlp-main.357.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | not specified | not specified | not specified |



#### EcomRetrieval

EcomRetrieval

**Dataset:** [`mteb/EcomRetrieval`](https://huggingface.co/datasets/mteb/EcomRetrieval) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2203.03367)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | not specified | not specified | not specified |



#### EstQA

EstQA is an Estonian question answering dataset based on Wikipedia.

**Dataset:** [`kardosdrur/estonian-qa`](https://huggingface.co/datasets/kardosdrur/estonian-qa) • **License:** not specified • [Learn more →](https://www.semanticscholar.org/paper/Extractive-Question-Answering-for-Estonian-Language-182912IAPM-Alum%C3%A4e/ea4f60ab36cadca059c880678bc4c51e293a85d6?utm_source=direct_link)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | est | Encyclopaedic, Written | human-annotated | found |



#### FEVER

FEVER (Fact Extraction and VERification) consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from.

**Dataset:** [`mteb/fever`](https://huggingface.co/datasets/mteb/fever) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://fever.ai/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



#### FEVER-NL

FEVER (Fact Extraction and VERification) consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. FEVER-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-fever`](https://huggingface.co/datasets/clips/beir-nl-fever) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-fever)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Encyclopaedic, Written | derived | machine-translated and verified |



#### FEVER-VN

A translated dataset from FEVER (Fact Extraction and VERification) consists of 185,445 claims generated by altering sentences
            extracted from Wikipedia and subsequently verified without knowledge of the sentence they were
            derived from.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/fever-vn`](https://huggingface.co/datasets/GreenNode/fever-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://fever.ai/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Encyclopaedic, Written | derived | machine-translated and LM verified |



#### FEVERHardNegatives

FEVER (Fact Extraction and VERification) consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/FEVER_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/FEVER_test_top_250_only_w_correct-v2) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://fever.ai/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | not specified |



#### FQuADRetrieval

This dataset has been built from the French SQuad dataset.

**Dataset:** [`manu/fquad2_test`](https://huggingface.co/datasets/manu/fquad2_test) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/manu/fquad2_test)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fra | Encyclopaedic, Written | human-annotated | created |



#### FaithDial

FaithDial is a faithful knowledge-grounded dialogue benchmark.It was curated by asking annotators to amend hallucinated utterances in Wizard of Wikipedia (WoW). It consists of conversation histories along with manually labelled relevant passage. For the purpose of retrieval, we only consider the instances marked as 'Edification' in the VRM field, as the gold passage associated with these instances is non-ambiguous.

**Dataset:** [`McGill-NLP/FaithDial`](https://huggingface.co/datasets/McGill-NLP/FaithDial) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://mcgill-nlp.github.io/FaithDial)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



#### FeedbackQARetrieval

Using Interactive Feedback to Improve the Accuracy and Explainability of Question Answering Systems Post-Deployment

**Dataset:** [`lt2c/fqa`](https://huggingface.co/datasets/lt2c/fqa) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2204.03025)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | precision_at_1 | eng | Government, Medical, Web, Written | human-annotated | created |



#### FiQA-PL

Financial Opinion Mining and Question Answering

**Dataset:** [`mteb/FiQA-PL`](https://huggingface.co/datasets/mteb/FiQA-PL) • **License:** not specified • [Learn more →](https://sites.google.com/view/fiqa/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Financial, Written | human-annotated | found |



#### FiQA2018

Financial Opinion Mining and Question Answering

**Dataset:** [`mteb/fiqa`](https://huggingface.co/datasets/mteb/fiqa) • **License:** not specified • [Learn more →](https://sites.google.com/view/fiqa/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Financial, Written | human-annotated | found |



#### FiQA2018-Fa

FiQA2018-Fa

**Dataset:** [`MCINext/fiqa-fa`](https://huggingface.co/datasets/MCINext/fiqa-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/fiqa-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### FiQA2018-NL

Financial Opinion Mining and Question Answering. FiQA2018-NL is a Dutch translation

**Dataset:** [`clips/beir-nl-fiqa`](https://huggingface.co/datasets/clips/beir-nl-fiqa) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-fiqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



#### FiQA2018-VN

A translated dataset from Financial Opinion Mining and Question Answering
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/fiqa-vn`](https://huggingface.co/datasets/GreenNode/fiqa-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://sites.google.com/view/fiqa/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Financial, Written | derived | machine-translated and LM verified |



#### FinQARetrieval

A financial retrieval task based on FinQA dataset containing numerical reasoning questions over financial documents. Each query is a financial question requiring numerical computation (e.g., 'What is the percentage change in operating expenses from 2019 to 2020?'), and the corpus contains financial document text with tables and numerical data. The task is to retrieve the correct financial information that enables answering the numerical question. Queries are numerical reasoning questions while the corpus contains financial text passages with embedded tables, figures, and quantitative financial data from earnings reports.

**Dataset:** [`embedding-benchmark/FinQA`](https://huggingface.co/datasets/embedding-benchmark/FinQA) • **License:** mit • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/FinQA)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng | Financial | expert-annotated | found |



#### FinanceBenchRetrieval

A financial retrieval task based on FinanceBench dataset containing financial questions and answers. Each query is a financial question (e.g., 'What was the total revenue in Q3 2023?'), and the corpus contains financial document excerpts and annual reports. The task is to retrieve the correct financial information that answers the question. Queries are financial questions while the corpus contains relevant excerpts from financial documents, earnings reports, and SEC filings with detailed financial data and metrics.

**Dataset:** [`embedding-benchmark/FinanceBench`](https://huggingface.co/datasets/embedding-benchmark/FinanceBench) • **License:** mit • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/FinanceBench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng | Financial | expert-annotated | found |



#### FreshStackRetrieval

A code retrieval task based on FreshStack dataset containing programming problems across multiple languages. Each query is a natural language description of a programming task (e.g., 'Write a function to reverse a string using recursion'), and the corpus contains code implementations in Python, JavaScript, and Go. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains function implementations with proper syntax and logic across different programming languages.

**Dataset:** [`embedding-benchmark/FreshStack_mteb`](https://huggingface.co/datasets/embedding-benchmark/FreshStack_mteb) • **License:** mit • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/FreshStack_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng, go, javascript, python | Programming | expert-annotated | found |



#### GeorgianFAQRetrieval

Frequently asked questions (FAQs) and answers mined from Georgian websites via Common Crawl.

**Dataset:** [`jupyterjazz/georgian-faq`](https://huggingface.co/datasets/jupyterjazz/georgian-faq) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jupyterjazz/georgian-faq)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | kat | Web, Written | derived | created |



#### GerDaLIR

GerDaLIR is a legal information retrieval dataset created from the Open Legal Data platform.

**Dataset:** [`jinaai/ger_da_lir`](https://huggingface.co/datasets/jinaai/ger_da_lir) • **License:** not specified • [Learn more →](https://github.com/lavis-nlp/GerDaLIR)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | deu | Legal | not specified | not specified |



#### GerDaLIRSmall

The dataset consists of documents, passages and relevance labels in German. In contrast to the original dataset, only documents that have corresponding queries in the query set are chosen to create a smaller corpus for evaluation purposes.

**Dataset:** [`mteb/GerDaLIRSmall`](https://huggingface.co/datasets/mteb/GerDaLIRSmall) • **License:** mit • [Learn more →](https://github.com/lavis-nlp/GerDaLIR)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | deu | Legal, Written | derived | found |



#### GermanDPR

GermanDPR is a German Question Answering dataset for open-domain QA. It associates questions with a textual context containing the answer

**Dataset:** [`deepset/germandpr`](https://huggingface.co/datasets/deepset/germandpr) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/deepset/germandpr)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | deu | Non-fiction, Web, Written | human-annotated | found |



#### GermanGovServiceRetrieval

LHM-Dienstleistungen-QA is a German question answering dataset for government services of the Munich city administration. It associates questions with a textual context containing the answer

**Dataset:** [`it-at-m/LHM-Dienstleistungen-QA`](https://huggingface.co/datasets/it-at-m/LHM-Dienstleistungen-QA) • **License:** mit • [Learn more →](https://huggingface.co/datasets/it-at-m/LHM-Dienstleistungen-QA)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_5 | deu | Government, Written | derived | found |



#### GermanQuAD-Retrieval

Context Retrieval for German Question Answering

**Dataset:** [`mteb/germanquad-retrieval`](https://huggingface.co/datasets/mteb/germanquad-retrieval) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/deepset/germanquad)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | mrr_at_5 | deu | Non-fiction, Web, Written | human-annotated | found |



#### GovReport

A dataset for evaluating the ability of information retrieval models to retrieve lengthy US government reports from their summaries.

**Dataset:** [`isaacus/mteb-GovReport`](https://huggingface.co/datasets/isaacus/mteb-GovReport) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/launch/gov_report)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Government, Legal | expert-annotated | found |



#### GreekCivicsQA

This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum curated by a large number of teachers to students on all subjects taught from in primary and secondary school

**Dataset:** [`ilsp/greek_civics_qa`](https://huggingface.co/datasets/ilsp/greek_civics_qa) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/antoinelb7/alloprof)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ell | Academic, Written | derived | found |



#### GreenNodeTableMarkdownRetrieval

GreenNodeTable documents

**Dataset:** [`GreenNode/GreenNode-Table-Markdown-Retrieval-VN`](https://huggingface.co/datasets/GreenNode/GreenNode-Table-Markdown-Retrieval-VN) • **License:** mit • [Learn more →](https://huggingface.co/GreenNode)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Encyclopaedic, Financial, Non-fiction | human-annotated | found |



#### HC3FinanceRetrieval

A financial retrieval task based on HC3 Finance dataset containing human vs AI-generated financial text detection. Each query is a financial question or prompt (e.g., 'Explain the impact of interest rate changes on bond prices'), and the corpus contains both human-written and AI-generated financial responses. The task is to retrieve the most relevant and accurate financial content that addresses the query. Queries are financial questions while the corpus contains detailed financial explanations, analysis, and educational content covering various financial concepts and market dynamics.

**Dataset:** [`embedding-benchmark/HC3Finance`](https://huggingface.co/datasets/embedding-benchmark/HC3Finance) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/HC3Finance)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng | Financial | expert-annotated | found |



#### HagridRetrieval

HAGRID (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset)is a dataset for generative information-seeking scenarios. It consists of queriesalong with a set of manually labelled relevant passages

**Dataset:** [`miracl/hagrid`](https://huggingface.co/datasets/miracl/hagrid) • **License:** apache-2.0 • [Learn more →](https://github.com/project-miracl/hagrid)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | expert-annotated | found |



#### HellaSwag

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on HellaSwag.

**Dataset:** [`RAR-b/hellaswag`](https://huggingface.co/datasets/RAR-b/hellaswag) • **License:** mit • [Learn more →](https://rowanzellers.com/hellaswag/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### HotpotQA

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.

**Dataset:** [`mteb/hotpotqa`](https://huggingface.co/datasets/mteb/hotpotqa) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Web, Written | human-annotated | found |



#### HotpotQA-Fa

HotpotQA-Fa

**Dataset:** [`MCINext/hotpotqa-fa`](https://huggingface.co/datasets/MCINext/hotpotqa-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/hotpotqa-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Encyclopaedic | derived | found |



#### HotpotQA-NL

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strongsupervision for supporting facts to enable more explainable question answering systems. HotpotQA-NL is a Dutch translation. 

**Dataset:** [`clips/beir-nl-hotpotqa`](https://huggingface.co/datasets/clips/beir-nl-hotpotqa) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Web, Written | derived | machine-translated and verified |



#### HotpotQA-PL

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.

**Dataset:** [`mteb/HotpotQA-PL`](https://huggingface.co/datasets/mteb/HotpotQA-PL) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web, Written | derived | machine-translated |



#### HotpotQA-PLHardNegatives

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/HotpotQA_PL_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/HotpotQA_PL_test_top_250_only_w_correct-v2) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web, Written | derived | machine-translated |



#### HotpotQA-VN

A translated dataset from HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong
            supervision for supporting facts to enable more explainable question answering systems.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/hotpotqa-vn`](https://huggingface.co/datasets/GreenNode/hotpotqa-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Web, Written | derived | machine-translated and LM verified |



#### HotpotQAHardNegatives

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.  The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/HotpotQA_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/HotpotQA_test_top_250_only_w_correct-v2) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Web, Written | human-annotated | found |



#### HumanEvalRetrieval

A code retrieval task based on 164 Python programming problems from HumanEval. Each query is a natural language description of a programming task (e.g., 'Check if in given list of numbers, are any two numbers closer to each other than given threshold'), and the corpus contains Python code implementations. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains Python function implementations with proper indentation and logic.

**Dataset:** [`embedding-benchmark/HumanEval`](https://huggingface.co/datasets/embedding-benchmark/HumanEval) • **License:** mit • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/HumanEval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng, python | Programming | derived | found |



#### HunSum2AbstractiveRetrieval

HunSum-2-abstractive is a Hungarian dataset containing news articles along with lead, titles and metadata.

**Dataset:** [`SZTAKI-HLT/HunSum-2-abstractive`](https://huggingface.co/datasets/SZTAKI-HLT/HunSum-2-abstractive) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2404.03555)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_1 | hun | News, Written | derived | found |



#### IndicQARetrieval

IndicQA is a manually curated cloze-style reading comprehension dataset that can be used for evaluating question-answering models in 11 Indic languages. It is repurposed retrieving relevant context for each question.

**Dataset:** [`mteb/IndicQARetrieval`](https://huggingface.co/datasets/mteb/IndicQARetrieval) • **License:** cc0-1.0 • [Learn more →](https://arxiv.org/abs/2212.05409)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | asm, ben, guj, hin, kan, ... (11) | Web, Written | human-annotated | machine-translated and verified |



#### JaCWIRRetrieval

JaCWIR is a small-scale Japanese information retrieval evaluation dataset consisting of
5000 question texts and approximately 500k web page titles and web page introductions or summaries
(meta descriptions, etc.). The question texts are created based on one of the 500k web pages,
and that data is used as a positive example for the question text.

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/hotchpotch/JaCWIR)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Web, Written | derived | found |



#### JaGovFaqsRetrieval

JaGovFaqs is a dataset consisting of FAQs manully extracted from the website of Japanese bureaus. The dataset consists of 22k FAQs, where the queries (questions) and corpus (answers) have been shuffled, and the goal is to match the answer with the question.

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-4.0 • [Learn more →](https://github.com/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Web, Written | derived | found |



#### JaQuADRetrieval

Human-annotated question-answer pairs for Japanese wikipedia pages.

**Dataset:** [`SkelterLabsInc/JaQuAD`](https://huggingface.co/datasets/SkelterLabsInc/JaQuAD) • **License:** cc-by-sa-3.0 • [Learn more →](https://arxiv.org/abs/2202.01764)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Encyclopaedic, Non-fiction, Written | human-annotated | found |



#### JaqketRetrieval

JAQKET (JApanese Questions on Knowledge of EnTities) is a QA dataset that is created based on quiz questions.

**Dataset:** [`mteb/jaqket`](https://huggingface.co/datasets/mteb/jaqket) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/kumapo/JAQKET-dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Encyclopaedic, Non-fiction, Written | human-annotated | found |



#### Ko-StrategyQA

Ko-StrategyQA

**Dataset:** [`taeminlee/Ko-StrategyQA`](https://huggingface.co/datasets/taeminlee/Ko-StrategyQA) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/taeminlee/Ko-StrategyQA)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | kor | not specified | not specified | not specified |



#### LEMBNarrativeQARetrieval

narrativeqa subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Fiction, Non-fiction, Written | derived | found |



#### LEMBNeedleRetrieval

needle subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_1 | eng | Academic, Blog, Written | derived | found |



#### LEMBPasskeyRetrieval

passkey subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_1 | eng | Fiction, Written | derived | found |



#### LEMBQMSumRetrieval

qmsum subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Spoken, Written | derived | found |



#### LEMBSummScreenFDRetrieval

summ_screen_fd subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Spoken, Written | derived | found |



#### LEMBWikimQARetrieval

2wikimqa subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### LIMITRetrieval

A simple retrieval task designed to test all combinations of top-2 documents. This version includes all 50k docs.

**Dataset:** [`orionweller/LIMIT`](https://huggingface.co/datasets/orionweller/LIMIT) • **License:** apache-2.0 • [Learn more →](https://github.com/google-deepmind/limit)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_2 | eng | Fiction | human-annotated | created |



#### LIMITSmallRetrieval

A simple retrieval task designed to test all combinations of top-2 documents. This version only includes the 46 documents that are relevant to the 1000 queries.

**Dataset:** [`orionweller/LIMIT-small`](https://huggingface.co/datasets/orionweller/LIMIT-small) • **License:** apache-2.0 • [Learn more →](https://github.com/google-deepmind/limit)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_2 | eng | Fiction | human-annotated | created |



#### LeCaRDv2

The task involves identifying and retrieving the case document that best matches or is most relevant to the scenario described in each of the provided queries.

**Dataset:** [`mteb/LeCaRDv2`](https://huggingface.co/datasets/mteb/LeCaRDv2) • **License:** mit • [Learn more →](https://github.com/THUIR/LeCaRDv2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | zho | Legal, Written | derived | found |



#### LegalBenchConsumerContractsQA

The dataset includes questions and answers related to contracts.

**Dataset:** [`mteb/legalbench_consumer_contracts_qa`](https://huggingface.co/datasets/mteb/legalbench_consumer_contracts_qa) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench/viewer/consumer_contracts_qa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Legal, Written | derived | found |



#### LegalBenchCorporateLobbying

The dataset includes bill titles and bill summaries related to corporate lobbying.

**Dataset:** [`mteb/legalbench_corporate_lobbying`](https://huggingface.co/datasets/mteb/legalbench_corporate_lobbying) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench/viewer/corporate_lobbying)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Legal, Written | derived | found |



#### LegalQuAD

The dataset consists of questions and legal documents in German.

**Dataset:** [`mteb/LegalQuAD`](https://huggingface.co/datasets/mteb/LegalQuAD) • **License:** cc-by-4.0 • [Learn more →](https://github.com/Christoph911/AIKE2021_Appendix)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | deu | Legal, Written | derived | found |



#### LegalSummarization

The dataset consistes of 439 pairs of contracts and their summarizations from https://tldrlegal.com and https://tosdr.org/.

**Dataset:** [`mteb/legal_summarization`](https://huggingface.co/datasets/mteb/legal_summarization) • **License:** apache-2.0 • [Learn more →](https://github.com/lauramanor/legal_summarization)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Legal, Written | derived | found |



#### LitSearchRetrieval


        The dataset contains the query set and retrieval corpus for the paper LitSearch: A Retrieval Benchmark for
        Scientific Literature Search. It introduces LitSearch, a retrieval benchmark comprising 597 realistic literature
        search queries about recent ML and NLP papers. LitSearch is constructed using a combination of (1) questions
        generated by GPT-4 based on paragraphs containing inline citations from research papers and (2) questions about
        recently published papers, manually written by their authors. All LitSearch questions were manually examined or
        edited by experts to ensure high quality.
        

**Dataset:** [`princeton-nlp/LitSearch`](https://huggingface.co/datasets/princeton-nlp/LitSearch) • **License:** mit • [Learn more →](https://github.com/princeton-nlp/LitSearch)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Non-fiction, Written | LM-generated | found |



#### LoTTE

LoTTE (Long-Tail Topic-stratified Evaluation for IR) is designed to evaluate retrieval models on underrepresented, long-tail topics. Unlike MSMARCO or BEIR, LoTTE features domain-specific queries and passages from StackExchange (covering writing, recreation, science, technology, and lifestyle), providing a challenging out-of-domain generalization benchmark.

**Dataset:** [`mteb/LoTTE`](https://huggingface.co/datasets/mteb/LoTTE) • **License:** mit • [Learn more →](https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | precision_at_5 | eng | Academic, Social, Web | derived | found |



#### MBPPRetrieval

A code retrieval task based on 378 Python programming problems from MBPP (Mostly Basic Python Programming). Each query is a natural language description of a programming task (e.g., 'Write a function to find the shared elements from the given two lists'), and the corpus contains Python code implementations. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains Python function implementations with proper syntax and logic.

**Dataset:** [`embedding-benchmark/MBPP`](https://huggingface.co/datasets/embedding-benchmark/MBPP) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/MBPP)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng, python | Programming | expert-annotated | found |



#### MIRACLRetrieval

MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages.

**Dataset:** [`miracl/mmteb-miracl`](https://huggingface.co/datasets/miracl/mmteb-miracl) • **License:** cc-by-sa-4.0 • [Learn more →](http://miracl.ai/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, ben, deu, eng, fas, ... (18) | Encyclopaedic, Written | expert-annotated | created |



#### MIRACLRetrievalHardNegatives

MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/miracl-hard-negatives`](https://huggingface.co/datasets/mteb/miracl-hard-negatives) • **License:** cc-by-sa-4.0 • [Learn more →](http://miracl.ai/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, ben, deu, eng, fas, ... (18) | Encyclopaedic, Written | expert-annotated | created |



#### MKQARetrieval

Multilingual Knowledge Questions & Answers (MKQA)contains 10,000 queries sampled from the Google Natural Questions dataset.
        For each query we collect new passage-independent answers. These queries and answers are then human translated into 25 Non-English languages.

**Dataset:** [`apple/mkqa`](https://huggingface.co/datasets/apple/mkqa) • **License:** cc-by-3.0 • [Learn more →](https://github.com/apple/ml-mkqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, dan, deu, eng, fin, ... (24) | Written | human-annotated | found |



#### MLQARetrieval

MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating cross-lingual question answering performance.
        MLQA consists of over 5K extractive QA instances (12K in English) in SQuAD format in seven languages - English, Arabic,
        German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA is highly parallel, with QA instances parallel between
        4 different languages on average.

**Dataset:** [`facebook/mlqa`](https://huggingface.co/datasets/facebook/mlqa) • **License:** cc-by-sa-3.0 • [Learn more →](https://huggingface.co/datasets/mlqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, deu, eng, hin, spa, ... (7) | Encyclopaedic, Written | human-annotated | found |



#### MLQuestions

MLQuestions is a domain adaptation dataset for the machine learning domainIt consists of ML questions along with passages from Wikipedia machine learning pages (https://en.wikipedia.org/wiki/Category:Machine_learning)

**Dataset:** [`McGill-NLP/mlquestions`](https://huggingface.co/datasets/McGill-NLP/mlquestions) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://github.com/McGill-NLP/MLQuestions)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Encyclopaedic, Written | human-annotated | found |



#### MMarcoRetrieval

MMarcoRetrieval

**Dataset:** [`mteb/MMarcoRetrieval`](https://huggingface.co/datasets/mteb/MMarcoRetrieval) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2309.07597)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | not specified | not specified | not specified |



#### MSMARCO

MS MARCO is a collection of datasets focused on deep learning in search

**Dataset:** [`mteb/msmarco`](https://huggingface.co/datasets/mteb/msmarco) • **License:** msr-la-nc • [Learn more →](https://microsoft.github.io/msmarco/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Blog, Encyclopaedic, Government, Medical, ... (10) | derived | found |



#### MSMARCO-Fa

MSMARCO-Fa

**Dataset:** [`MCINext/msmarco-fa`](https://huggingface.co/datasets/MCINext/msmarco-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/msmarco-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### MSMARCO-PL

MS MARCO is a collection of datasets focused on deep learning in search

**Dataset:** [`mteb/MSMARCO-PL`](https://huggingface.co/datasets/mteb/MSMARCO-PL) • **License:** https://microsoft.github.io/msmarco/ • [Learn more →](https://microsoft.github.io/msmarco/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web, Written | derived | machine-translated |



#### MSMARCO-PLHardNegatives

MS MARCO is a collection of datasets focused on deep learning in search. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/MSMARCO_PL_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/MSMARCO_PL_test_top_250_only_w_correct-v2) • **License:** https://microsoft.github.io/msmarco/ • [Learn more →](https://microsoft.github.io/msmarco/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web, Written | derived | machine-translated |



#### MSMARCO-VN

A translated dataset from MS MARCO is a collection of datasets focused on deep learning in search
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/msmarco-vn`](https://huggingface.co/datasets/GreenNode/msmarco-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://microsoft.github.io/msmarco/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Academic, Blog, Encyclopaedic, Government, Medical, ... (10) | derived | machine-translated and LM verified |



#### MSMARCOHardNegatives

MS MARCO is a collection of datasets focused on deep learning in search. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/MSMARCO_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/MSMARCO_test_top_250_only_w_correct-v2) • **License:** msr-la-nc • [Learn more →](https://microsoft.github.io/msmarco/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Blog, Encyclopaedic, Government, Medical, ... (10) | derived | found |



#### MSMARCOv2

MS MARCO is a collection of datasets focused on deep learning in search. This version is derived from BEIR

**Dataset:** [`mteb/msmarco-v2`](https://huggingface.co/datasets/mteb/msmarco-v2) • **License:** msr-la-nc • [Learn more →](https://microsoft.github.io/msmarco/TREC-Deep-Learning.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Blog, Encyclopaedic, Government, Medical, ... (10) | derived | found |



#### MedicalQARetrieval

The dataset consists 2048 medical question and answer pairs.

**Dataset:** [`mteb/medical_qa`](https://huggingface.co/datasets/mteb/medical_qa) • **License:** cc0-1.0 • [Learn more →](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical, Written | derived | found |



#### MedicalRetrieval

MedicalRetrieval

**Dataset:** [`mteb/MedicalRetrieval`](https://huggingface.co/datasets/mteb/MedicalRetrieval) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2203.03367)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | not specified | not specified | not specified |



#### MintakaRetrieval

We introduce Mintaka, a complex, natural, and multilingual dataset designed for experimenting with end-to-end question-answering models. Mintaka is composed of 20,000 question-answer pairs collected in English, annotated with Wikidata entities, and translated into Arabic, French, German, Hindi, Italian, Japanese, Portuguese, and Spanish for a total of 180,000 samples. Mintaka includes 8 types of complex questions, including superlative, intersection, and multi-hop questions, which were naturally elicited from crowd workers. 

**Dataset:** [`jinaai/mintakaqa`](https://huggingface.co/datasets/jinaai/mintakaqa) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/jinaai/mintakaqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, deu, fra, hin, ita, ... (8) | Encyclopaedic, Written | derived | human-translated |



#### MrTidyRetrieval

Mr. TyDi is a multi-lingual benchmark dataset built on TyDi, covering eleven typologically diverse languages. It is designed for monolingual retrieval, specifically to evaluate ranking with learned dense representations.

**Dataset:** [`mteb/mrtidy`](https://huggingface.co/datasets/mteb/mrtidy) • **License:** cc-by-sa-3.0 • [Learn more →](https://huggingface.co/datasets/castorini/mr-tydi)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, ben, eng, fin, ind, ... (11) | Encyclopaedic, Written | human-annotated | found |



#### MultiLongDocRetrieval

Multi Long Doc Retrieval (MLDR) 'is curated by the multilingual articles from Wikipedia, Wudao and mC4 (see Table 7), and NarrativeQA (Kocˇisky ́ et al., 2018; Gu ̈nther et al., 2023), which is only for English.' (Chen et al., 2024).
        It is constructed by sampling lengthy articles from Wikipedia, Wudao and mC4 datasets and randomly choose paragraphs from them. Then we use GPT-3.5 to generate questions based on these paragraphs. The generated question and the sampled article constitute a new text pair to the dataset.

**Dataset:** [`Shitao/MLDR`](https://huggingface.co/datasets/Shitao/MLDR) • **License:** mit • [Learn more →](https://arxiv.org/abs/2402.03216)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, cmn, deu, eng, fra, ... (13) | Encyclopaedic, Fiction, Non-fiction, Web, Written | LM-generated | found |



#### NFCorpus

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval

**Dataset:** [`mteb/nfcorpus`](https://huggingface.co/datasets/mteb/nfcorpus) • **License:** not specified • [Learn more →](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Medical, Written | not specified | not specified |



#### NFCorpus-Fa

NFCorpus-Fa

**Dataset:** [`MCINext/nfcorpus-fa`](https://huggingface.co/datasets/MCINext/nfcorpus-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/nfcorpus-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Medical | derived | found |



#### NFCorpus-NL

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval. NFCorpus-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-nfcorpus`](https://huggingface.co/datasets/clips/beir-nl-nfcorpus) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-nfcorpus)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Academic, Medical, Written | derived | machine-translated and verified |



#### NFCorpus-PL

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval

**Dataset:** [`mteb/NFCorpus-PL`](https://huggingface.co/datasets/mteb/NFCorpus-PL) • **License:** not specified • [Learn more →](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | not specified | not specified | not specified |



#### NFCorpus-VN

A translated dataset from NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/nfcorpus-vn`](https://huggingface.co/datasets/GreenNode/nfcorpus-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Academic, Medical, Written | derived | machine-translated and LM verified |



#### NLPJournalAbsArticleRetrieval

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding full article with the given abstract. This is the V1 dataset (last updated 2020-06-15).

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



#### NLPJournalAbsArticleRetrieval.V2

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding full article with the given abstract. This is the V2 dataset (last updated 2025-06-15).

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | jpn | Academic, Written | derived | found |



#### NLPJournalAbsIntroRetrieval

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding introduction with the given abstract. This is the V1 dataset (last update 2020-06-15).

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



#### NLPJournalAbsIntroRetrieval.V2

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding introduction with the given abstract. This is the V2 dataset (last update 2025-06-15).

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



#### NLPJournalTitleAbsRetrieval

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding abstract with the given title. This is the V1 dataset (last updated 2020-06-15).

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



#### NLPJournalTitleAbsRetrieval.V2

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding abstract with the given title. This is the V2 dataset (last updated 2025-06-15).

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



#### NLPJournalTitleIntroRetrieval

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding introduction with the given title. This is the V1 dataset (last updated 2020-06-15).

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



#### NLPJournalTitleIntroRetrieval.V2

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding introduction with the given title. This is the V2 dataset (last updated 2025-06-15).

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



#### NQ

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval

**Dataset:** [`mteb/nq`](https://huggingface.co/datasets/mteb/nq) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



#### NQ-Fa

NQ-Fa

**Dataset:** [`MCINext/nq-fa`](https://huggingface.co/datasets/MCINext/nq-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/nq-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Encyclopaedic | derived | found |



#### NQ-NL

NQ-NL is a translation of NQ

**Dataset:** [`clips/beir-nl-nq`](https://huggingface.co/datasets/clips/beir-nl-nq) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-nq)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Encyclopaedic, Written | derived | machine-translated and verified |



#### NQ-PL

Natural Questions: A Benchmark for Question Answering Research

**Dataset:** [`mteb/NQ-PL`](https://huggingface.co/datasets/mteb/NQ-PL) • **License:** not specified • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | not specified | not specified | machine-translated |



#### NQ-PLHardNegatives

Natural Questions: A Benchmark for Question Answering Research. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/NQ_PL_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/NQ_PL_test_top_250_only_w_correct-v2) • **License:** not specified • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | not specified | not specified | machine-translated |



#### NQ-VN

A translated dataset from NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/nq-vn`](https://huggingface.co/datasets/GreenNode/nq-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Encyclopaedic, Written | derived | machine-translated and LM verified |



#### NQHardNegatives

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/NQ_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/NQ_test_top_250_only_w_correct-v2) • **License:** not specified • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | not specified | not specified | not specified |



#### NanoArguAnaRetrieval

NanoArguAna is a smaller subset of ArguAna, a dataset for argument retrieval in debate contexts.

**Dataset:** [`zeta-alpha-ai/NanoArguAna`](https://huggingface.co/datasets/zeta-alpha-ai/NanoArguAna) • **License:** cc-by-4.0 • [Learn more →](http://argumentation.bplaced.net/arguana/data)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical, Written | expert-annotated | found |



#### NanoClimateFeverRetrieval

NanoClimateFever is a small version of the BEIR dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change.

**Dataset:** [`zeta-alpha-ai/NanoClimateFEVER`](https://huggingface.co/datasets/zeta-alpha-ai/NanoClimateFEVER) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2012.00614)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, News, Non-fiction | expert-annotated | found |



#### NanoDBPediaRetrieval

NanoDBPediaRetrieval is a small version of the standard test collection for entity search over the DBpedia knowledge base.

**Dataset:** [`zeta-alpha-ai/NanoDBPedia`](https://huggingface.co/datasets/zeta-alpha-ai/NanoDBPedia) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/zeta-alpha-ai/NanoDBPedia)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic | expert-annotated | found |



#### NanoFEVERRetrieval

NanoFEVER is a smaller version of FEVER (Fact Extraction and VERification), which consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from.

**Dataset:** [`zeta-alpha-ai/NanoFEVER`](https://huggingface.co/datasets/zeta-alpha-ai/NanoFEVER) • **License:** cc-by-4.0 • [Learn more →](https://fever.ai/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Encyclopaedic | expert-annotated | found |



#### NanoFiQA2018Retrieval

NanoFiQA2018 is a smaller subset of the Financial Opinion Mining and Question Answering dataset.

**Dataset:** [`zeta-alpha-ai/NanoFiQA2018`](https://huggingface.co/datasets/zeta-alpha-ai/NanoFiQA2018) • **License:** cc-by-4.0 • [Learn more →](https://sites.google.com/view/fiqa/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Social | human-annotated | found |



#### NanoHotpotQARetrieval

NanoHotpotQARetrieval is a smaller subset of the HotpotQA dataset, which is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.

**Dataset:** [`zeta-alpha-ai/NanoHotpotQA`](https://huggingface.co/datasets/zeta-alpha-ai/NanoHotpotQA) • **License:** cc-by-4.0 • [Learn more →](https://hotpotqa.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Web, Written | human-annotated | found |



#### NanoMSMARCORetrieval

NanoMSMARCORetrieval is a smaller subset of MS MARCO, a collection of datasets focused on deep learning in search.

**Dataset:** [`zeta-alpha-ai/NanoMSMARCO`](https://huggingface.co/datasets/zeta-alpha-ai/NanoMSMARCO) • **License:** cc-by-4.0 • [Learn more →](https://microsoft.github.io/msmarco/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Web | human-annotated | found |



#### NanoNFCorpusRetrieval

NanoNFCorpus is a smaller subset of NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval.

**Dataset:** [`zeta-alpha-ai/NanoNFCorpus`](https://huggingface.co/datasets/zeta-alpha-ai/NanoNFCorpus) • **License:** cc-by-4.0 • [Learn more →](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Medical, Written | expert-annotated | found |



#### NanoNQRetrieval

NanoNQ is a smaller subset of a dataset which contains questions from real users, and it requires QA systems to read and comprehend an entire Wikipedia article that may or may not contain the answer to the question.

**Dataset:** [`zeta-alpha-ai/NanoNQ`](https://huggingface.co/datasets/zeta-alpha-ai/NanoNQ) • **License:** cc-by-4.0 • [Learn more →](https://ai.google.com/research/NaturalQuestions)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Web | human-annotated | found |



#### NanoQuoraRetrieval

NanoQuoraRetrieval is a smaller subset of the QuoraRetrieval dataset, which is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.

**Dataset:** [`zeta-alpha-ai/NanoQuoraRetrieval`](https://huggingface.co/datasets/zeta-alpha-ai/NanoQuoraRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Social | human-annotated | found |



#### NanoSCIDOCSRetrieval

NanoFiQA2018 is a smaller subset of SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation.

**Dataset:** [`zeta-alpha-ai/NanoSCIDOCS`](https://huggingface.co/datasets/zeta-alpha-ai/NanoSCIDOCS) • **License:** cc-by-4.0 • [Learn more →](https://allenai.org/data/scidocs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Non-fiction, Written | expert-annotated | found |



#### NanoSciFactRetrieval

NanoSciFact is a smaller subset of SciFact, which verifies scientific claims using evidence from the research literature containing scientific paper abstracts.

**Dataset:** [`zeta-alpha-ai/NanoSciFact`](https://huggingface.co/datasets/zeta-alpha-ai/NanoSciFact) • **License:** cc-by-4.0 • [Learn more →](https://github.com/allenai/scifact)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Medical, Written | expert-annotated | found |



#### NanoTouche2020Retrieval

NanoTouche2020 is a smaller subset of Touché Task 1: Argument Retrieval for Controversial Questions.

**Dataset:** [`zeta-alpha-ai/NanoTouche2020`](https://huggingface.co/datasets/zeta-alpha-ai/NanoTouche2020) • **License:** cc-by-4.0 • [Learn more →](https://webis.de/events/touche-20/shared-task-1.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic | human-annotated | found |



#### NarrativeQARetrieval

NarrativeQA is a dataset for the task of question answering on long narratives. It consists of realistic QA instances collected from literature (fiction and non-fiction) and movie scripts. 

**Dataset:** [`deepmind/narrativeqa`](https://huggingface.co/datasets/deepmind/narrativeqa) • **License:** not specified • [Learn more →](https://metatext.io/datasets/narrativeqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | not specified | not specified | not specified |



#### NeuCLIR2022Retrieval

The task involves identifying and retrieving the documents that are relevant to the queries.

**Dataset:** [`mteb/neuclir-2022`](https://huggingface.co/datasets/mteb/neuclir-2022) • **License:** odc-by • [Learn more →](https://neuclir.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | fas, rus, zho | News, Written | expert-annotated | found |



#### NeuCLIR2022RetrievalHardNegatives

The task involves identifying and retrieving the documents that are relevant to the queries. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/neuclir-2022-hard-negatives`](https://huggingface.co/datasets/mteb/neuclir-2022-hard-negatives) • **License:** odc-by • [Learn more →](https://neuclir.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | fas, rus, zho | News, Written | expert-annotated | found |



#### NeuCLIR2023Retrieval

The task involves identifying and retrieving the documents that are relevant to the queries.

**Dataset:** [`mteb/neuclir-2023`](https://huggingface.co/datasets/mteb/neuclir-2023) • **License:** odc-by • [Learn more →](https://neuclir.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | fas, rus, zho | News, Written | expert-annotated | found |



#### NeuCLIR2023RetrievalHardNegatives

The task involves identifying and retrieving the documents that are relevant to the queries. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/neuclir-2023-hard-negatives`](https://huggingface.co/datasets/mteb/neuclir-2023-hard-negatives) • **License:** odc-by • [Learn more →](https://neuclir.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | fas, rus, zho | News, Written | expert-annotated | found |



#### NorQuadRetrieval

Human-created question for Norwegian wikipedia passages.

**Dataset:** [`mteb/norquad_retrieval`](https://huggingface.co/datasets/mteb/norquad_retrieval) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.17/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nob | Encyclopaedic, Non-fiction, Written | derived | found |



#### PIQA

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on PIQA.

**Dataset:** [`RAR-b/piqa`](https://huggingface.co/datasets/RAR-b/piqa) • **License:** afl-3.0 • [Learn more →](https://arxiv.org/abs/1911.11641)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### PUGGRetrieval

Information Retrieval PUGG dataset for the Polish language.

**Dataset:** [`clarin-pl/PUGG_IR`](https://huggingface.co/datasets/clarin-pl/PUGG_IR) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2024.findings-acl.652/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web | human-annotated | multiple |



#### PersianWebDocumentRetrieval

Persian dataset designed specifically for the task of text information retrieval through the web.

**Dataset:** [`MCINext/persian-web-document-retrieval`](https://huggingface.co/datasets/MCINext/persian-web-document-retrieval) • **License:** not specified • [Learn more →](https://ieeexplore.ieee.org/document/10553090)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### PublicHealthQA

A multilingual dataset for public health question answering, based on FAQ sourced from CDC and WHO.

**Dataset:** [`xhluca/publichealth-qa`](https://huggingface.co/datasets/xhluca/publichealth-qa) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://huggingface.co/datasets/xhluca/publichealth-qa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, eng, fra, kor, rus, ... (8) | Government, Medical, Web, Written | derived | found |



#### Quail

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on Quail.

**Dataset:** [`RAR-b/quail`](https://huggingface.co/datasets/RAR-b/quail) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://text-machine.cs.uml.edu/lab2/projects/quail/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### Quora-NL

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions. QuoraRetrieval-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-quora`](https://huggingface.co/datasets/clips/beir-nl-quora) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-quora)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Written | derived | machine-translated and verified |



#### Quora-PL

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.

**Dataset:** [`mteb/Quora-PL`](https://huggingface.co/datasets/mteb/Quora-PL) • **License:** not specified • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | not specified | not specified | machine-translated |



#### Quora-PLHardNegatives

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/Quora_PL_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/Quora_PL_test_top_250_only_w_correct-v2) • **License:** not specified • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | not specified | not specified | machine-translated |



#### Quora-VN

A translated dataset from QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a
            question, find other (duplicate) questions.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/quora-vn`](https://huggingface.co/datasets/GreenNode/quora-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Blog, Web, Written | derived | machine-translated and LM verified |



#### QuoraRetrieval

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.

**Dataset:** [`mteb/quora`](https://huggingface.co/datasets/mteb/quora) • **License:** not specified • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Blog, Web, Written | human-annotated | found |



#### QuoraRetrieval-Fa

QuoraRetrieval-Fa

**Dataset:** [`MCINext/quora-fa`](https://huggingface.co/datasets/MCINext/quora-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/quora-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



#### QuoraRetrievalHardNegatives

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/QuoraRetrieval_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/QuoraRetrieval_test_top_250_only_w_correct-v2) • **License:** not specified • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | not specified | not specified | not specified |



#### R2MEDBioinformaticsRetrieval

Bioinformatics retrieval dataset.

**Dataset:** [`R2MED/Bioinformatics`](https://huggingface.co/datasets/R2MED/Bioinformatics) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/Bioinformatics)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



#### R2MEDBiologyRetrieval

Biology retrieval dataset.

**Dataset:** [`R2MED/Biology`](https://huggingface.co/datasets/R2MED/Biology) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/Biology)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



#### R2MEDIIYiClinicalRetrieval

IIYi-Clinical retrieval dataset.

**Dataset:** [`R2MED/IIYi-Clinical`](https://huggingface.co/datasets/R2MED/IIYi-Clinical) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/IIYi-Clinical)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



#### R2MEDMedQADiagRetrieval

MedQA-Diag retrieval dataset.

**Dataset:** [`R2MED/MedQA-Diag`](https://huggingface.co/datasets/R2MED/MedQA-Diag) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/MedQA-Diag)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



#### R2MEDMedXpertQAExamRetrieval

MedXpertQA-Exam retrieval dataset.

**Dataset:** [`R2MED/MedXpertQA-Exam`](https://huggingface.co/datasets/R2MED/MedXpertQA-Exam) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/MedXpertQA-Exam)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



#### R2MEDMedicalSciencesRetrieval

Medical-Sciences retrieval dataset.

**Dataset:** [`R2MED/Medical-Sciences`](https://huggingface.co/datasets/R2MED/Medical-Sciences) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/Medical-Sciences)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



#### R2MEDPMCClinicalRetrieval

PMC-Clinical retrieval dataset.

**Dataset:** [`R2MED/PMC-Clinical`](https://huggingface.co/datasets/R2MED/PMC-Clinical) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/PMC-Clinical)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



#### R2MEDPMCTreatmentRetrieval

PMC-Treatment retrieval dataset.

**Dataset:** [`R2MED/PMC-Treatment`](https://huggingface.co/datasets/R2MED/PMC-Treatment) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/PMC-Treatment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



#### RARbCode

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on RAR-b code-pooled dataset.

**Dataset:** [`RAR-b/humanevalpack-mbpp-pooled`](https://huggingface.co/datasets/RAR-b/humanevalpack-mbpp-pooled) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2404.06347)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Programming, Written | derived | found |



#### RARbMath

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on RAR-b math-pooled dataset.

**Dataset:** [`RAR-b/math-pooled`](https://huggingface.co/datasets/RAR-b/math-pooled) • **License:** mit • [Learn more →](https://arxiv.org/abs/2404.06347)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### RiaNewsRetrieval

News article retrieval by headline. Based on Rossiya Segodnya dataset.

**Dataset:** [`ai-forever/ria-news-retrieval`](https://huggingface.co/datasets/ai-forever/ria-news-retrieval) • **License:** cc-by-nc-nd-4.0 • [Learn more →](https://arxiv.org/abs/1901.07786)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | rus | News, Written | derived | found |



#### RiaNewsRetrievalHardNegatives

News article retrieval by headline. Based on Rossiya Segodnya dataset. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/RiaNewsRetrieval_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/RiaNewsRetrieval_test_top_250_only_w_correct-v2) • **License:** cc-by-nc-nd-4.0 • [Learn more →](https://arxiv.org/abs/1901.07786)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | rus | News, Written | derived | found |



#### RuBQRetrieval

Paragraph retrieval based on RuBQ 2.0. Retrieve paragraphs from Wikipedia that answer the question.

**Dataset:** [`ai-forever/rubq-retrieval`](https://huggingface.co/datasets/ai-forever/rubq-retrieval) • **License:** cc-by-sa-4.0 • [Learn more →](https://openreview.net/pdf?id=P5UQFFoQ4PJ)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | rus | Encyclopaedic, Written | human-annotated | created |



#### RuSciBenchCiteRetrieval

This task is focused on Direct Citation Prediction for scientific papers from eLibrary,
        Russia's largest electronic library of scientific publications. Given a query paper (title and abstract),
        the goal is to retrieve papers that are directly cited by it from a larger corpus of papers.
        The dataset for this task consists of 3,000 query papers, 15,000 relevant (cited) papers,
        and 75,000 irrelevant papers. The task is available for both Russian and English scientific texts.

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_cite_retrieval`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_cite_retrieval) • **License:** mit • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng, rus | Academic, Non-fiction, Written | derived | found |



#### RuSciBenchCociteRetrieval

This task focuses on Co-citation Prediction for scientific papers from eLibrary,
        Russia's largest electronic library of scientific publications. Given a query paper (title and abstract),
        the goal is to retrieve other papers that are co-cited with it. Two papers are considered co-cited
        if they are both cited by at least 5 of the same other papers. Similar to the Direct Citation task,
        this task employs a retrieval setup: for a given query paper, all other papers in the corpus that
        are not co-cited with it are considered negative examples. The task is available for both Russian
        and English scientific texts.

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_cocite_retrieval`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_cocite_retrieval) • **License:** mit • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng, rus | Academic, Non-fiction, Written | derived | found |



#### SCIDOCS

SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation.

**Dataset:** [`mteb/scidocs`](https://huggingface.co/datasets/mteb/scidocs) • **License:** cc-by-sa-4.0 • [Learn more →](https://allenai.org/data/scidocs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Non-fiction, Written | not specified | found |



#### SCIDOCS-Fa

SCIDOCS-Fa

**Dataset:** [`MCINext/scidocs-fa`](https://huggingface.co/datasets/MCINext/scidocs-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/scidocs-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Academic | derived | found |



#### SCIDOCS-NL

SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation. SciDocs-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-scidocs`](https://huggingface.co/datasets/clips/beir-nl-scidocs) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-scidocs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Academic, Non-fiction, Written | derived | machine-translated and verified |



#### SCIDOCS-PL

SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation.

**Dataset:** [`mteb/SCIDOCS-PL`](https://huggingface.co/datasets/mteb/SCIDOCS-PL) • **License:** not specified • [Learn more →](https://allenai.org/data/scidocs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | not specified | not specified | not specified |



#### SCIDOCS-VN

A translated dataset from SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation
            prediction, to document classification and recommendation.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/scidocs-vn`](https://huggingface.co/datasets/GreenNode/scidocs-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://allenai.org/data/scidocs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified |



#### SIQA

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on SIQA.

**Dataset:** [`RAR-b/siqa`](https://huggingface.co/datasets/RAR-b/siqa) • **License:** not specified • [Learn more →](https://leaderboard.allenai.org/socialiqa/submissions/get-started)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### SKQuadRetrieval

Retrieval SK Quad evaluates Slovak search performance using questions and answers derived from the SK-QuAD dataset. It measures relevance with scores assigned to answers based on their relevancy to corresponding questions, which is vital for improving Slovak language search systems.

**Dataset:** [`TUKE-KEMT/retrieval-skquad`](https://huggingface.co/datasets/TUKE-KEMT/retrieval-skquad) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/TUKE-KEMT/retrieval-skquad)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | slk | Encyclopaedic | human-annotated | found |



#### SNLRetrieval

Webscrabed articles and ingresses from the Norwegian lexicon 'Det Store Norske Leksikon'.

**Dataset:** [`adrlau/navjordj-SNL_summarization_copy`](https://huggingface.co/datasets/adrlau/navjordj-SNL_summarization_copy) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/mteb/SNLRetrieval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nob | Encyclopaedic, Non-fiction, Written | derived | found |



#### SadeemQuestionRetrieval

SadeemQuestion: A Benchmark Data Set for Community Question-Retrieval Research

**Dataset:** [`sadeem-ai/sadeem-ar-eval-retrieval-questions`](https://huggingface.co/datasets/sadeem-ai/sadeem-ar-eval-retrieval-questions) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/sadeem-ai/sadeem-ar-eval-retrieval-questions)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara | Written, Written | derived | found |



#### SciFact

SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.

**Dataset:** [`mteb/scifact`](https://huggingface.co/datasets/mteb/scifact) • **License:** not specified • [Learn more →](https://github.com/allenai/scifact)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Medical, Written | not specified | not specified |



#### SciFact-Fa

SciFact-Fa

**Dataset:** [`MCINext/scifact-fa`](https://huggingface.co/datasets/MCINext/scifact-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/scifact-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Academic | derived | found |



#### SciFact-NL

SciFactNL verifies scientific claims in Dutch using evidence from the research literature containing scientific paper abstracts.

**Dataset:** [`clips/beir-nl-scifact`](https://huggingface.co/datasets/clips/beir-nl-scifact) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-scifact)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Academic, Medical, Written | derived | machine-translated and verified |



#### SciFact-PL

SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.

**Dataset:** [`mteb/SciFact-PL`](https://huggingface.co/datasets/mteb/SciFact-PL) • **License:** not specified • [Learn more →](https://github.com/allenai/scifact)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic, Medical, Written | not specified | not specified |



#### SciFact-VN

A translated dataset from SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/scifact-vn`](https://huggingface.co/datasets/GreenNode/scifact-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/allenai/scifact)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Academic, Medical, Written | derived | machine-translated and LM verified |



#### SlovakSumRetrieval


            SlovakSum, a Slovak news summarization dataset consisting of over 200 thousand
            news articles with titles and short abstracts obtained from multiple Slovak newspapers.

            Originally intended as a summarization task, but since no human annotations were provided
            here reformulated to a retrieval task.
        

**Dataset:** [`NaiveNeuron/slovaksum`](https://huggingface.co/datasets/NaiveNeuron/slovaksum) • **License:** openrail • [Learn more →](https://huggingface.co/datasets/NaiveNeuron/slovaksum)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | slk | News, Social, Web, Written | derived | found |



#### SpanishPassageRetrievalS2P

Test collection for passage retrieval from health-related Web resources in Spanish.

**Dataset:** [`jinaai/spanish_passage_retrieval`](https://huggingface.co/datasets/jinaai/spanish_passage_retrieval) • **License:** not specified • [Learn more →](https://mklab.iti.gr/results/spanish-passage-retrieval-dataset/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | spa | not specified | not specified | not specified |



#### SpanishPassageRetrievalS2S

Test collection for passage retrieval from health-related Web resources in Spanish.

**Dataset:** [`jinaai/spanish_passage_retrieval`](https://huggingface.co/datasets/jinaai/spanish_passage_retrieval) • **License:** not specified • [Learn more →](https://mklab.iti.gr/results/spanish-passage-retrieval-dataset/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | spa | not specified | not specified | not specified |



#### SpartQA

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on SpartQA.

**Dataset:** [`RAR-b/spartqa`](https://huggingface.co/datasets/RAR-b/spartqa) • **License:** mit • [Learn more →](https://github.com/HLR/SpartQA_generation)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### StackOverflowQA

The dataset is a collection of natural language queries and their corresponding response which may include some text mixed with code snippets. The task is to retrieve the most relevant response for a given query.

**Dataset:** [`CoIR-Retrieval/stackoverflow-qa`](https://huggingface.co/datasets/CoIR-Retrieval/stackoverflow-qa) • **License:** mit • [Learn more →](https://arxiv.org/abs/2407.02883)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Programming, Written | derived | found |



#### StatcanDialogueDatasetRetrieval

A Dataset for Retrieving Data Tables through Conversations with Genuine Intents, available in English and French.

**Dataset:** [`McGill-NLP/statcan-dialogue-dataset-retrieval`](https://huggingface.co/datasets/McGill-NLP/statcan-dialogue-dataset-retrieval) • **License:** https://huggingface.co/datasets/McGill-NLP/statcan-dialogue-dataset-retrieval/blob/main/LICENSE.md • [Learn more →](https://mcgill-nlp.github.io/statcan-dialogue-dataset/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_10 | eng, fra | Government, Web, Written | derived | found |



#### SweFaqRetrieval

A Swedish QA dataset derived from FAQ

**Dataset:** [`AI-Sweden/SuperLim`](https://huggingface.co/datasets/AI-Sweden/SuperLim) • **License:** cc-by-sa-4.0 • [Learn more →](https://spraakbanken.gu.se/en/resources/superlim)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | swe | Government, Non-fiction, Written | derived | found |



#### SwednRetrieval

The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail dataset both regarding textual structure

**Dataset:** [`sbx/superlim-2`](https://huggingface.co/datasets/sbx/superlim-2) • **License:** cc-by-sa-4.0 • [Learn more →](https://spraakbanken.gu.se/en/resources/swedn)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | swe | News, Non-fiction, Written | derived | found |



#### SynPerChatbotRAGFAQRetrieval

Synthetic Persian Chatbot RAG FAQ Retrieval

**Dataset:** [`MCINext/synthetic-persian-chatbot-rag-faq-retrieval`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-faq-retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-faq-retrieval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotRAGTopicsRetrieval

Synthetic Persian Chatbot RAG Topics Retrieval

**Dataset:** [`MCINext/synthetic-persian-chatbot-rag-topics-retrieval`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-topics-retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-topics-retrieval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotTopicsRetrieval

Synthetic Persian Chatbot Topics Retrieval

**Dataset:** [`MCINext/synthetic-persian-chatbot-topics-retrieval`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-topics-retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-topics-retrieval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerQARetrieval

Synthetic Persian QA Retrieval

**Dataset:** [`MCINext/synthetic-persian-qa-retrieval`](https://huggingface.co/datasets/MCINext/synthetic-persian-qa-retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/synthetic-persian-qa-retrieval/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | LM-generated | LM-generated and verified |



#### SyntecRetrieval

This dataset has been built from the Syntec Collective bargaining agreement.

**Dataset:** [`lyon-nlp/mteb-fr-retrieval-syntec-s2p`](https://huggingface.co/datasets/lyon-nlp/mteb-fr-retrieval-syntec-s2p) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/lyon-nlp/mteb-fr-retrieval-syntec-s2p)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fra | Legal, Written | human-annotated | created |



#### SyntheticText2SQL

The dataset is a collection of natural language queries and their corresponding sql snippets. The task is to retrieve the most relevant code snippet for a given query.

**Dataset:** [`CoIR-Retrieval/synthetic-text2sql`](https://huggingface.co/datasets/CoIR-Retrieval/synthetic-text2sql) • **License:** mit • [Learn more →](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng, sql | Programming, Written | derived | found |



#### T2Retrieval

T2Ranking: A large-scale Chinese Benchmark for Passage Ranking

**Dataset:** [`mteb/T2Retrieval`](https://huggingface.co/datasets/mteb/T2Retrieval) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2304.03679)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | Academic, Financial, Government, Medical, Non-fiction | human-annotated | not specified |



#### TRECCOVID

TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic.

**Dataset:** [`mteb/trec-covid`](https://huggingface.co/datasets/mteb/trec-covid) • **License:** not specified • [Learn more →](https://ir.nist.gov/covidSubmit/index.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Medical, Written | not specified | not specified |



#### TRECCOVID-Fa

TRECCOVID-Fa

**Dataset:** [`MCINext/trec-covid-fa`](https://huggingface.co/datasets/MCINext/trec-covid-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/trec-covid-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Medical | derived | found |



#### TRECCOVID-NL

TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic. TRECCOVID-NL is a Dutch translation. 

**Dataset:** [`clips/beir-nl-trec-covid`](https://huggingface.co/datasets/clips/beir-nl-trec-covid) • **License:** cc-by-4.0 • [Learn more →](https://colab.research.google.com/drive/1R99rjeAGt8S9IfAIRR3wS052sNu3Bjo-#scrollTo=4HduGW6xHnrZ)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Academic, Medical, Written | derived | machine-translated and verified |



#### TRECCOVID-PL

TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic.

**Dataset:** [`mteb/TRECCOVID-PL`](https://huggingface.co/datasets/mteb/TRECCOVID-PL) • **License:** not specified • [Learn more →](https://ir.nist.gov/covidSubmit/index.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic, Medical, Non-fiction, Written | derived | machine-translated |



#### TRECCOVID-VN

A translated dataset from TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/trec-covid-vn`](https://huggingface.co/datasets/GreenNode/trec-covid-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://ir.nist.gov/covidSubmit/index.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Academic, Medical, Written | derived | machine-translated and LM verified |



#### TV2Nordretrieval

News Article and corresponding summaries extracted from the Danish newspaper TV2 Nord.

**Dataset:** [`alexandrainst/nordjylland-news-summarization`](https://huggingface.co/datasets/alexandrainst/nordjylland-news-summarization) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/alexandrainst/nordjylland-news-summarization)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | dan | News, Non-fiction, Written | derived | found |



#### TempReasonL1

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l1.

**Dataset:** [`RAR-b/TempReason-l1`](https://huggingface.co/datasets/RAR-b/TempReason-l1) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### TempReasonL2Context

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l2-context.

**Dataset:** [`RAR-b/TempReason-l2-context`](https://huggingface.co/datasets/RAR-b/TempReason-l2-context) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### TempReasonL2Fact

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l2-fact.

**Dataset:** [`RAR-b/TempReason-l2-fact`](https://huggingface.co/datasets/RAR-b/TempReason-l2-fact) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### TempReasonL2Pure

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l2-pure.

**Dataset:** [`RAR-b/TempReason-l2-pure`](https://huggingface.co/datasets/RAR-b/TempReason-l2-pure) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### TempReasonL3Context

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l3-context.

**Dataset:** [`RAR-b/TempReason-l3-context`](https://huggingface.co/datasets/RAR-b/TempReason-l3-context) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### TempReasonL3Fact

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l3-fact.

**Dataset:** [`RAR-b/TempReason-l3-fact`](https://huggingface.co/datasets/RAR-b/TempReason-l3-fact) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### TempReasonL3Pure

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l3-pure.

**Dataset:** [`RAR-b/TempReason-l3-pure`](https://huggingface.co/datasets/RAR-b/TempReason-l3-pure) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### TopiOCQA

TopiOCQA (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset) is information-seeking conversational dataset with challenging topic switching phenomena. It consists of conversation histories along with manually labelled relevant/gold passage.

**Dataset:** [`McGill-NLP/TopiOCQA`](https://huggingface.co/datasets/McGill-NLP/TopiOCQA) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://mcgill-nlp.github.io/topiocqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



#### TopiOCQAHardNegatives

TopiOCQA (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset) is information-seeking conversational dataset with challenging topic switching phenomena. It consists of conversation histories along with manually labelled relevant/gold passage. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/TopiOCQA_validation_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/TopiOCQA_validation_top_250_only_w_correct-v2) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://mcgill-nlp.github.io/topiocqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



#### Touche2020

Touché Task 1: Argument Retrieval for Controversial Questions

**Dataset:** [`mteb/touche2020`](https://huggingface.co/datasets/mteb/touche2020) • **License:** cc-by-sa-4.0 • [Learn more →](https://webis.de/events/touche-20/shared-task-1.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic | human-annotated | found |



#### Touche2020-Fa

Touche2020-Fa

**Dataset:** [`MCINext/touche2020-fa`](https://huggingface.co/datasets/MCINext/touche2020-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/touche2020-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Spoken | derived | found |



#### Touche2020-NL

Touché Task 1: Argument Retrieval for Controversial Questions. Touche2020-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-webis-touche2020`](https://huggingface.co/datasets/clips/beir-nl-webis-touche2020) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-webis-touche2020)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Academic, Non-fiction | derived | machine-translated and verified |



#### Touche2020-PL

Touché Task 1: Argument Retrieval for Controversial Questions

**Dataset:** [`mteb/Touche2020-PL`](https://huggingface.co/datasets/mteb/Touche2020-PL) • **License:** not specified • [Learn more →](https://webis.de/events/touche-20/shared-task-1.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic | derived | machine-translated |



#### Touche2020-VN

A translated dataset from Touché Task 1: Argument Retrieval for Controversial Questions
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/webis-touche2020-vn`](https://huggingface.co/datasets/GreenNode/webis-touche2020-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://webis.de/events/touche-20/shared-task-1.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | vie | Academic | derived | machine-translated and LM verified |



#### Touche2020Retrieval.v3

Touché Task 1: Argument Retrieval for Controversial Questions

**Dataset:** [`mteb/webis-touche2020-v3`](https://huggingface.co/datasets/mteb/webis-touche2020-v3) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/castorini/touche-error-analysis)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic | human-annotated | found |



#### TurHistQuadRetrieval

Question Answering dataset on Ottoman History in Turkish

**Dataset:** [`asparius/TurHistQuAD`](https://huggingface.co/datasets/asparius/TurHistQuAD) • **License:** mit • [Learn more →](https://github.com/okanvk/Turkish-Reading-Comprehension-Question-Answering-Dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | tur | Academic, Encyclopaedic, Non-fiction, Written | derived | found |



#### TwitterHjerneRetrieval

Danish question asked on Twitter with the Hashtag #Twitterhjerne ('Twitter brain') and their corresponding answer.

**Dataset:** [`sorenmulli/da-hashtag-twitterhjerne`](https://huggingface.co/datasets/sorenmulli/da-hashtag-twitterhjerne) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sorenmulli/da-hashtag-twitterhjerne)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | dan | Social, Written | derived | found |



#### VDRMultilingualRetrieval

Multilingual Visual Document retrieval Dataset covering 5 languages: Italian, Spanish, English, French and German

**Dataset:** [`llamaindex/vdr-multilingual-test`](https://huggingface.co/datasets/llamaindex/vdr-multilingual-test) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/llamaindex/vdr-multilingual-test)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to image, text (it2it) | ndcg_at_5 | deu, eng, fra, ita, spa | Web | LM-generated | found |



#### VideoRetrieval

VideoRetrieval

**Dataset:** [`mteb/VideoRetrieval`](https://huggingface.co/datasets/mteb/VideoRetrieval) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2203.03367)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | not specified | not specified | not specified |



#### VieQuADRetrieval

A Vietnamese dataset for evaluating Machine Reading Comprehension from Wikipedia articles.

**Dataset:** [`taidng/UIT-ViQuAD2.0`](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0) • **License:** mit • [Learn more →](https://aclanthology.org/2020.coling-main.233.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Encyclopaedic, Non-fiction, Written | human-annotated | found |



#### WebFAQRetrieval

WebFAQ is a broad-coverage corpus of natural question-answer pairs in 75 languages, gathered from FAQ pages on the web.

**Dataset:** [`PaDaS-Lab/webfaq-retrieval`](https://huggingface.co/datasets/PaDaS-Lab/webfaq-retrieval) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/PaDaS-Lab)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, aze, ben, bul, cat, ... (49) | Web, Written | derived | found |



#### WikiSQLRetrieval

A code retrieval task based on WikiSQL dataset with natural language questions and corresponding SQL queries. Each query is a natural language question (e.g., 'What is the name of the team that has scored the most goals?'), and the corpus contains SQL query implementations. The task is to retrieve the correct SQL query that answers the natural language question. Queries are natural language questions while the corpus contains SQL SELECT statements with proper syntax and logic for querying database tables.

**Dataset:** [`embedding-benchmark/WikiSQL_mteb`](https://huggingface.co/datasets/embedding-benchmark/WikiSQL_mteb) • **License:** bsd-3-clause • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/WikiSQL_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | ndcg_at_10 | eng, sql | Programming | expert-annotated | found |



#### WikipediaRetrievalMultilingual

The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.

**Dataset:** [`mteb/WikipediaRetrievalMultilingual`](https://huggingface.co/datasets/mteb/WikipediaRetrievalMultilingual) • **License:** cc-by-sa-3.0 • [Learn more →](https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-multilingual-queries)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ben, bul, ces, dan, deu, ... (16) | Encyclopaedic, Written | LM-generated and reviewed | LM-generated and verified |



#### WinoGrande

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on winogrande.

**Dataset:** [`RAR-b/winogrande`](https://huggingface.co/datasets/RAR-b/winogrande) • **License:** not specified • [Learn more →](https://winogrande.allenai.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



#### XMarket

XMarket

**Dataset:** [`jinaai/xmarket_ml`](https://huggingface.co/datasets/jinaai/xmarket_ml) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jinaai/xmarket_ml)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | deu, eng, spa | not specified | not specified | not specified |



#### XPQARetrieval

XPQARetrieval

**Dataset:** [`jinaai/xpqa`](https://huggingface.co/datasets/jinaai/xpqa) • **License:** cdla-sharing-1.0 • [Learn more →](https://arxiv.org/abs/2305.09249)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, cmn, deu, eng, fra, ... (13) | Reviews, Written | human-annotated | found |



#### XQuADRetrieval

XQuAD is a benchmark dataset for evaluating cross-lingual question answering performance. It is repurposed retrieving relevant context for each question.

**Dataset:** [`google/xquad`](https://huggingface.co/datasets/google/xquad) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/xquad)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | arb, deu, ell, eng, hin, ... (12) | Web, Written | human-annotated | created |



#### ZacLegalTextRetrieval

Zalo Legal Text documents

**Dataset:** [`GreenNode/zalo-ai-legal-text-retrieval-vn`](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn) • **License:** mit • [Learn more →](https://challenge.zalo.ai/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Legal | human-annotated | found |



#### mMARCO-NL

mMARCO is a multi-lingual (translated) collection of datasets focused on deep learning in search

**Dataset:** [`clips/beir-nl-mmarco`](https://huggingface.co/datasets/clips/beir-nl-mmarco) • **License:** apache-2.0 • [Learn more →](https://github.com/unicamp-dl/mMARCO)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Web, Written | derived | machine-translated and verified |
