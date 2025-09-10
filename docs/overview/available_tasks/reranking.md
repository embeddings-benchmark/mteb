
# Reranking

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 29 

#### AlloprofReranking

This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum curated by a large number of teachers to students on all subjects taught from in primary and secondary school

**Dataset:** [`mteb/AlloprofReranking`](https://huggingface.co/datasets/mteb/AlloprofReranking) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/antoinelb7/alloprof)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | fra | Academic, Web, Written | expert-annotated | found |



#### AskUbuntuDupQuestions

AskUbuntu Question Dataset - Questions from AskUbuntu with manual annotations marking pairs of questions as similar or non-similar

**Dataset:** [`mteb/AskUbuntuDupQuestions`](https://huggingface.co/datasets/mteb/AskUbuntuDupQuestions) • **License:** not specified • [Learn more →](https://github.com/taolei87/askubuntu)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | eng | Programming, Web | human-annotated | found |



#### AskUbuntuDupQuestions-VN

A translated dataset from AskUbuntu Question Dataset - Questions from AskUbuntu with manual annotations marking pairs of questions as similar or non-similar
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/askubuntudupquestions-reranking-vn`](https://huggingface.co/datasets/GreenNode/askubuntudupquestions-reranking-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/taolei87/askubuntu)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | map | vie | Programming, Web | derived | machine-translated and LM verified |



#### BuiltBenchReranking

Reranking of built asset entity type/class descriptions given a query describing an entity as represented in well-established industry classification systems such as Uniclass, IFC, etc.

**Dataset:** [`mteb/BuiltBenchReranking`](https://huggingface.co/datasets/mteb/BuiltBenchReranking) • **License:** cc-by-nd-4.0 • [Learn more →](https://arxiv.org/abs/2411.12056)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map | eng | Engineering, Written | derived | created |



#### CMedQAv1-reranking

Chinese community medical question answering

**Dataset:** [`mteb/CMedQAv1-reranking`](https://huggingface.co/datasets/mteb/CMedQAv1-reranking) • **License:** not specified • [Learn more →](https://github.com/zhangsheng93/cMedQA)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | cmn | Medical, Written | expert-annotated | found |



#### CMedQAv2-reranking

Chinese community medical question answering

**Dataset:** [`mteb/CMedQAv2-reranking`](https://huggingface.co/datasets/mteb/CMedQAv2-reranking) • **License:** not specified • [Learn more →](https://github.com/zhangsheng93/cMedQA2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | cmn | Medical, Written | not specified | not specified |



#### CodeRAGLibraryDocumentationSolutions

Evaluation of code library documentation retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant Python library documentation sections given code-related queries.

**Dataset:** [`code-rag-bench/library-documentation`](https://huggingface.co/datasets/code-rag-bench/library-documentation) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2406.14497)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | python | Programming | derived | found |



#### CodeRAGOnlineTutorials

Evaluation of online programming tutorial retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant tutorials from online platforms given code-related queries.

**Dataset:** [`code-rag-bench/online-tutorials`](https://huggingface.co/datasets/code-rag-bench/online-tutorials) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2406.14497)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | python | Programming | derived | found |



#### CodeRAGProgrammingSolutions

Evaluation of programming solution retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant programming solutions given code-related queries.

**Dataset:** [`code-rag-bench/programming-solutions`](https://huggingface.co/datasets/code-rag-bench/programming-solutions) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2406.14497)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | python | Programming | derived | found |



#### CodeRAGStackoverflowPosts

Evaluation of StackOverflow post retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant StackOverflow posts given code-related queries.

**Dataset:** [`code-rag-bench/stackoverflow-posts`](https://huggingface.co/datasets/code-rag-bench/stackoverflow-posts) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2406.14497)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | python | Programming | derived | found |



#### ESCIReranking



**Dataset:** [`mteb/ESCIReranking`](https://huggingface.co/datasets/mteb/ESCIReranking) • **License:** apache-2.0 • [Learn more →](https://github.com/amazon-science/esci-data/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | eng, jpn, spa | Written | derived | created |



#### JQaRAReranking

JQaRA: Japanese Question Answering with Retrieval Augmentation  - 検索拡張(RAG)評価のための日本語 Q&A データセット. JQaRA is an information retrieval task for questions against 100 candidate data (including one or more correct answers).

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/hotchpotch/JQaRA)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map | jpn | Encyclopaedic, Non-fiction, Written | derived | found |



#### JaCWIRReranking

JaCWIR is a small-scale Japanese information retrieval evaluation dataset consisting of 5000 question texts and approximately 500k web page titles and web page introductions or summaries (meta descriptions, etc.). The question texts are created based on one of the 500k web pages, and that data is used as a positive example for the question text.

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/hotchpotch/JaCWIR)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map | jpn | Web, Written | derived | found |



#### MIRACLReranking

MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages.

**Dataset:** [`mteb/MIRACLReranking`](https://huggingface.co/datasets/mteb/MIRACLReranking) • **License:** cc-by-sa-4.0 • [Learn more →](https://project-miracl.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, ben, deu, eng, fas, ... (18) | Encyclopaedic, Written | expert-annotated | created |



#### MMarcoReranking

mMARCO is a multilingual version of the MS MARCO passage ranking dataset

**Dataset:** [`mteb/MMarcoReranking`](https://huggingface.co/datasets/mteb/MMarcoReranking) • **License:** not specified • [Learn more →](https://github.com/unicamp-dl/mMARCO)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | cmn | not specified | not specified | not specified |



#### MindSmallReranking

Microsoft News Dataset: A Large-Scale English Dataset for News Recommendation Research

**Dataset:** [`mteb/MindSmallReranking`](https://huggingface.co/datasets/mteb/MindSmallReranking) • **License:** https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf • [Learn more →](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_over_subqueries_map_at_1000 | eng | News, Written | expert-annotated | found |



#### NamaaMrTydiReranking

Mr. TyDi is a multi-lingual benchmark dataset built on TyDi, covering eleven typologically diverse languages. It is designed for monolingual retrieval, specifically to evaluate ranking with learned dense representations. This dataset adapts the arabic test split for Reranking evaluation purposes by the addition of multiple (Hard) Negatives to each query and positive

**Dataset:** [`mteb/NamaaMrTydiReranking`](https://huggingface.co/datasets/mteb/NamaaMrTydiReranking) • **License:** cc-by-sa-3.0 • [Learn more →](https://huggingface.co/NAMAA-Space)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map | ara | Encyclopaedic, Written | human-annotated | found |



#### NevIR

Paired evaluation of real world negation in retrieval, with questions and passages. Since models generally prefer one passage over the other always, there are two questions that the model must get right to understand the negation (hence the `paired_accuracy` metric).

**Dataset:** [`orionweller/NevIR-mteb`](https://huggingface.co/datasets/orionweller/NevIR-mteb) • **License:** mit • [Learn more →](https://github.com/orionw/NevIR)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | paired_accuracy | eng | Web | human-annotated | created |



#### RuBQReranking

Paragraph reranking based on RuBQ 2.0. Give paragraphs that answer the question higher scores.

**Dataset:** [`mteb/RuBQReranking`](https://huggingface.co/datasets/mteb/RuBQReranking) • **License:** cc-by-sa-4.0 • [Learn more →](https://openreview.net/pdf?id=P5UQFFoQ4PJ)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | rus | Encyclopaedic, Written | human-annotated | created |



#### SciDocsRR

Ranking of related scientific papers based on their title.

**Dataset:** [`mteb/SciDocsRR`](https://huggingface.co/datasets/mteb/SciDocsRR) • **License:** cc-by-4.0 • [Learn more →](https://allenai.org/data/scidocs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | eng | Academic, Non-fiction, Written | not specified | found |



#### SciDocsRR-VN

A translated dataset from Ranking of related scientific papers based on their title.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/scidocs-reranking-vn`](https://huggingface.co/datasets/GreenNode/scidocs-reranking-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://allenai.org/data/scidocs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | map | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified |



#### StackOverflowDupQuestions

Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python

**Dataset:** [`mteb/StackOverflowDupQuestions`](https://huggingface.co/datasets/mteb/StackOverflowDupQuestions) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | eng | Blog, Programming, Written | derived | found |



#### StackOverflowDupQuestions-VN

A translated dataset from Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/stackoverflowdupquestions-reranking-vn`](https://huggingface.co/datasets/GreenNode/stackoverflowdupquestions-reranking-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | map | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified |



#### SyntecReranking

This dataset has been built from the Syntec Collective bargaining agreement.

**Dataset:** [`mteb/SyntecReranking`](https://huggingface.co/datasets/mteb/SyntecReranking) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/lyon-nlp/mteb-fr-reranking-syntec-s2p)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | fra | Legal, Written | human-annotated | found |



#### T2Reranking

T2Ranking: A large-scale Chinese Benchmark for Passage Ranking

**Dataset:** [`mteb/T2Reranking`](https://huggingface.co/datasets/mteb/T2Reranking) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2304.03679)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | cmn | not specified | not specified | not specified |



#### VoyageMMarcoReranking

a hard-negative augmented version of the Japanese MMARCO dataset as used in Voyage AI Evaluation Suite

**Dataset:** [`mteb/VoyageMMarcoReranking`](https://huggingface.co/datasets/mteb/VoyageMMarcoReranking) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2312.16144)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | jpn | Academic, Non-fiction, Written | derived | found |



#### WebLINXCandidatesReranking

WebLINX is a large-scale benchmark of 100K interactions across 2300 expert demonstrations of conversational web navigation. The reranking task focuses on finding relevant elements at every given step in the trajectory.

**Dataset:** [`mteb/WebLINXCandidatesReranking`](https://huggingface.co/datasets/mteb/WebLINXCandidatesReranking) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://mcgill-nlp.github.io/weblinx)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | mrr_at_10 | eng | Academic, Web, Written | expert-annotated | created |



#### WikipediaRerankingMultilingual

The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.

**Dataset:** [`mteb/WikipediaRerankingMultilingual`](https://huggingface.co/datasets/mteb/WikipediaRerankingMultilingual) • **License:** cc-by-sa-3.0 • [Learn more →](https://huggingface.co/datasets/ellamind/wikipedia-2023-11-reranking-multilingual)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | ben, bul, ces, dan, deu, ... (16) | Encyclopaedic, Written | LM-generated and reviewed | LM-generated and verified |



#### XGlueWPRReranking

XGLUE is a new benchmark dataset to evaluate the performance of cross-lingual pre-trained models
        with respect to cross-lingual natural language understanding and generation. XGLUE is composed of 11 tasks spans 19 languages.

**Dataset:** [`forresty/xglue`](https://huggingface.co/datasets/forresty/xglue) • **License:** http://hdl.handle.net/11234/1-3105 • [Learn more →](https://github.com/microsoft/XGLUE)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map | deu, eng, fra, ita, por, ... (7) | Written | human-annotated | found |
