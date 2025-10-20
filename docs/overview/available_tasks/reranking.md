
# Reranking

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 39

#### AlloprofReranking

This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum curated by a large number of teachers to students on all subjects taught from in primary and secondary school

**Dataset:** [`mteb/AlloprofReranking`](https://huggingface.co/datasets/mteb/AlloprofReranking) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/antoinelb7/alloprof)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | fra | Academic, Web, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{lef23,
      author = {Lefebvre-Brossard, Antoine and Gazaille, Stephane and Desmarais, Michel C.},
      copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International},
      doi = {10.48550/ARXIV.2302.07738},
      keywords = {Computation and Language (cs.CL), Information Retrieval (cs.IR), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
      publisher = {arXiv},
      title = {Alloprof: a new French question-answer education dataset and its use in an information retrieval case study},
      url = {https://arxiv.org/abs/2302.07738},
      year = {2023},
    }

    ```




#### AskUbuntuDupQuestions

AskUbuntu Question Dataset - Questions from AskUbuntu with manual annotations marking pairs of questions as similar or non-similar

**Dataset:** [`mteb/AskUbuntuDupQuestions`](https://huggingface.co/datasets/mteb/AskUbuntuDupQuestions) • **License:** not specified • [Learn more →](https://github.com/taolei87/askubuntu)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | eng | Programming, Web | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{wang-2021-TSDAE,
      author = {Wang, Kexin and Reimers, Nils and  Gurevych, Iryna},
      journal = {arXiv preprint arXiv:2104.06979},
      month = {4},
      title = {TSDAE: Using Transformer-based Sequential Denoising Auto-Encoderfor Unsupervised Sentence Embedding Learning},
      url = {https://arxiv.org/abs/2104.06979},
      year = {2021},
    }

    ```




#### AskUbuntuDupQuestions-VN

A translated dataset from AskUbuntu Question Dataset - Questions from AskUbuntu with manual annotations marking pairs of questions as similar or non-similar
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`mteb/AskUbuntuDupQuestions-VN`](https://huggingface.co/datasets/mteb/AskUbuntuDupQuestions-VN) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/taolei87/askubuntu)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | vie | Programming, Web | derived | machine-translated and LM verified |



??? quote "Citation"


    ```bibtex

    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }

    ```




#### BuiltBenchReranking

Reranking of built asset entity type/class descriptions given a query describing an entity as represented in well-established industry classification systems such as Uniclass, IFC, etc.

**Dataset:** [`mteb/BuiltBenchReranking`](https://huggingface.co/datasets/mteb/BuiltBenchReranking) • **License:** cc-by-nd-4.0 • [Learn more →](https://arxiv.org/abs/2411.12056)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | eng | Engineering, Written | derived | created |



??? quote "Citation"


    ```bibtex

    @article{shahinmoghadam2024benchmarking,
      author = {Shahinmoghadam, Mehrzad and Motamedi, Ali},
      journal = {arXiv preprint arXiv:2411.12056},
      title = {Benchmarking pre-trained text embedding models in aligning built asset information},
      year = {2024},
    }

    ```




#### CMedQAv1-reranking

Chinese community medical question answering

**Dataset:** [`mteb/CMedQAv1-reranking`](https://huggingface.co/datasets/mteb/CMedQAv1-reranking) • **License:** not specified • [Learn more →](https://github.com/zhangsheng93/cMedQA)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | cmn | Medical, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{zhang2017chinese,
      author = {Zhang, Sheng and Zhang, Xin and Wang, Hui and Cheng, Jiajun and Li, Pei and Ding, Zhaoyun},
      journal = {Applied Sciences},
      number = {8},
      pages = {767},
      publisher = {Multidisciplinary Digital Publishing Institute},
      title = {Chinese Medical Question Answer Matching Using End-to-End Character-Level Multi-Scale CNNs},
      volume = {7},
      year = {2017},
    }

    ```




#### CMedQAv2-reranking

Chinese community medical question answering

**Dataset:** [`mteb/CMedQAv2-reranking`](https://huggingface.co/datasets/mteb/CMedQAv2-reranking) • **License:** not specified • [Learn more →](https://github.com/zhangsheng93/cMedQA2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | cmn | Medical, Written | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @article{8548603,
      author = {S. Zhang and X. Zhang and H. Wang and L. Guo and S. Liu},
      doi = {10.1109/ACCESS.2018.2883637},
      issn = {2169-3536},
      journal = {IEEE Access},
      keywords = {Biomedical imaging;Data mining;Semantics;Medical services;Feature extraction;Knowledge discovery;Medical question answering;interactive attention;deep learning;deep neural networks},
      month = {},
      number = {},
      pages = {74061-74071},
      title = {Multi-Scale Attentive Interaction Networks for Chinese Medical Question Answer Selection},
      volume = {6},
      year = {2018},
    }

    ```




#### CodeRAGLibraryDocumentationSolutions

Evaluation of code library documentation retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant Python library documentation sections given code-related queries.

**Dataset:** [`code-rag-bench/library-documentation`](https://huggingface.co/datasets/code-rag-bench/library-documentation) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2406.14497)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | python | Programming | derived | found |



??? quote "Citation"


    ```bibtex

        @misc{wang2024coderagbenchretrievalaugmentcode,
      archiveprefix = {arXiv},
      author = {Zora Zhiruo Wang and Akari Asai and Xinyan Velocity Yu and Frank F. Xu and Yiqing Xie and Graham Neubig and Daniel Fried},
      eprint = {2406.14497},
      primaryclass = {cs.SE},
      title = {CodeRAG-Bench: Can Retrieval Augment Code Generation?},
      url = {https://arxiv.org/abs/2406.14497},
      year = {2024},
    }

    ```




#### CodeRAGOnlineTutorials

Evaluation of online programming tutorial retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant tutorials from online platforms given code-related queries.

**Dataset:** [`code-rag-bench/online-tutorials`](https://huggingface.co/datasets/code-rag-bench/online-tutorials) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2406.14497)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | python | Programming | derived | found |



??? quote "Citation"


    ```bibtex

        @misc{wang2024coderagbenchretrievalaugmentcode,
      archiveprefix = {arXiv},
      author = {Zora Zhiruo Wang and Akari Asai and Xinyan Velocity Yu and Frank F. Xu and Yiqing Xie and Graham Neubig and Daniel Fried},
      eprint = {2406.14497},
      primaryclass = {cs.SE},
      title = {CodeRAG-Bench: Can Retrieval Augment Code Generation?},
      url = {https://arxiv.org/abs/2406.14497},
      year = {2024},
    }

    ```




#### CodeRAGProgrammingSolutions

Evaluation of programming solution retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant programming solutions given code-related queries.

**Dataset:** [`code-rag-bench/programming-solutions`](https://huggingface.co/datasets/code-rag-bench/programming-solutions) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2406.14497)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | python | Programming | derived | found |



??? quote "Citation"


    ```bibtex

        @misc{wang2024coderagbenchretrievalaugmentcode,
      archiveprefix = {arXiv},
      author = {Zora Zhiruo Wang and Akari Asai and Xinyan Velocity Yu and Frank F. Xu and Yiqing Xie and Graham Neubig and Daniel Fried},
      eprint = {2406.14497},
      primaryclass = {cs.SE},
      title = {CodeRAG-Bench: Can Retrieval Augment Code Generation?},
      url = {https://arxiv.org/abs/2406.14497},
      year = {2024},
    }

    ```




#### CodeRAGStackoverflowPosts

Evaluation of StackOverflow post retrieval using CodeRAG-Bench. Tests the ability to retrieve relevant StackOverflow posts given code-related queries.

**Dataset:** [`code-rag-bench/stackoverflow-posts`](https://huggingface.co/datasets/code-rag-bench/stackoverflow-posts) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2406.14497)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | python | Programming | derived | found |



??? quote "Citation"


    ```bibtex

        @misc{wang2024coderagbenchretrievalaugmentcode,
      archiveprefix = {arXiv},
      author = {Zora Zhiruo Wang and Akari Asai and Xinyan Velocity Yu and Frank F. Xu and Yiqing Xie and Graham Neubig and Daniel Fried},
      eprint = {2406.14497},
      primaryclass = {cs.SE},
      title = {CodeRAG-Bench: Can Retrieval Augment Code Generation?},
      url = {https://arxiv.org/abs/2406.14497},
      year = {2024},
    }

    ```




#### ESCIReranking



**Dataset:** [`mteb/ESCIReranking`](https://huggingface.co/datasets/mteb/ESCIReranking) • **License:** apache-2.0 • [Learn more →](https://github.com/amazon-science/esci-data/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | eng, jpn, spa | Written | derived | created |



??? quote "Citation"


    ```bibtex
    @article{reddy2022shopping,
      archiveprefix = {arXiv},
      author = {Chandan K. Reddy and Lluís Màrquez and Fran Valero and Nikhil Rao and Hugo Zaragoza and Sambaran Bandyopadhyay and Arnab Biswas and Anlu Xing and Karthik Subbian},
      eprint = {2206.06588},
      title = {Shopping Queries Dataset: A Large-Scale {ESCI} Benchmark for Improving Product Search},
      year = {2022},
    }
    ```




#### HUMECore17InstructionReranking

Human evaluation subset of Core17 instruction retrieval dataset for reranking evaluation.

**Dataset:** [`mteb/HUMECore17InstructionReranking`](https://huggingface.co/datasets/mteb/HUMECore17InstructionReranking) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2403.15246)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | eng | News, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{weller2024followir,
      archiveprefix = {arXiv},
      author = {Orion Weller and Benjamin Chang and Sean MacAvaney and Kyle Lo and Arman Cohan and Benjamin Van Durme and Dawn Lawrie and Luca Soldaini},
      eprint = {2403.15246},
      primaryclass = {cs.IR},
      title = {FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions},
      year = {2024},
    }

    ```




#### HUMENews21InstructionReranking

Human evaluation subset of News21 instruction retrieval dataset for reranking evaluation.

**Dataset:** [`mteb/HUMENews21InstructionReranking`](https://huggingface.co/datasets/mteb/HUMENews21InstructionReranking) • **License:** not specified • [Learn more →](https://trec.nist.gov/data/news2021.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | eng | News, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{soboroff2021trec,
      author = {Soboroff, Ian and Macdonald, Craig and McCreadie, Richard},
      booktitle = {TREC},
      title = {TREC 2021 News Track Overview},
      year = {2021},
    }

    ```




#### HUMERobust04InstructionReranking

Human evaluation subset of Robust04 instruction retrieval dataset for reranking evaluation.

**Dataset:** [`mteb/HUMERobust04InstructionReranking`](https://huggingface.co/datasets/mteb/HUMERobust04InstructionReranking) • **License:** not specified • [Learn more →](https://trec.nist.gov/data/robust/04.guidelines.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | eng | News, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{voorhees2005trec,
      author = {Voorhees, Ellen M},
      booktitle = {TREC},
      title = {TREC 2004 Robust Retrieval Track Overview},
      year = {2005},
    }

    ```




#### HUMEWikipediaRerankingMultilingual

Human evaluation subset of Wikipedia reranking dataset across multiple languages.

**Dataset:** [`mteb/HUMEWikipediaRerankingMultilingual`](https://huggingface.co/datasets/mteb/HUMEWikipediaRerankingMultilingual) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/ellamind/wikipedia-2023-11-reranking-multilingual)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | dan, eng, nob | Encyclopaedic, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{wikipedia_reranking_2023,
      author = {Ellamind},
      title = {Wikipedia 2023-11 Reranking Multilingual Dataset},
      url = {https://github.com/ellamind/wikipedia-2023-11-reranking-multilingual},
      year = {2023},
    }

    ```




#### JQaRAReranking

JQaRA: Japanese Question Answering with Retrieval Augmentation  - 検索拡張(RAG)評価のための日本語 Q&A データセット. JQaRA is an information retrieval task for questions against 100 candidate data (including one or more correct answers).

**Dataset:** [`mteb/JQaRAReranking`](https://huggingface.co/datasets/mteb/JQaRAReranking) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/hotchpotch/JQaRA)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | jpn | Encyclopaedic, Non-fiction, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{yuichi-tateno-2024-jqara,
      author = {Yuichi Tateno},
      title = {JQaRA: Japanese Question Answering with Retrieval Augmentation - 検索拡張(RAG)評価のための日本語Q&Aデータセット},
      url = {https://huggingface.co/datasets/hotchpotch/JQaRA},
    }

    ```




#### JaCWIRReranking

JaCWIR is a small-scale Japanese information retrieval evaluation dataset consisting of 5000 question texts and approximately 500k web page titles and web page introductions or summaries (meta descriptions, etc.). The question texts are created based on one of the 500k web pages, and that data is used as a positive example for the question text.

**Dataset:** [`mteb/JaCWIRReranking`](https://huggingface.co/datasets/mteb/JaCWIRReranking) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/hotchpotch/JaCWIR)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | jpn | Web, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{yuichi-tateno-2024-jacwir,
      author = {Yuichi Tateno},
      title = {JaCWIR: Japanese Casual Web IR - 日本語情報検索評価のための小規模でカジュアルなWebタイトルと概要のデータセット},
      url = {https://huggingface.co/datasets/hotchpotch/JaCWIR},
    }

    ```




#### LocBenchRR

Software Issue Localization.

**Dataset:** [`mteb/LocBenchRR`](https://huggingface.co/datasets/mteb/LocBenchRR) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.09089)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_10 | eng, python | Programming, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{chen2025locagentgraphguidedllmagents,
      archiveprefix = {arXiv},
      author = {Zhaoling Chen and Xiangru Tang and Gangda Deng and Fang Wu and Jialong Wu and Zhiwei Jiang and Viktor Prasanna and Arman Cohan and Xingyao Wang},
      eprint = {2503.09089},
      primaryclass = {cs.SE},
      title = {LocAgent: Graph-Guided LLM Agents for Code Localization},
      url = {https://arxiv.org/abs/2503.09089},
      year = {2025},
    }

    ```




#### MIRACLReranking

MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages.

**Dataset:** [`mteb/MIRACLReranking`](https://huggingface.co/datasets/mteb/MIRACLReranking) • **License:** cc-by-sa-4.0 • [Learn more →](https://project-miracl.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, ben, deu, eng, fas, ... (18) | Encyclopaedic, Written | expert-annotated | created |



??? quote "Citation"


    ```bibtex
    @article{10.1162/tacl_a_00595,
      author = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
      doi = {10.1162/tacl_a_00595},
      issn = {2307-387X},
      journal = {Transactions of the Association for Computational Linguistics},
      month = {09},
      pages = {1114-1131},
      title = {{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}},
      volume = {11},
      year = {2023},
    }
    ```




#### MMarcoReranking

mMARCO is a multilingual version of the MS MARCO passage ranking dataset

**Dataset:** [`mteb/MMarcoReranking`](https://huggingface.co/datasets/mteb/MMarcoReranking) • **License:** not specified • [Learn more →](https://github.com/unicamp-dl/mMARCO)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | cmn | not specified | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @misc{bonifacio2021mmarco,
      archiveprefix = {arXiv},
      author = {Luiz Henrique Bonifacio and Vitor Jeronymo and Hugo Queiroz Abonizio and Israel Campiotti and Marzieh Fadaee and  and Roberto Lotufo and Rodrigo Nogueira},
      eprint = {2108.13897},
      primaryclass = {cs.CL},
      title = {mMARCO: A Multilingual Version of MS MARCO Passage Ranking Dataset},
      year = {2021},
    }

    ```




#### MindSmallReranking

Microsoft News Dataset: A Large-Scale English Dataset for News Recommendation Research

**Dataset:** [`mteb/MindSmallReranking`](https://huggingface.co/datasets/mteb/MindSmallReranking) • **License:** https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf • [Learn more →](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | max_over_subqueries_map_at_1000 | eng | News, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{wu-etal-2020-mind,
      address = {Online},
      author = {Wu, Fangzhao  and Qiao, Ying  and Chen, Jiun-Hung  and Wu, Chuhan  and Qi,
    Tao  and Lian, Jianxun  and Liu, Danyang  and Xie, Xing  and Gao, Jianfeng  and Wu, Winnie  and Zhou, Ming},
      booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
      doi = {10.18653/v1/2020.acl-main.331},
      editor = {Jurafsky, Dan  and Chai, Joyce  and Schluter, Natalie  and Tetreault, Joel},
      month = jul,
      pages = {3597--3606},
      publisher = {Association for Computational Linguistics},
      title = {{MIND}: A Large-scale Dataset for News
    Recommendation},
      url = {https://aclanthology.org/2020.acl-main.331},
      year = {2020},
    }

    ```




#### MultiSWEbenchRR

Multilingual Software Issue Localization.

**Dataset:** [`mteb/MultiSWEbenchRR`](https://huggingface.co/datasets/mteb/MultiSWEbenchRR) • **License:** mit • [Learn more →](https://multi-swe-bench.github.io/#/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_10 | eng, python | Programming, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{zan2025multiswebench,
      archiveprefix = {arXiv},
      author = {Daoguang Zan and Zhirong Huang and Wei Liu and Hanwu Chen and Linhao Zhang and Shulin Xin and Lu Chen and Qi Liu and Xiaojian Zhong and Aoyan Li and Siyao Liu and Yongsheng Xiao and Liangqiang Chen and Yuyu Zhang and Jing Su and Tianyu Liu and Rui Long and Kai Shen and Liang Xiang},
      eprint = {2504.02605},
      primaryclass = {cs.SE},
      title = {Multi-SWE-bench: A Multilingual Benchmark for Issue Resolving},
      url = {https://arxiv.org/abs/2504.02605},
      year = {2025},
    }

    ```




#### NamaaMrTydiReranking

Mr. TyDi is a multi-lingual benchmark dataset built on TyDi, covering eleven typologically diverse languages. It is designed for monolingual retrieval, specifically to evaluate ranking with learned dense representations. This dataset adapts the arabic test split for Reranking evaluation purposes by the addition of multiple (Hard) Negatives to each query and positive

**Dataset:** [`mteb/NamaaMrTydiReranking`](https://huggingface.co/datasets/mteb/NamaaMrTydiReranking) • **License:** cc-by-sa-3.0 • [Learn more →](https://huggingface.co/NAMAA-Space)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | ara | Encyclopaedic, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{muennighoff2022mteb,
      author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Lo{\"\\i}c and Reimers, Nils},
      doi = {10.48550/ARXIV.2210.07316},
      journal = {arXiv preprint arXiv:2210.07316},
      publisher = {arXiv},
      title = {MTEB: Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2210.07316},
      year = {2022},
    }

    ```




#### NevIR

Paired evaluation of real world negation in retrieval, with questions and passages. Since models generally prefer one passage over the other always, there are two questions that the model must get right to understand the negation (hence the `paired_accuracy` metric).

**Dataset:** [`orionweller/NevIR-mteb`](https://huggingface.co/datasets/orionweller/NevIR-mteb) • **License:** mit • [Learn more →](https://github.com/orionw/NevIR)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | paired_accuracy | eng | Web | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{Weller2023NevIRNI,
      author = {{Orion Weller and Dawn J Lawrie and Benjamin Van Durme}},
      booktitle = {{Conference of the European Chapter of the Association for Computational Linguistics}},
      title = {{NevIR: Negation in Neural Information Retrieval}},
      url = {{https://api.semanticscholar.org/CorpusID:258676146}},
      year = {{2023}},
    }

    ```




#### RuBQReranking

Paragraph reranking based on RuBQ 2.0. Give paragraphs that answer the question higher scores.

**Dataset:** [`mteb/RuBQReranking`](https://huggingface.co/datasets/mteb/RuBQReranking) • **License:** cc-by-sa-4.0 • [Learn more →](https://openreview.net/pdf?id=P5UQFFoQ4PJ)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | rus | Encyclopaedic, Written | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{RuBQ2021,
      author = {Ivan Rybin and Vladislav Korablinov and Pavel Efimov and Pavel Braslavski},
      booktitle = {ESWC},
      pages = {532--547},
      title = {RuBQ 2.0: An Innovated Russian Question Answering Dataset},
      year = {2021},
    }

    ```




#### SWEPolyBenchRR

Multilingual Software Issue Localization.

**Dataset:** [`mteb/SWEPolyBenchRR`](https://huggingface.co/datasets/mteb/SWEPolyBenchRR) • **License:** mit • [Learn more →](https://amazon-science.github.io/SWE-PolyBench/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_10 | eng, python | Programming, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{rashid2025swepolybenchmultilanguagebenchmarkrepository,
      archiveprefix = {arXiv},
      author = {Muhammad Shihab Rashid and Christian Bock and Yuan Zhuang and Alexander Buchholz and Tim Esler and Simon Valentin and Luca Franceschi and Martin Wistuba and Prabhu Teja Sivaprasad and Woo Jung Kim and Anoop Deoras and Giovanni Zappella and Laurent Callot},
      eprint = {2504.08703},
      primaryclass = {cs.SE},
      title = {SWE-PolyBench: A multi-language benchmark for repository level evaluation of coding agents},
      url = {https://arxiv.org/abs/2504.08703},
      year = {2025},
    }

    ```




#### SWEbenchLiteRR

Software Issue Localization.

**Dataset:** [`mteb/SWEbenchLiteRR`](https://huggingface.co/datasets/mteb/SWEbenchLiteRR) • **License:** mit • [Learn more →](https://www.swebench.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_10 | eng, python | Programming, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{jimenez2024swebenchlanguagemodelsresolve,
      archiveprefix = {arXiv},
      author = {Carlos E. Jimenez and John Yang and Alexander Wettig and Shunyu Yao and Kexin Pei and Ofir Press and Karthik Narasimhan},
      eprint = {2310.06770},
      primaryclass = {cs.CL},
      title = {SWE-bench: Can Language Models Resolve Real-World GitHub Issues?},
      url = {https://arxiv.org/abs/2310.06770},
      year = {2024},
    }

    ```




#### SWEbenchMultilingualRR

Multilingual Software Issue Localization.

**Dataset:** [`mteb/SWEbenchMultilingualRR`](https://huggingface.co/datasets/mteb/SWEbenchMultilingualRR) • **License:** mit • [Learn more →](https://www.swebench.com/multilingual.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_10 | eng, python | Programming, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{yang2025swesmith,
      archiveprefix = {arXiv},
      author = {John Yang and Kilian Lieret and Carlos E. Jimenez and Alexander Wettig and Kabir Khandpur and Yanzhe Zhang and Binyuan Hui and Ofir Press and Ludwig Schmidt and Diyi Yang},
      eprint = {2504.21798},
      primaryclass = {cs.SE},
      title = {SWE-smith: Scaling Data for Software Engineering Agents},
      url = {https://arxiv.org/abs/2504.21798},
      year = {2025},
    }

    ```




#### SWEbenchVerifiedRR

Software Issue Localization for SWE-bench Verified

**Dataset:** [`mteb/SWEbenchVerifiedRR`](https://huggingface.co/datasets/mteb/SWEbenchVerifiedRR) • **License:** mit • [Learn more →](https://openai.com/index/introducing-swe-bench-verified/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_10 | eng, python | Programming, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{openai2024swebenchverified,
      author = {OpenAI},
      title = {Introducing swe-bench verified},
      url = {https://openai.com/index/introducing-swe-bench-verified/},
      year = {2024},
    }

    ```




#### SciDocsRR

Ranking of related scientific papers based on their title.

**Dataset:** [`mteb/SciDocsRR`](https://huggingface.co/datasets/mteb/SciDocsRR) • **License:** cc-by-4.0 • [Learn more →](https://allenai.org/data/scidocs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | eng | Academic, Non-fiction, Written | not specified | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{cohan-etal-2020-specter,
      address = {Online},
      author = {Cohan, Arman  and
    Feldman, Sergey  and
    Beltagy, Iz  and
    Downey, Doug  and
    Weld, Daniel},
      booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
      doi = {10.18653/v1/2020.acl-main.207},
      editor = {Jurafsky, Dan  and
    Chai, Joyce  and
    Schluter, Natalie  and
    Tetreault, Joel},
      month = jul,
      pages = {2270--2282},
      publisher = {Association for Computational Linguistics},
      title = {{SPECTER}: Document-level Representation Learning using Citation-informed Transformers},
      url = {https://aclanthology.org/2020.acl-main.207},
      year = {2020},
    }

    ```




#### SciDocsRR-VN

A translated dataset from Ranking of related scientific papers based on their title.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`mteb/SciDocsRR-VN`](https://huggingface.co/datasets/mteb/SciDocsRR-VN) • **License:** cc-by-sa-4.0 • [Learn more →](https://allenai.org/data/scidocs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified |



??? quote "Citation"


    ```bibtex

    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }

    ```




#### StackOverflowDupQuestions

Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python

**Dataset:** [`mteb/StackOverflowDupQuestions`](https://huggingface.co/datasets/mteb/StackOverflowDupQuestions) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | eng | Blog, Programming, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{Liu2018LinkSOAD,
      author = {Xueqing Liu and Chi Wang and Yue Leng and ChengXiang Zhai},
      journal = {Proceedings of the 4th ACM SIGSOFT International Workshop on NLP for Software Engineering},
      title = {LinkSO: a dataset for learning to retrieve similar question answer pairs on software development forums},
      url = {https://api.semanticscholar.org/CorpusID:53111679},
      year = {2018},
    }

    ```




#### StackOverflowDupQuestions-VN

A translated dataset from Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`mteb/StackOverflowDupQuestions-VN`](https://huggingface.co/datasets/mteb/StackOverflowDupQuestions-VN) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified |



??? quote "Citation"


    ```bibtex

    @misc{pham2025vnmtebvietnamesemassivetext,
      archiveprefix = {arXiv},
      author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
      eprint = {2507.21500},
      primaryclass = {cs.CL},
      title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
      url = {https://arxiv.org/abs/2507.21500},
      year = {2025},
    }

    ```




#### SyntecReranking

This dataset has been built from the Syntec Collective bargaining agreement.

**Dataset:** [`mteb/SyntecReranking`](https://huggingface.co/datasets/mteb/SyntecReranking) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/lyon-nlp/mteb-fr-reranking-syntec-s2p)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | fra | Legal, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{ciancone2024extending,
      archiveprefix = {arXiv},
      author = {Mathieu Ciancone and Imene Kerboua and Marion Schaeffer and Wissam Siblini},
      eprint = {2405.20468},
      primaryclass = {cs.CL},
      title = {Extending the Massive Text Embedding Benchmark to French},
      year = {2024},
    }

    ```




#### T2Reranking

T2Ranking: A large-scale Chinese Benchmark for Passage Ranking

**Dataset:** [`mteb/T2Reranking`](https://huggingface.co/datasets/mteb/T2Reranking) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2304.03679)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | cmn | not specified | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @misc{xie2023t2ranking,
      archiveprefix = {arXiv},
      author = {Xiaohui Xie and Qian Dong and Bingning Wang and Feiyang Lv and Ting Yao and Weinan Gan and Zhijing Wu and Xiangsheng Li and Haitao Li and Yiqun Liu and Jin Ma},
      eprint = {2304.03679},
      primaryclass = {cs.IR},
      title = {T2Ranking: A large-scale Chinese Benchmark for Passage Ranking},
      year = {2023},
    }

    ```




#### VoyageMMarcoReranking

a hard-negative augmented version of the Japanese MMARCO dataset as used in Voyage AI Evaluation Suite

**Dataset:** [`mteb/VoyageMMarcoReranking`](https://huggingface.co/datasets/mteb/VoyageMMarcoReranking) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2312.16144)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | jpn | Academic, Non-fiction, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{clavié2023jacolbert,
      archiveprefix = {arXiv},
      author = {Benjamin Clavié},
      eprint = {2312.16144},
      title = {JaColBERT and Hard Negatives, Towards Better Japanese-First Embeddings for Retrieval: Early Technical Report},
      year = {2023},
    }

    ```




#### WebLINXCandidatesReranking

WebLINX is a large-scale benchmark of 100K interactions across 2300 expert demonstrations of conversational web navigation. The reranking task focuses on finding relevant elements at every given step in the trajectory.

**Dataset:** [`mteb/WebLINXCandidatesReranking`](https://huggingface.co/datasets/mteb/WebLINXCandidatesReranking) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://mcgill-nlp.github.io/weblinx)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | mrr_at_10 | eng | Academic, Web, Written | expert-annotated | created |



??? quote "Citation"


    ```bibtex

    @misc{lù2024weblinx,
      archiveprefix = {arXiv},
      author = {Xing Han Lù and Zdeněk Kasner and Siva Reddy},
      eprint = {2402.05930},
      primaryclass = {cs.CL},
      title = {WebLINX: Real-World Website Navigation with Multi-Turn Dialogue},
      year = {2024},
    }

    ```




#### WikipediaRerankingMultilingual

The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.

**Dataset:** [`mteb/WikipediaRerankingMultilingual`](https://huggingface.co/datasets/mteb/WikipediaRerankingMultilingual) • **License:** cc-by-sa-3.0 • [Learn more →](https://huggingface.co/datasets/ellamind/wikipedia-2023-11-reranking-multilingual)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | ben, bul, ces, dan, deu, ... (18) | Encyclopaedic, Written | LM-generated and reviewed | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    @online{wikidump,
      author = {Wikimedia Foundation},
      title = {Wikimedia Downloads},
      url = {https://dumps.wikimedia.org},
    }

    ```




#### XGlueWPRReranking

XGLUE is a new benchmark dataset to evaluate the performance of cross-lingual pre-trained models
        with respect to cross-lingual natural language understanding and generation. XGLUE is composed of 11 tasks spans 19 languages.

**Dataset:** [`mteb/XGlueWPRReranking`](https://huggingface.co/datasets/mteb/XGlueWPRReranking) • **License:** http://hdl.handle.net/11234/1-3105 • [Learn more →](https://github.com/microsoft/XGLUE)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | map_at_1000 | deu, eng, fra, ita, por, ... (7) | Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{11234/1-3105,
      author = {Zeman, Daniel and Nivre, Joakim and Abrams, Mitchell and Aepli, No{\"e}mi and Agi{\'c}, {\v Z}eljko and Ahrenberg, Lars and Aleksandravi{\v c}i{\=u}t{\.e}, Gabriel{\.e} and Antonsen, Lene and Aplonova, Katya and Aranzabe, Maria Jesus and Arutie, Gashaw and Asahara, Masayuki and Ateyah, Luma and Attia, Mohammed and Atutxa, Aitziber and Augustinus, Liesbeth and Badmaeva, Elena and Ballesteros, Miguel and Banerjee, Esha and Bank, Sebastian and Barbu Mititelu, Verginica and Basmov, Victoria and Batchelor, Colin and Bauer, John and Bellato, Sandra and Bengoetxea, Kepa and Berzak, Yevgeni and Bhat, Irshad Ahmad and Bhat, Riyaz Ahmad and Biagetti, Erica and Bick, Eckhard and Bielinskien{\.e}, Agn{\.e} and Blokland, Rogier and Bobicev, Victoria and Boizou, Lo{\"{\i}}c and Borges V{\"o}lker, Emanuel and B{\"o}rstell, Carl and Bosco, Cristina and Bouma, Gosse and Bowman, Sam and Boyd, Adriane and Brokait{\.e}, Kristina and Burchardt, Aljoscha and Candito, Marie and Caron, Bernard and Caron, Gauthier and Cavalcanti, Tatiana and Cebiro{\u g}lu Eryi{\u g}it, G{\"u}l{\c s}en and Cecchini, Flavio Massimiliano and Celano, Giuseppe G. A. and {\v C}{\'e}pl{\"o}, Slavom{\'{\i}}r and Cetin, Savas and Chalub, Fabricio and Choi, Jinho and Cho, Yongseok and Chun, Jayeol and Cignarella, Alessandra T. and Cinkov{\'a}, Silvie and Collomb, Aur{\'e}lie and {\c C}{\"o}ltekin, {\c C}a{\u g}r{\i} and Connor, Miriam and Courtin, Marine and Davidson, Elizabeth and de Marneffe, Marie-Catherine and de Paiva, Valeria and de Souza, Elvis and Diaz de Ilarraza, Arantza and Dickerson, Carly and Dione, Bamba and Dirix, Peter and Dobrovoljc, Kaja and Dozat, Timothy and Droganova, Kira and Dwivedi, Puneet and Eckhoff, Hanne and Eli, Marhaba and Elkahky, Ali and Ephrem, Binyam and Erina, Olga and Erjavec, Toma{\v z} and Etienne, Aline and Evelyn, Wograine and Farkas, Rich{\'a}rd and Fernandez Alcalde, Hector and Foster, Jennifer and Freitas, Cl{\'a}udia and Fujita, Kazunori and Gajdo{\v s}ov{\'a}, Katar{\'{\i}}na and Galbraith, Daniel and Garcia, Marcos and G{\"a}rdenfors, Moa and Garza, Sebastian and Gerdes, Kim and Ginter, Filip and Goenaga, Iakes and Gojenola, Koldo and G{\"o}k{\i}rmak, Memduh and Goldberg, Yoav and G{\'o}mez Guinovart, Xavier and Gonz{\'a}lez Saavedra, Berta and Grici{\=u}t{\.e}, Bernadeta and Grioni, Matias and Gr{\=u}z{\={\i}}tis, Normunds and Guillaume, Bruno and Guillot-Barbance, C{\'e}line and Habash, Nizar and Haji{\v c}, Jan and Haji{\v c} jr., Jan and H{\"a}m{\"a}l{\"a}inen, Mika and H{\`a} M{\~y}, Linh and Han, Na-Rae and Harris, Kim and Haug, Dag and Heinecke, Johannes and Hennig, Felix and Hladk{\'a}, Barbora and Hlav{\'a}{\v c}ov{\'a}, Jaroslava and Hociung, Florinel and Hohle, Petter and Hwang, Jena and Ikeda, Takumi and Ion, Radu and Irimia, Elena and Ishola, {\d O}l{\'a}j{\'{\i}}d{\'e} and Jel{\'{\i}}nek, Tom{\'a}{\v s} and Johannsen, Anders and J{\o}rgensen, Fredrik and Juutinen, Markus and Ka{\c s}{\i}kara, H{\"u}ner and Kaasen, Andre and Kabaeva, Nadezhda and Kahane, Sylvain and Kanayama, Hiroshi and Kanerva, Jenna and Katz, Boris and Kayadelen, Tolga and Kenney, Jessica and Kettnerov{\'a}, V{\'a}clava and Kirchner, Jesse and Klementieva, Elena and K{\"o}hn, Arne and Kopacewicz, Kamil and Kotsyba, Natalia and Kovalevskait{\.e}, Jolanta and Krek, Simon and Kwak, Sookyoung and Laippala, Veronika and Lambertino, Lorenzo and Lam, Lucia and Lando, Tatiana and Larasati, Septina Dian and Lavrentiev, Alexei and Lee, John and L{\^e} H{\`{\^o}}ng, Phương and Lenci, Alessandro and Lertpradit, Saran and Leung, Herman and Li, Cheuk Ying and Li, Josie and Li, Keying and Lim, {KyungTae} and Liovina, Maria and Li, Yuan and Ljube{\v s}i{\'c}, Nikola and Loginova, Olga and Lyashevskaya, Olga and Lynn, Teresa and Macketanz, Vivien and Makazhanov, Aibek and Mandl, Michael and Manning, Christopher and Manurung, Ruli and M{\u a}r{\u a}nduc, C{\u a}t{\u a}lina and Mare{\v c}ek, David and Marheinecke, Katrin and Mart{\'{\i}}nez Alonso, H{\'e}ctor and Martins, Andr{\'e} and Ma{\v s}ek, Jan and Matsumoto, Yuji and {McDonald}, Ryan and {McGuinness}, Sarah and Mendon{\c c}a, Gustavo and Miekka, Niko and Misirpashayeva, Margarita and Missil{\"a}, Anna and Mititelu, C{\u a}t{\u a}lin and Mitrofan, Maria and Miyao, Yusuke and Montemagni, Simonetta and More, Amir and Moreno Romero, Laura and Mori, Keiko Sophie and Morioka, Tomohiko and Mori, Shinsuke and Moro, Shigeki and Mortensen, Bjartur and Moskalevskyi, Bohdan and Muischnek, Kadri and Munro, Robert and Murawaki, Yugo and M{\"u}{\"u}risep, Kaili and Nainwani, Pinkey and Navarro Hor{\~n}iacek, Juan Ignacio and Nedoluzhko, Anna and Ne{\v s}pore-B{\=e}rzkalne, Gunta and Nguy{\~{\^e}}n Th{\d i}, Lương and Nguy{\~{\^e}}n Th{\d i} Minh, Huy{\`{\^e}}n and Nikaido, Yoshihiro and Nikolaev, Vitaly and Nitisaroj, Rattima and Nurmi, Hanna and Ojala, Stina and Ojha, Atul Kr. and Ol{\'u}{\`o}kun, Ad{\'e}day{\d o}̀ and Omura, Mai and Osenova, Petya and {\"O}stling, Robert and {\O}vrelid, Lilja and Partanen, Niko and Pascual, Elena and Passarotti, Marco and Patejuk, Agnieszka and Paulino-Passos, Guilherme and Peljak-{\L}api{\'n}ska, Angelika and Peng, Siyao and Perez, Cenel-Augusto and Perrier, Guy and Petrova, Daria and Petrov, Slav and Phelan, Jason and Piitulainen, Jussi and Pirinen, Tommi A and Pitler, Emily and Plank, Barbara and Poibeau, Thierry and Ponomareva, Larisa and Popel, Martin and Pretkalni{\c n}a, Lauma and Pr{\'e}vost, Sophie and Prokopidis, Prokopis and Przepi{\'o}rkowski, Adam and Puolakainen, Tiina and Pyysalo, Sampo and Qi, Peng and R{\"a}{\"a}bis, Andriela and Rademaker, Alexandre and Ramasamy, Loganathan and Rama, Taraka and Ramisch, Carlos and Ravishankar, Vinit and Real, Livy and Reddy, Siva and Rehm, Georg and Riabov, Ivan and Rie{\ss}ler, Michael and Rimkut{\.e}, Erika and Rinaldi, Larissa and Rituma, Laura and Rocha, Luisa and Romanenko, Mykhailo and Rosa, Rudolf and Rovati, Davide and Roșca, Valentin and Rudina, Olga and Rueter, Jack and Sadde, Shoval and Sagot, Beno{\^{\i}}t and Saleh, Shadi and Salomoni, Alessio and Samard{\v z}i{\'c}, Tanja and Samson, Stephanie and Sanguinetti, Manuela and S{\"a}rg, Dage and Saul{\={\i}}te, Baiba and Sawanakunanon, Yanin and Schneider, Nathan and Schuster, Sebastian and Seddah, Djam{\'e} and Seeker, Wolfgang and Seraji, Mojgan and Shen, Mo and Shimada, Atsuko and Shirasu, Hiroyuki and Shohibussirri, Muh and Sichinava, Dmitry and Silveira, Aline and Silveira, Natalia and Simi, Maria and Simionescu, Radu and Simk{\'o}, Katalin and {\v S}imkov{\'a}, M{\'a}ria and Simov, Kiril and Smith, Aaron and Soares-Bastos, Isabela and Spadine, Carolyn and Stella, Antonio and Straka, Milan and Strnadov{\'a}, Jana and Suhr, Alane and Sulubacak, Umut and Suzuki, Shingo and Sz{\'a}nt{\'o}, Zsolt and Taji, Dima and Takahashi, Yuta and Tamburini, Fabio and Tanaka, Takaaki and Tellier, Isabelle and Thomas, Guillaume and Torga, Liisi and Trosterud, Trond and Trukhina, Anna and Tsarfaty, Reut and Tyers, Francis and Uematsu, Sumire and Ure{\v s}ov{\'a}, Zde{\v n}ka and Uria, Larraitz and Uszkoreit, Hans and Utka, Andrius and Vajjala, Sowmya and van Niekerk, Daniel and van Noord, Gertjan and Varga, Viktor and Villemonte de la Clergerie, Eric and Vincze, Veronika and Wallin, Lars and Walsh, Abigail and Wang, Jing Xian and Washington, Jonathan North and Wendt, Maximilan and Williams, Seyi and Wir{\'e}n, Mats and Wittern, Christian and Woldemariam, Tsegay and Wong, Tak-sum and Wr{\'o}blewska, Alina and Yako, Mary and Yamazaki, Naoki and Yan, Chunxiao and Yasuoka, Koichi and Yavrumyan, Marat M. and Yu, Zhuoran and {\v Z}abokrtsk{\'y}, Zden{\v e}k and Zeldes, Amir and Zhang, Manying and Zhu, Hanzhi},
      copyright = {Licence Universal Dependencies v2.5},
      note = {{LINDAT}/{CLARIAH}-{CZ} digital library at the Institute of Formal and Applied Linguistics ({{\'U}FAL}), Faculty of Mathematics and Physics, Charles University},
      title = {Universal Dependencies 2.5},
      url = {http://hdl.handle.net/11234/1-3105},
      year = {2019},
    }

    @inproceedings{Conneau2018XNLIEC,
      author = {Alexis Conneau and Guillaume Lample and Ruty Rinott and Adina Williams and Samuel R. Bowman and Holger Schwenk and Veselin Stoyanov},
      booktitle = {EMNLP},
      title = {XNLI: Evaluating Cross-lingual Sentence Representations},
      year = {2018},
    }

    @article{Lewis2019MLQAEC,
      author = {Patrick Lewis and Barlas Oguz and Ruty Rinott and Sebastian Riedel and Holger Schwenk},
      journal = {ArXiv},
      title = {MLQA: Evaluating Cross-lingual Extractive Question Answering},
      volume = {abs/1910.07475},
      year = {2019},
    }

    @article{Liang2020XGLUEAN,
      author = {Yaobo Liang and Nan Duan and Yeyun Gong and Ning Wu and Fenfei Guo and Weizhen Qi and Ming Gong and Linjun Shou and Daxin Jiang and Guihong Cao and Xiaodong Fan and Ruofei Zhang and Rahul Agrawal and Edward Cui and Sining Wei and Taroon Bharti and Ying Qiao and Jiun-Hung Chen and Winnie Wu and Shuguang Liu and Fan Yang and Daniel Campos and Rangan Majumder and Ming Zhou},
      journal = {arXiv},
      title = {XGLUE: A New Benchmark Dataset for Cross-lingual Pre-training, Understanding and Generation},
      volume = {abs/2004.01401},
      year = {2020},
    }

    @article{Sang2002IntroductionTT,
      author = {Erik F. Tjong Kim Sang},
      journal = {ArXiv},
      title = {Introduction to the CoNLL-2002 Shared Task: Language-Independent Named Entity Recognition},
      volume = {cs.CL/0209010},
      year = {2002},
    }

    @article{Sang2003IntroductionTT,
      author = {Erik F. Tjong Kim Sang and Fien De Meulder},
      journal = {ArXiv},
      title = {Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition},
      volume = {cs.CL/0306050},
      year = {2003},
    }

    @article{Yang2019PAWSXAC,
      author = {Yinfei Yang and Yuan Zhang and Chris Tar and Jason Baldridge},
      journal = {ArXiv},
      title = {PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification},
      volume = {abs/1908.11828},
      year = {2019},
    }

    ```
