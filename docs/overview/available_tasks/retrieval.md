
# Retrieval

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 356

#### AILACasedocs

The task is to retrieve the case document that most closely matches or is most relevant to the scenario described in the provided query.

**Dataset:** [`mteb/AILA_casedocs`](https://huggingface.co/datasets/mteb/AILA_casedocs) • **License:** cc-by-4.0 • [Learn more →](https://zenodo.org/records/4063986)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Legal, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @dataset{paheli_bhattacharya_2020_4063986,
      author = {Paheli Bhattacharya and
    Kripabandhu Ghosh and
    Saptarshi Ghosh and
    Arindam Pal and
    Parth Mehta and
    Arnab Bhattacharya and
    Prasenjit Majumder},
      doi = {10.5281/zenodo.4063986},
      month = oct,
      publisher = {Zenodo},
      title = {AILA 2019 Precedent \& Statute Retrieval Task},
      url = {https://doi.org/10.5281/zenodo.4063986},
      year = {2020},
    }
    
    ```
    



#### AILAStatutes

This dataset is structured for the task of identifying the most relevant statutes for a given situation.

**Dataset:** [`mteb/AILA_statutes`](https://huggingface.co/datasets/mteb/AILA_statutes) • **License:** cc-by-4.0 • [Learn more →](https://zenodo.org/records/4063986)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Legal, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @dataset{paheli_bhattacharya_2020_4063986,
      author = {Paheli Bhattacharya and
    Kripabandhu Ghosh and
    Saptarshi Ghosh and
    Arindam Pal and
    Parth Mehta and
    Arnab Bhattacharya and
    Prasenjit Majumder},
      doi = {10.5281/zenodo.4063986},
      month = oct,
      publisher = {Zenodo},
      title = {AILA 2019 Precedent \& Statute Retrieval Task},
      url = {https://doi.org/10.5281/zenodo.4063986},
      year = {2020},
    }
    
    ```
    



#### ARCChallenge

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on ARC-Challenge.

**Dataset:** [`mteb/ARCChallenge`](https://huggingface.co/datasets/mteb/ARCChallenge) • **License:** cc-by-sa-4.0 • [Learn more →](https://allenai.org/data/arc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{clark2018think,
      author = {Clark, Peter and Cowhey, Isaac and Etzioni, Oren and Khot, Tushar and Sabharwal, Ashish and Schoenick, Carissa and Tafjord, Oyvind},
      journal = {arXiv preprint arXiv:1803.05457},
      title = {Think you have solved question answering? try arc, the ai2 reasoning challenge},
      year = {2018},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### AlloprofRetrieval

This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum curated by a large number of teachers to students on all subjects taught from in primary and secondary school

**Dataset:** [`mteb/AlloprofRetrieval`](https://huggingface.co/datasets/mteb/AlloprofRetrieval) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/antoinelb7/alloprof)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fra | Encyclopaedic, Written | human-annotated | found |



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
    



#### AlphaNLI

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on AlphaNLI.

**Dataset:** [`mteb/AlphaNLI`](https://huggingface.co/datasets/mteb/AlphaNLI) • **License:** cc-by-nc-4.0 • [Learn more →](https://leaderboard.allenai.org/anli/submissions/get-started)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{bhagavatula2019abductive,
      author = {Bhagavatula, Chandra and Bras, Ronan Le and Malaviya, Chaitanya and Sakaguchi, Keisuke and Holtzman, Ari and Rashkin, Hannah and Downey, Doug and Yih, Scott Wen-tau and Choi, Yejin},
      journal = {arXiv preprint arXiv:1908.05739},
      title = {Abductive commonsense reasoning},
      year = {2019},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### AppsRetrieval

The dataset is a collection of natural language queries and their corresponding code snippets. The task is to retrieve the most relevant code snippet for a given query.

**Dataset:** [`CoIR-Retrieval/apps`](https://huggingface.co/datasets/CoIR-Retrieval/apps) • **License:** mit • [Learn more →](https://arxiv.org/abs/2105.09938)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng, python | Programming, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{hendrycksapps2021,
      author = {Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
      journal = {NeurIPS},
      title = {Measuring Coding Challenge Competence With APPS},
      year = {2021},
    }
    
    ```
    



#### ArguAna

ArguAna: Retrieval of the Best Counterargument without Prior Topic Knowledge

**Dataset:** [`mteb/arguana`](https://huggingface.co/datasets/mteb/arguana) • **License:** cc-by-sa-4.0 • [Learn more →](http://argumentation.bplaced.net/arguana/data)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Social, Web, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{wachsmuth2018retrieval,
      author = {Wachsmuth, Henning and Syed, Shahbaz and Stein, Benno},
      booktitle = {ACL},
      title = {Retrieval of the Best Counterargument without Prior Topic Knowledge},
      year = {2018},
    }
    
    ```
    



#### ArguAna-Fa

ArguAna-Fa

**Dataset:** [`MCINext/arguana-fa`](https://huggingface.co/datasets/MCINext/arguana-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/arguana-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Blog | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### ArguAna-Fa.v2

ArguAna-Fa.v2

**Dataset:** [`MCINext/arguana-fa-v2`](https://huggingface.co/datasets/MCINext/arguana-fa-v2) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/arguana-fa-v2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Blog | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### ArguAna-NL

ArguAna involves the task of retrieval of the best counterargument to an argument. ArguAna-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-arguana`](https://huggingface.co/datasets/clips/beir-nl-arguana) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-arguana)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### ArguAna-PL

ArguAna-PL

**Dataset:** [`mteb/ArguAna-PL`](https://huggingface.co/datasets/mteb/ArguAna-PL) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clarin-knext/arguana-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Medical, Written | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### ArguAna-VN

A translated dataset from NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/arguana-vn`](https://huggingface.co/datasets/GreenNode/arguana-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://argumentation.bplaced.net/arguana/data)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Medical, Written | derived | machine-translated and LM verified |



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
    



#### AutoRAGRetrieval

This dataset enables the evaluation of Korean RAG performance across various domains—finance, public sector, healthcare, legal, and commerce—by providing publicly accessible documents, questions, and answers.

**Dataset:** [`yjoonjang/markers_bm`](https://huggingface.co/datasets/yjoonjang/markers_bm) • **License:** mit • [Learn more →](https://arxiv.org/abs/2410.20878)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | kor | Financial, Government, Legal, Medical, Social | human-annotated | created |



??? quote "Citation"

    
    ```bibtex
    
    @misc{kim2024autoragautomatedframeworkoptimization,
      archiveprefix = {arXiv},
      author = {Dongkyu Kim and Byoungwook Kim and Donggeon Han and Matouš Eibich},
      eprint = {2410.20878},
      primaryclass = {cs.CL},
      title = {AutoRAG: Automated Framework for optimization of Retrieval Augmented Generation Pipeline},
      url = {https://arxiv.org/abs/2410.20878},
      year = {2024},
    }
    
    ```
    



#### BIRCO-ArguAna

Retrieval task using the ArguAna dataset from BIRCO. This dataset contains 100 queries where both queries and passages are complex one-paragraph arguments about current affairs. The objective is to retrieve the counter-argument that directly refutes the query’s stance.

**Dataset:** [`mteb/BIRCO-ArguAna-Test`](https://huggingface.co/datasets/mteb/BIRCO-ArguAna-Test) • **License:** cc-by-4.0 • [Learn more →](https://github.com/BIRCO-benchmark/BIRCO)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Written | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wang2024bircobenchmarkinformationretrieval,
      archiveprefix = {arXiv},
      author = {Xiaoyue Wang and Jianyou Wang and Weili Cao and Kaicheng Wang and Ramamohan Paturi and Leon Bergen},
      eprint = {2402.14151},
      primaryclass = {cs.IR},
      title = {BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
      url = {https://arxiv.org/abs/2402.14151},
      year = {2024},
    }
    
    ```
    



#### BIRCO-ClinicalTrial

Retrieval task using the Clinical-Trial dataset from BIRCO. This dataset contains 50 queries that are patient case reports. Each query has a candidate pool comprising 30-110 clinical trial descriptions. Relevance is graded (0, 1, 2), where 1 and 2 are considered relevant.

**Dataset:** [`mteb/BIRCO-ClinicalTrial-Test`](https://huggingface.co/datasets/mteb/BIRCO-ClinicalTrial-Test) • **License:** cc-by-4.0 • [Learn more →](https://github.com/BIRCO-benchmark/BIRCO)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wang2024bircobenchmarkinformationretrieval,
      archiveprefix = {arXiv},
      author = {Xiaoyue Wang and Jianyou Wang and Weili Cao and Kaicheng Wang and Ramamohan Paturi and Leon Bergen},
      eprint = {2402.14151},
      primaryclass = {cs.IR},
      title = {BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
      url = {https://arxiv.org/abs/2402.14151},
      year = {2024},
    }
    
    ```
    



#### BIRCO-DorisMae

Retrieval task using the DORIS-MAE dataset from BIRCO. This dataset contains 60 queries that are complex research questions from computer scientists. Each query has a candidate pool of approximately 110 abstracts. Relevance is graded from 0 to 2 (scores of 1 and 2 are considered relevant).

**Dataset:** [`mteb/BIRCO-DorisMae-Test`](https://huggingface.co/datasets/mteb/BIRCO-DorisMae-Test) • **License:** cc-by-4.0 • [Learn more →](https://github.com/BIRCO-benchmark/BIRCO)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wang2024bircobenchmarkinformationretrieval,
      archiveprefix = {arXiv},
      author = {Xiaoyue Wang and Jianyou Wang and Weili Cao and Kaicheng Wang and Ramamohan Paturi and Leon Bergen},
      eprint = {2402.14151},
      primaryclass = {cs.IR},
      title = {BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
      url = {https://arxiv.org/abs/2402.14151},
      year = {2024},
    }
    
    ```
    



#### BIRCO-Relic

Retrieval task using the RELIC dataset from BIRCO. This dataset contains 100 queries which are excerpts from literary analyses with a missing quotation (indicated by [masked sentence(s)]). Each query has a candidate pool of 50 passages. The objective is to retrieve the passage that best completes the literary analysis.

**Dataset:** [`mteb/BIRCO-Relic-Test`](https://huggingface.co/datasets/mteb/BIRCO-Relic-Test) • **License:** cc-by-4.0 • [Learn more →](https://github.com/BIRCO-benchmark/BIRCO)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Fiction | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wang2024bircobenchmarkinformationretrieval,
      archiveprefix = {arXiv},
      author = {Xiaoyue Wang and Jianyou Wang and Weili Cao and Kaicheng Wang and Ramamohan Paturi and Leon Bergen},
      eprint = {2402.14151},
      primaryclass = {cs.IR},
      title = {BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
      url = {https://arxiv.org/abs/2402.14151},
      year = {2024},
    }
    
    ```
    



#### BIRCO-WTB

Retrieval task using the WhatsThatBook dataset from BIRCO. This dataset contains 100 queries where each query is an ambiguous description of a book. Each query has a candidate pool of 50 book descriptions. The objective is to retrieve the correct book description.

**Dataset:** [`mteb/BIRCO-WTB-Test`](https://huggingface.co/datasets/mteb/BIRCO-WTB-Test) • **License:** cc-by-4.0 • [Learn more →](https://github.com/BIRCO-benchmark/BIRCO)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Fiction | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wang2024bircobenchmarkinformationretrieval,
      archiveprefix = {arXiv},
      author = {Xiaoyue Wang and Jianyou Wang and Weili Cao and Kaicheng Wang and Ramamohan Paturi and Leon Bergen},
      eprint = {2402.14151},
      primaryclass = {cs.IR},
      title = {BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
      url = {https://arxiv.org/abs/2402.14151},
      year = {2024},
    }
    
    ```
    



#### BSARDRetrieval

The Belgian Statutory Article Retrieval Dataset (BSARD) is a French native dataset for studying legal information retrieval. BSARD consists of more than 22,600 statutory articles from Belgian law and about 1,100 legal questions posed by Belgian citizens and labeled by experienced jurists with relevant articles from the corpus.

**Dataset:** [`mteb/BSARDRetrieval`](https://huggingface.co/datasets/mteb/BSARDRetrieval) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/maastrichtlawtech/bsard)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_100 | fra | Legal, Spoken | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{louis2022statutory,
      address = {Dublin, Ireland},
      author = {Louis, Antoine and Spanakis, Gerasimos},
      booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
      doi = {10.18653/v1/2022.acl-long.468},
      month = may,
      pages = {6789–6803},
      publisher = {Association for Computational Linguistics},
      title = {A Statutory Article Retrieval Dataset in French},
      url = {https://aclanthology.org/2022.acl-long.468/},
      year = {2022},
    }
    
    ```
    



#### BSARDRetrieval.v2

BSARD is a French native dataset for legal information retrieval. BSARDRetrieval.v2 covers multi-article queries, fixing issues (#2906) with the previous data loading. 

**Dataset:** [`mteb/BSARDRetrieval.v2`](https://huggingface.co/datasets/mteb/BSARDRetrieval.v2) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/maastrichtlawtech/bsard)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_100 | fra | Legal, Spoken | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{louis2022statutory,
      address = {Dublin, Ireland},
      author = {Louis, Antoine and Spanakis, Gerasimos},
      booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
      doi = {10.18653/v1/2022.acl-long.468},
      month = may,
      pages = {6789–6803},
      publisher = {Association for Computational Linguistics},
      title = {A Statutory Article Retrieval Dataset in French},
      url = {https://aclanthology.org/2022.acl-long.468/},
      year = {2022},
    }
    
    ```
    



#### BarExamQA

A benchmark for retrieving legal provisions that answer US bar exam questions.

**Dataset:** [`isaacus/mteb-barexam-qa`](https://huggingface.co/datasets/isaacus/mteb-barexam-qa) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/reglab/barexam_qa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Legal | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Zheng_2025,
      author = {Zheng, Lucia and Guha, Neel and Arifov, Javokhir and Zhang, Sarah and Skreta, Michal and Manning, Christopher D. and Henderson, Peter and Ho, Daniel E.},
      booktitle = {Proceedings of the Symposium on Computer Science and Law on ZZZ},
      collection = {CSLAW ’25},
      doi = {10.1145/3709025.3712219},
      eprint = {2505.03970},
      month = mar,
      pages = {169–193},
      publisher = {ACM},
      series = {CSLAW ’25},
      title = {A Reasoning-Focused Legal Retrieval Benchmark},
      url = {http://dx.doi.org/10.1145/3709025.3712219},
      year = {2025},
    }
    
    ```
    



#### BelebeleRetrieval

Belebele is a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants (including 115 distinct languages and their scripts)

**Dataset:** [`facebook/belebele`](https://huggingface.co/datasets/facebook/belebele) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2308.16884)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | acm, afr, als, amh, apc, ... (115) | News, Web, Written | expert-annotated | created |



??? quote "Citation"

    
    ```bibtex
    
    @article{bandarkar2023belebele,
      author = {Lucas Bandarkar and Davis Liang and Benjamin Muller and Mikel Artetxe and Satya Narayan Shukla and Donald Husa and Naman Goyal and Abhinandan Krishnan and Luke Zettlemoyer and Madian Khabsa},
      journal = {arXiv preprint arXiv:2308.16884},
      title = {The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants},
      year = {2023},
    }
    
    ```
    



#### BillSumCA

A benchmark for retrieving Californian bills based on their summaries.

**Dataset:** [`isaacus/mteb-BillSumCA`](https://huggingface.co/datasets/isaacus/mteb-BillSumCA) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/FiscalNote/billsum)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Government, Legal | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Eidelman_2019,
      author = {Eidelman, Vladimir},
      booktitle = {Proceedings of the 2nd Workshop on New Frontiers in Summarization},
      doi = {10.18653/v1/d19-5406},
      pages = {48–56},
      publisher = {Association for Computational Linguistics},
      title = {BillSum: A Corpus for Automatic Summarization of US Legislation},
      url = {http://dx.doi.org/10.18653/v1/D19-5406},
      year = {2019},
    }
    
    ```
    



#### BillSumUS

A benchmark for retrieving US federal bills based on their summaries.

**Dataset:** [`isaacus/mteb-BillSumUS`](https://huggingface.co/datasets/isaacus/mteb-BillSumUS) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/FiscalNote/billsum)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Government, Legal | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Eidelman_2019,
      author = {Eidelman, Vladimir},
      booktitle = {Proceedings of the 2nd Workshop on New Frontiers in Summarization},
      doi = {10.18653/v1/d19-5406},
      pages = {48–56},
      publisher = {Association for Computational Linguistics},
      title = {BillSum: A Corpus for Automatic Summarization of US Legislation},
      url = {http://dx.doi.org/10.18653/v1/D19-5406},
      year = {2019},
    }
    
    ```
    



#### BrightLongRetrieval

Bright retrieval dataset with long documents.

**Dataset:** [`xlangai/BRIGHT`](https://huggingface.co/datasets/xlangai/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{su2024brightrealisticchallengingbenchmark,
      archiveprefix = {arXiv},
      author = {Hongjin Su and Howard Yen and Mengzhou Xia and Weijia Shi and Niklas Muennighoff and Han-yu Wang and Haisu Liu and Quan Shi and Zachary S. Siegel and Michael Tang and Ruoxi Sun and Jinsung Yoon and Sercan O. Arik and Danqi Chen and Tao Yu},
      eprint = {2407.12883},
      primaryclass = {cs.CL},
      title = {BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval},
      url = {https://arxiv.org/abs/2407.12883},
      year = {2024},
    }
    
    ```
    



#### BrightRetrieval

Bright retrieval dataset.

**Dataset:** [`xlangai/BRIGHT`](https://huggingface.co/datasets/xlangai/BRIGHT) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/xlangai/BRIGHT)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{su2024brightrealisticchallengingbenchmark,
      archiveprefix = {arXiv},
      author = {Hongjin Su and Howard Yen and Mengzhou Xia and Weijia Shi and Niklas Muennighoff and Han-yu Wang and Haisu Liu and Quan Shi and Zachary S. Siegel and Michael Tang and Ruoxi Sun and Jinsung Yoon and Sercan O. Arik and Danqi Chen and Tao Yu},
      eprint = {2407.12883},
      primaryclass = {cs.CL},
      title = {BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval},
      url = {https://arxiv.org/abs/2407.12883},
      year = {2024},
    }
    
    ```
    



#### BuiltBenchRetrieval

Retrieval of built asset entity type/class descriptions given a query describing an entity as represented in well-established industry classification systems such as Uniclass, IFC, etc.

**Dataset:** [`mteb/BuiltBenchRetrieval`](https://huggingface.co/datasets/mteb/BuiltBenchRetrieval) • **License:** cc-by-nd-4.0 • [Learn more →](https://arxiv.org/abs/2411.12056)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Engineering, Written | derived | created |



??? quote "Citation"

    
    ```bibtex
    
    @article{shahinmoghadam2024benchmarking,
      author = {Shahinmoghadam, Mehrzad and Motamedi, Ali},
      journal = {arXiv preprint arXiv:2411.12056},
      title = {Benchmarking pre-trained text embedding models in aligning built asset information},
      year = {2024},
    }
    
    ```
    



#### COIRCodeSearchNetRetrieval

The dataset is a collection of code snippets and their corresponding natural language queries. The task is to retrieve the most relevant code summary given a code snippet.

**Dataset:** [`CoIR-Retrieval/CodeSearchNet`](https://huggingface.co/datasets/CoIR-Retrieval/CodeSearchNet) • **License:** mit • [Learn more →](https://huggingface.co/datasets/code_search_net/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | go, java, javascript, php, python, ... (6) | Programming, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{husain2019codesearchnet,
      author = {Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
      journal = {arXiv preprint arXiv:1909.09436},
      title = {{CodeSearchNet} challenge: Evaluating the state of semantic code search},
      year = {2019},
    }
    
    ```
    



#### CQADupstack-Android-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Android-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Android-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-android-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Non-fiction, Programming, Web, Written | derived | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### CQADupstack-English-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-English-PL`](https://huggingface.co/datasets/mteb/CQADupstack-English-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-english-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Written | derived | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### CQADupstack-Gaming-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Gaming-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Gaming-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-gaming-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web, Written | derived | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### CQADupstack-Gis-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Gis-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Gis-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-gis-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic, Non-fiction, Written | derived | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### CQADupstack-Mathematica-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Mathematica-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Mathematica-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-mathematica-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic, Non-fiction, Written | derived | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### CQADupstack-Physics-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Physics-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Physics-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-physics-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic, Non-fiction, Written | derived | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### CQADupstack-Programmers-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Programmers-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Programmers-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-programmers-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Non-fiction, Programming, Written | derived | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### CQADupstack-Stats-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Stats-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Stats-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-stats-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic, Non-fiction, Written | derived | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### CQADupstack-Tex-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Tex-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Tex-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-tex-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Non-fiction, Written | derived | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### CQADupstack-Unix-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Unix-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Unix-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-unix-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Programming, Web, Written | derived | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### CQADupstack-Webmasters-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Webmasters-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Webmasters-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-webmasters-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web, Written | derived | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### CQADupstack-Wordpress-PL

CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset

**Dataset:** [`mteb/CQADupstack-Wordpress-PL`](https://huggingface.co/datasets/mteb/CQADupstack-Wordpress-PL) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/clarin-knext/cqadupstack-wordpress-pl)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Programming, Web, Written | derived | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### CQADupstackAndroid-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### CQADupstackAndroid-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-android-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-android-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Non-fiction, Programming, Web, Written | derived | machine-translated and LM verified |



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
    



#### CQADupstackAndroidRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-android`](https://huggingface.co/datasets/mteb/cqadupstack-android) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Non-fiction, Programming, Web, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### CQADupstackAndroidRetrieval-Fa

CQADupstackAndroidRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-android-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-android-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-android-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### CQADupstackEnglish-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### CQADupstackEnglishRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-english`](https://huggingface.co/datasets/mteb/cqadupstack-english) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### CQADupstackEnglishRetrieval-Fa

CQADupstackEnglishRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-english-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-english-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-english-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### CQADupstackGaming-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### CQADupstackGamingRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-gaming`](https://huggingface.co/datasets/mteb/cqadupstack-gaming) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Web, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### CQADupstackGamingRetrieval-Fa

CQADupstackGamingRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-gaming-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-gaming-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-gaming-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### CQADupstackGis-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### CQADupstackGis-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-gis-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-gis-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Non-fiction, Written | derived | machine-translated and LM verified |



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
    



#### CQADupstackGisRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-gis`](https://huggingface.co/datasets/mteb/cqadupstack-gis) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### CQADupstackGisRetrieval-Fa

CQADupstackGisRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-gis-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-gis-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-gis-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### CQADupstackMathematica-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### CQADupstackMathematica-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-mathematica-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-mathematica-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified |



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
    



#### CQADupstackMathematicaRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-mathematica`](https://huggingface.co/datasets/mteb/cqadupstack-mathematica) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### CQADupstackMathematicaRetrieval-Fa

CQADupstackMathematicaRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-mathematica-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-mathematica-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-mathematica-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### CQADupstackPhysics-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### CQADupstackPhysics-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-physics-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-physics-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified |



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
    



#### CQADupstackPhysicsRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-physics`](https://huggingface.co/datasets/mteb/cqadupstack-physics) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### CQADupstackPhysicsRetrieval-Fa

CQADupstackPhysicsRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-physics-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-physics-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-physics-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### CQADupstackProgrammers-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### CQADupstackProgrammers-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-programmers-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-programmers-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Non-fiction, Programming, Written | derived | machine-translated and LM verified |



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
    



#### CQADupstackProgrammersRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-programmers`](https://huggingface.co/datasets/mteb/cqadupstack-programmers) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Non-fiction, Programming, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### CQADupstackProgrammersRetrieval-Fa

CQADupstackProgrammersRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-programmers-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-programmers-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-programmers-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### CQADupstackStats-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### CQADupstackStats-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-stats-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-stats-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified |



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
    



#### CQADupstackStatsRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-stats`](https://huggingface.co/datasets/mteb/cqadupstack-stats) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### CQADupstackStatsRetrieval-Fa

CQADupstackStatsRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-stats-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-stats-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-stats-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### CQADupstackTex-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### CQADupstackTex-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-tex-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-tex-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Non-fiction, Written | derived | machine-translated and LM verified |



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
    



#### CQADupstackTexRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-tex`](https://huggingface.co/datasets/mteb/cqadupstack-tex) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### CQADupstackTexRetrieval-Fa

CQADupstackTexRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-tex-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-tex-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-tex-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### CQADupstackUnix-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### CQADupstackUnix-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-unix-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-unix-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Programming, Web, Written | derived | machine-translated and LM verified |



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
    



#### CQADupstackUnixRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-unix`](https://huggingface.co/datasets/mteb/cqadupstack-unix) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Programming, Web, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### CQADupstackUnixRetrieval-Fa

CQADupstackUnixRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-unix-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-unix-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-unix-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### CQADupstackWebmasters-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### CQADupstackWebmasters-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-webmasters-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-webmasters-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Web, Written | derived | machine-translated and LM verified |



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
    



#### CQADupstackWebmastersRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-webmasters`](https://huggingface.co/datasets/mteb/cqadupstack-webmasters) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Web, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### CQADupstackWebmastersRetrieval-Fa

CQADupstackWebmastersRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-webmasters-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-webmasters-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-webmasters-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### CQADupstackWordpress-NL

CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a Dutch-translated version.

**Dataset:** [`clips/beir-nl-cqadupstack`](https://huggingface.co/datasets/clips/beir-nl-cqadupstack) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-cqadupstack)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### CQADupstackWordpress-VN

A translated dataset from CQADupStack: A Benchmark Data Set for Community Question-Answering Research
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/cqadupstack-wordpress-vn`](https://huggingface.co/datasets/GreenNode/cqadupstack-wordpress-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Programming, Web, Written | derived | machine-translated and LM verified |



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
    



#### CQADupstackWordpressRetrieval

CQADupStack: A Benchmark Data Set for Community Question-Answering Research

**Dataset:** [`mteb/cqadupstack-wordpress`](https://huggingface.co/datasets/mteb/cqadupstack-wordpress) • **License:** apache-2.0 • [Learn more →](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Programming, Web, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{hoogeveen2015,
      acmid = {2838934},
      address = {New York, NY, USA},
      articleno = {3},
      author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
      booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
      doi = {10.1145/2838931.2838934},
      isbn = {978-1-4503-4040-3},
      location = {Parramatta, NSW, Australia},
      numpages = {8},
      pages = {3:1--3:8},
      publisher = {ACM},
      series = {ADCS '15},
      title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
      url = {http://doi.acm.org/10.1145/2838931.2838934},
      year = {2015},
    }
    
    ```
    



#### CQADupstackWordpressRetrieval-Fa

CQADupstackWordpressRetrieval-Fa

**Dataset:** [`MCINext/cqadupstack-wordpress-fa`](https://huggingface.co/datasets/MCINext/cqadupstack-wordpress-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/cqadupstack-wordpress-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



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
| text to text (t2t) | ndcg_at_10 | eng | Medical | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{chatdoctor_healthcaremagic,
      title = {ChatDoctor HealthCareMagic: Medical Question-Answer Retrieval Dataset},
      url = {https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k},
      year = {2023},
    }
    
    ```
    



#### ChemHotpotQARetrieval

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/ChemHotpotQARetrieval`](https://huggingface.co/datasets/BASF-AI/ChemHotpotQARetrieval) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Chemistry | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }
    
    @inproceedings{yang-etal-2018-hotpotqa,
      abstract = {Existing question answering (QA) datasets fail to train QA systems to perform complex reasoning and provide explanations for answers. We introduce HotpotQA, a new dataset with 113k Wikipedia-based question-answer pairs with four key features: (1) the questions require finding and reasoning over multiple supporting documents to answer; (2) the questions are diverse and not constrained to any pre-existing knowledge bases or knowledge schemas; (3) we provide sentence-level supporting facts required for reasoning, allowing QA systems to reason with strong supervision and explain the predictions; (4) we offer a new type of factoid comparison questions to test QA systems{'} ability to extract relevant facts and perform necessary comparison. We show that HotpotQA is challenging for the latest QA systems, and the supporting facts enable models to improve performance and make explainable predictions.},
      address = {Brussels, Belgium},
      author = {Yang, Zhilin  and
    Qi, Peng  and
    Zhang, Saizheng  and
    Bengio, Yoshua  and
    Cohen, William  and
    Salakhutdinov, Ruslan  and
    Manning, Christopher D.},
      booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
      doi = {10.18653/v1/D18-1259},
      editor = {Riloff, Ellen  and
    Chiang, David  and
    Hockenmaier, Julia  and
    Tsujii, Jun{'}ichi},
      month = oct # {-} # nov,
      pages = {2369--2380},
      publisher = {Association for Computational Linguistics},
      title = {{H}otpot{QA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
      url = {https://aclanthology.org/D18-1259},
      year = {2018},
    }
    
    ```
    



#### ChemNQRetrieval

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/ChemNQRetrieval`](https://huggingface.co/datasets/BASF-AI/ChemNQRetrieval) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Chemistry | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{47761,
      author = {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh
    and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee
    and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le
    and Slav Petrov},
      journal = {Transactions of the Association of Computational Linguistics},
      title = {Natural Questions: a Benchmark for Question Answering Research},
      year = {2019},
    }
    
    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }
    
    ```
    



#### ClimateFEVER

CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims (queries) regarding climate-change. The underlying corpus is the same as FVER.

**Dataset:** [`mteb/climate-fever`](https://huggingface.co/datasets/mteb/climate-fever) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{diggelmann2021climatefever,
      archiveprefix = {arXiv},
      author = {Thomas Diggelmann and Jordan Boyd-Graber and Jannis Bulian and Massimiliano Ciaramita and Markus Leippold},
      eprint = {2012.00614},
      primaryclass = {cs.CL},
      title = {CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims},
      year = {2021},
    }
    
    ```
    



#### ClimateFEVER-Fa

ClimateFEVER-Fa

**Dataset:** [`MCINext/climate-fever-fa`](https://huggingface.co/datasets/MCINext/climate-fever-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/climate-fever-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### ClimateFEVER-NL

CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change. ClimateFEVER-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-climate-fever`](https://huggingface.co/datasets/clips/beir-nl-climate-fever) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-climate-fever)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Encyclopaedic, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### ClimateFEVER-VN

A translated dataset from CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/climate-fever-vn`](https://huggingface.co/datasets/GreenNode/climate-fever-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Encyclopaedic, Written | derived | machine-translated and LM verified |



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
    



#### ClimateFEVER.v2

CLIMATE-FEVER is a dataset following the FEVER methodology, containing 1,535 real-world climate change claims. This updated version addresses corpus mismatches and qrel inconsistencies in MTEB, restoring labels while refining corpus-query alignment for better accuracy. 

**Dataset:** [`mteb/climate-fever-v2`](https://huggingface.co/datasets/mteb/climate-fever-v2) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{diggelmann2021climatefever,
      archiveprefix = {arXiv},
      author = {Thomas Diggelmann and Jordan Boyd-Graber and Jannis Bulian and Massimiliano Ciaramita and Markus Leippold},
      eprint = {2012.00614},
      primaryclass = {cs.CL},
      title = {CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims},
      year = {2021},
    }
    
    ```
    



#### ClimateFEVERHardNegatives

CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/ClimateFEVER_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/ClimateFEVER_test_top_250_only_w_correct-v2) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{diggelmann2021climatefever,
      archiveprefix = {arXiv},
      author = {Thomas Diggelmann and Jordan Boyd-Graber and Jannis Bulian and Massimiliano Ciaramita and Markus Leippold},
      eprint = {2012.00614},
      primaryclass = {cs.CL},
      title = {CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims},
      year = {2021},
    }
    
    ```
    



#### CmedqaRetrieval

Online medical consultation text. Used the CMedQAv2 as its underlying dataset.

**Dataset:** [`mteb/CmedqaRetrieval`](https://huggingface.co/datasets/mteb/CmedqaRetrieval) • **License:** not specified • [Learn more →](https://aclanthology.org/2022.emnlp-main.357.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | Medical, Written | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{qiu2022dureaderretrievallargescalechinesebenchmark,
      archiveprefix = {arXiv},
      author = {Yifu Qiu and Hongyu Li and Yingqi Qu and Ying Chen and Qiaoqiao She and Jing Liu and Hua Wu and Haifeng Wang},
      eprint = {2203.10232},
      primaryclass = {cs.CL},
      title = {DuReader_retrieval: A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine},
      url = {https://arxiv.org/abs/2203.10232},
      year = {2022},
    }
    
    ```
    



#### CodeEditSearchRetrieval

The dataset is a collection of unified diffs of code changes, paired with a short instruction that describes the change. The dataset is derived from the CommitPackFT dataset.

**Dataset:** [`cassanof/CodeEditSearch`](https://huggingface.co/datasets/cassanof/CodeEditSearch) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/cassanof/CodeEditSearch/viewer)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | c, c++, go, java, javascript, ... (13) | Programming, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{muennighoff2023octopack,
      author = {Niklas Muennighoff and Qian Liu and Armel Zebaze and Qinkai Zheng and Binyuan Hui and Terry Yue Zhuo and Swayam Singh and Xiangru Tang and Leandro von Werra and Shayne Longpre},
      journal = {arXiv preprint arXiv:2308.07124},
      title = {OctoPack: Instruction Tuning Code Large Language Models},
      year = {2023},
    }
    
    ```
    



#### CodeFeedbackMT

The dataset is a collection of user queries and assistant responses. The task is to retrieve the most relevant response for a given query.

**Dataset:** [`CoIR-Retrieval/codefeedback-mt`](https://huggingface.co/datasets/CoIR-Retrieval/codefeedback-mt) • **License:** mit • [Learn more →](https://arxiv.org/abs/2402.14658)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Programming, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{zheng2024opencodeinterpreterintegratingcodegeneration,
      archiveprefix = {arXiv},
      author = {Tianyu Zheng and Ge Zhang and Tianhao Shen and Xueling Liu and Bill Yuchen Lin and Jie Fu and Wenhu Chen and Xiang Yue},
      eprint = {2402.14658},
      primaryclass = {cs.SE},
      title = {OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement},
      url = {https://arxiv.org/abs/2402.14658},
      year = {2024},
    }
    
    ```
    



#### CodeFeedbackST

The dataset is a collection of user queries and assistant responses. The task is to retrieve the most relevant response for a given query.

**Dataset:** [`CoIR-Retrieval/codefeedback-st`](https://huggingface.co/datasets/CoIR-Retrieval/codefeedback-st) • **License:** mit • [Learn more →](https://arxiv.org/abs/2407.02883)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Programming, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{li2024coircomprehensivebenchmarkcode,
      archiveprefix = {arXiv},
      author = {Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Yichun Yin and Hao Zhang and Yong Liu and Yasheng Wang and Ruiming Tang},
      eprint = {2407.02883},
      primaryclass = {cs.IR},
      title = {CoIR: A Comprehensive Benchmark for Code Information Retrieval Models},
      url = {https://arxiv.org/abs/2407.02883},
      year = {2024},
    }
    
    ```
    



#### CodeSearchNetCCRetrieval

The dataset is a collection of code snippets. The task is to retrieve the most relevant code snippet for a given code snippet.

**Dataset:** [`CoIR-Retrieval/CodeSearchNet-ccr`](https://huggingface.co/datasets/CoIR-Retrieval/CodeSearchNet-ccr) • **License:** mit • [Learn more →](https://arxiv.org/abs/2407.02883)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | go, java, javascript, php, python, ... (6) | Programming, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{li2024coircomprehensivebenchmarkcode,
      archiveprefix = {arXiv},
      author = {Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Yichun Yin and Hao Zhang and Yong Liu and Yasheng Wang and Ruiming Tang},
      eprint = {2407.02883},
      primaryclass = {cs.IR},
      title = {CoIR: A Comprehensive Benchmark for Code Information Retrieval Models},
      url = {https://arxiv.org/abs/2407.02883},
      year = {2024},
    }
    
    ```
    



#### CodeSearchNetRetrieval

The dataset is a collection of code snippets and their corresponding natural language queries. The task is to retrieve the most relevant code snippet for a given query.

**Dataset:** [`mteb/CodeSearchNetRetrieval`](https://huggingface.co/datasets/mteb/CodeSearchNetRetrieval) • **License:** mit • [Learn more →](https://huggingface.co/datasets/code_search_net/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | go, java, javascript, php, python, ... (6) | Programming, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{husain2019codesearchnet,
      author = {Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
      journal = {arXiv preprint arXiv:1909.09436},
      title = {{CodeSearchNet} challenge: Evaluating the state of semantic code search},
      year = {2019},
    }
    
    ```
    



#### CodeTransOceanContest

The dataset is a collection of code snippets and their corresponding natural language queries. The task is to retrieve the most relevant code snippet

**Dataset:** [`CoIR-Retrieval/codetrans-contest`](https://huggingface.co/datasets/CoIR-Retrieval/codetrans-contest) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2310.04951)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | c++, python | Programming, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{yan2023codetransoceancomprehensivemultilingualbenchmark,
      archiveprefix = {arXiv},
      author = {Weixiang Yan and Yuchen Tian and Yunzhe Li and Qian Chen and Wen Wang},
      eprint = {2310.04951},
      primaryclass = {cs.AI},
      title = {CodeTransOcean: A Comprehensive Multilingual Benchmark for Code Translation},
      url = {https://arxiv.org/abs/2310.04951},
      year = {2023},
    }
    
    ```
    



#### CodeTransOceanDL

The dataset is a collection of equivalent Python Deep Learning code snippets written in different machine learning framework. The task is to retrieve the equivalent code snippet in another framework, given a query code snippet from one framework.

**Dataset:** [`CoIR-Retrieval/codetrans-dl`](https://huggingface.co/datasets/CoIR-Retrieval/codetrans-dl) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2310.04951)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | python | Programming, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{yan2023codetransoceancomprehensivemultilingualbenchmark,
      archiveprefix = {arXiv},
      author = {Weixiang Yan and Yuchen Tian and Yunzhe Li and Qian Chen and Wen Wang},
      eprint = {2310.04951},
      primaryclass = {cs.AI},
      title = {CodeTransOcean: A Comprehensive Multilingual Benchmark for Code Translation},
      url = {https://arxiv.org/abs/2310.04951},
      year = {2023},
    }
    
    ```
    



#### CosQA

The dataset is a collection of natural language queries and their corresponding code snippets. The task is to retrieve the most relevant code snippet for a given query.

**Dataset:** [`CoIR-Retrieval/cosqa`](https://huggingface.co/datasets/CoIR-Retrieval/cosqa) • **License:** mit • [Learn more →](https://arxiv.org/abs/2105.13239)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng, python | Programming, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{huang2021cosqa20000webqueries,
      archiveprefix = {arXiv},
      author = {Junjie Huang and Duyu Tang and Linjun Shou and Ming Gong and Ke Xu and Daxin Jiang and Ming Zhou and Nan Duan},
      eprint = {2105.13239},
      primaryclass = {cs.CL},
      title = {CoSQA: 20,000+ Web Queries for Code Search and Question Answering},
      url = {https://arxiv.org/abs/2105.13239},
      year = {2021},
    }
    
    ```
    



#### CovidRetrieval

COVID-19 news articles

**Dataset:** [`mteb/CovidRetrieval`](https://huggingface.co/datasets/mteb/CovidRetrieval) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2203.03367)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | Entertainment, Medical | human-annotated | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{long2022multicprmultidomainchinese,
      archiveprefix = {arXiv},
      author = {Dingkun Long and Qiong Gao and Kuan Zou and Guangwei Xu and Pengjun Xie and Ruijie Guo and Jian Xu and Guanjun Jiang and Luxi Xing and Ping Yang},
      eprint = {2203.03367},
      primaryclass = {cs.IR},
      title = {Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval},
      url = {https://arxiv.org/abs/2203.03367},
      year = {2022},
    }
    
    ```
    



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



#### DAPFAMAllTitlAbsClmToFullTextRetrieval

In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, and target patent families are represented by Title, Abstract, Claims, and Description. Relevant target families have a citation link (cited or citing) with the query family. Additionally, no International Patent Classification-based filtering is applied. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to evaluate retrieval performance using Claims-augmented query patent family representations full-text target patent family representations across all technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_100 | eng | Chemistry, Engineering, Legal | derived | created |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### DAPFAMAllTitlAbsClmToTitlAbsClmRetrieval

In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, and target patent families are represented by Title, Abstract, and Claims. Relevant target families have a citation link (cited or citing) with the query family. Additionally, no International Patent Classification-based filtering is applied. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to evaluate retrieval when both query and target patent families use Claims-augmented representations across all technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_100 | eng | Chemistry, Engineering, Legal | derived | created |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### DAPFAMAllTitlAbsClmToTitlAbsRetrieval

In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, and target patent families are represented by Title and Abstract. Relevant target families have a citation link (cited or citing) with the query family. Additionally, no International Patent Classification-based filtering is applied. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to measure the effect of Claims-augmented query patent family representations when targets are limited to Title and Abstract across all technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_100 | eng | Chemistry, Engineering, Legal | derived | created |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### DAPFAMAllTitlAbsToFullTextRetrieval

In this patent family retrieval task, query patent families are represented by Title and Abstract, and target patent families are represented by Title, Abstract, Claims, and Description. Relevant target families have a citation link (cited or citing) with the query family. Additionally, no International Patent Classification-based filtering is applied. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to evaluate retrieval performance using Title and Abstract query patent family representations and full-text target patent family representations across all technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_100 | eng | Chemistry, Engineering, Legal | derived | created |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### DAPFAMAllTitlAbsToTitlAbsClmRetrieval

In this patent family retrieval task, query patent families are represented by Title and Abstract, and target patent families are represented by Title, Abstract, and Claims. Relevant target families have a citation link (cited or citing) with the query family. Additionally, no International Patent Classification-based filtering is applied. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to assess how adding Claims text to target patent family representations improves retrieval of citation-linked patent families across all technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_100 | eng | Chemistry, Engineering, Legal | derived | created |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### DAPFAMAllTitlAbsToTitlAbsRetrieval

In this patent family retrieval task, query patent families are represented by Title and Abstract, and target patent families are represented by Title and Abstract. Relevant target families have a citation link (cited or citing) with the query family. Additionally, no International Patent Classification-based filtering is applied. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to retrieve citation-linked patent families using query and target patent family representations of Title and Abstract across all technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_100 | eng | Chemistry, Engineering, Legal | derived | created |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### DAPFAMInTitlAbsClmToFullTextRetrieval

In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, and target patent families are represented by Title, Abstract, Claims, and Description. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to evaluate retrieval performance using Claims-augmented query patent family representations full-text target patent family representations within the same technical domain.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_100 | eng | Chemistry, Engineering, Legal | derived | created |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### DAPFAMInTitlAbsClmToTitlAbsClmRetrieval

In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, and target patent families are represented by Title, Abstract, and Claims. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to evaluate retrieval when both query and target patent families use Claims-augmented representations within the same technical domain.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_100 | eng | Chemistry, Engineering, Legal | derived | created |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### DAPFAMInTitlAbsClmToTitlAbsRetrieval

In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, and target patent families are represented by Title and Abstract. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to measure the effect of Claims-augmented query patent family representations when targets are limited to Title and Abstract within the same technical domain.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_100 | eng | Chemistry, Engineering, Legal | derived | created |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### DAPFAMInTitlAbsToFullTextRetrieval

In this patent family retrieval task, query patent families are represented by Title and Abstract, and target patent families are represented by Title, Abstract, Claims, and Description. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to evaluate retrieval performance using Title and Abstract query patent family representations and full-text target patent family representations within the same technical domain.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_100 | eng | Chemistry, Engineering, Legal | derived | created |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### DAPFAMInTitlAbsToTitlAbsClmRetrieval

In this patent family retrieval task, query patent families are represented by Title and Abstract, and target patent families are represented by Title, Abstract, and Claims. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to assess how adding Claims text to target patent family representations improves retrieval of citation-linked patent families within the same technical domain.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_100 | eng | Chemistry, Engineering, Legal | derived | created |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### DAPFAMInTitlAbsToTitlAbsRetrieval

In this patent family retrieval task, query patent families are represented by Title and Abstract, and target patent families are represented by Title and Abstract. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to retrieve citation-linked patent families using query and target patent family representations of Title and Abstract within the same technical domain.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_100 | eng | Chemistry, Engineering, Legal | derived | created |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### DAPFAMOutTitlAbsClmToFullTextRetrieval

In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, and target patent families are represented by Title, Abstract, Claims, and Description. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing no three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to evaluate retrieval performance using Claims-augmented query patent family representations full-text target patent family representations across different technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_100 | eng | Chemistry, Engineering, Legal | derived | created |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### DAPFAMOutTitlAbsClmToTitlAbsClmRetrieval

In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, and target patent families are represented by Title, Abstract, and Claims. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing no three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to evaluate retrieval when both query and target patent families use Claims-augmented representations across different technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_100 | eng | Chemistry, Engineering, Legal | derived | created |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### DAPFAMOutTitlAbsClmToTitlAbsRetrieval

In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, and target patent families are represented by Title and Abstract. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing no three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to measure the effect of Claims-augmented query patent family representations when targets are limited to Title and Abstract across different technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_100 | eng | Chemistry, Engineering, Legal | derived | created |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### DAPFAMOutTitlAbsToFullTextRetrieval

In this patent family retrieval task, query patent families are represented by Title and Abstract, and target patent families are represented by Title, Abstract, Claims, and Description. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing no three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to evaluate retrieval performance using Title and Abstract query patent family representations and full-text target patent family representations across different technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_100 | eng | Chemistry, Engineering, Legal | derived | created |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### DAPFAMOutTitlAbsToTitlAbsClmRetrieval

In this patent family retrieval task, query patent families are represented by Title and Abstract, and target patent families are represented by Title, Abstract, and Claims. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing no three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to assess how adding Claims text to target patent family representations improves retrieval of citation-linked patent families across different technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_100 | eng | Chemistry, Engineering, Legal | derived | created |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### DAPFAMOutTitlAbsToTitlAbsRetrieval

In this patent family retrieval task, query patent families are represented by Title and Abstract, and target patent families are represented by Title and Abstract. Relevant target families have a citation link (cited or citing) with the query family. Additionally, only targets sharing no three-character International Patent Classification code with the query family. Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141.Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. The goal of the task is to retrieve citation-linked patent families using query and target patent family representations of Title and Abstract across different technical domains.

**Dataset:** [`datalyes/DAPFAM_patent`](https://huggingface.co/datasets/datalyes/DAPFAM_patent) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2506.22141)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_100 | eng | Chemistry, Engineering, Legal | derived | created |



??? quote "Citation"

    
    ```bibtex
    @misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      archiveprefix = {arXiv},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      eprint = {2506.22141},
      primaryclass = {cs.CL},
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      url = {https://arxiv.org/abs/2506.22141},
      year = {2025},
    }
    ```
    



#### DBPedia

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base

**Dataset:** [`mteb/dbpedia`](https://huggingface.co/datasets/mteb/dbpedia) • **License:** mit • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Hasibi:2017:DVT,
      author = {Hasibi, Faegheh and Nikolaev, Fedor and Xiong, Chenyan and Balog, Krisztian and Bratsberg, Svein Erik and Kotov, Alexander and Callan, Jamie},
      booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
      doi = {10.1145/3077136.3080751},
      pages = {1265--1268},
      publisher = {ACM},
      series = {SIGIR '17},
      title = {DBpedia-Entity V2: A Test Collection for Entity Search},
      year = {2017},
    }
    
    ```
    



#### DBPedia-Fa

DBPedia-Fa

**Dataset:** [`MCINext/dbpedia-fa`](https://huggingface.co/datasets/MCINext/dbpedia-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/dbpedia-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Encyclopaedic | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### DBPedia-NL

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base. DBPedia-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-dbpedia-entity`](https://huggingface.co/datasets/clips/beir-nl-dbpedia-entity) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-dbpedia-entity)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Encyclopaedic, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### DBPedia-PL

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base

**Dataset:** [`mteb/DBPedia-PL`](https://huggingface.co/datasets/mteb/DBPedia-PL) • **License:** mit • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Encyclopaedic, Written | derived | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Hasibi:2017:DVT,
      author = {Hasibi, Faegheh and Nikolaev, Fedor and Xiong, Chenyan and Balog, Krisztian and Bratsberg, Svein Erik and Kotov, Alexander and Callan, Jamie},
      booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
      doi = {10.1145/3077136.3080751},
      pages = {1265--1268},
      publisher = {ACM},
      series = {SIGIR '17},
      title = {DBpedia-Entity V2: A Test Collection for Entity Search},
      year = {2017},
    }
    
    ```
    



#### DBPedia-PLHardNegatives

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/DBPedia-PLHardNegatives`](https://huggingface.co/datasets/mteb/DBPedia-PLHardNegatives) • **License:** mit • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Encyclopaedic, Written | derived | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Hasibi:2017:DVT,
      author = {Hasibi, Faegheh and Nikolaev, Fedor and Xiong, Chenyan and Balog, Krisztian and Bratsberg, Svein Erik and Kotov, Alexander and Callan, Jamie},
      booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
      doi = {10.1145/3077136.3080751},
      pages = {1265--1268},
      publisher = {ACM},
      series = {SIGIR '17},
      title = {DBpedia-Entity V2: A Test Collection for Entity Search},
      year = {2017},
    }
    
    ```
    



#### DBPedia-VN

A translated dataset from DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/dbpedia-vn`](https://huggingface.co/datasets/GreenNode/dbpedia-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Encyclopaedic, Written | derived | machine-translated and LM verified |



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
    



#### DBPediaHardNegatives

DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/DBPedia_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/DBPedia_test_top_250_only_w_correct-v2) • **License:** mit • [Learn more →](https://github.com/iai-group/DBpedia-Entity/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Hasibi:2017:DVT,
      author = {Hasibi, Faegheh and Nikolaev, Fedor and Xiong, Chenyan and Balog, Krisztian and Bratsberg, Svein Erik and Kotov, Alexander and Callan, Jamie},
      booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
      doi = {10.1145/3077136.3080751},
      pages = {1265--1268},
      publisher = {ACM},
      series = {SIGIR '17},
      title = {DBpedia-Entity V2: A Test Collection for Entity Search},
      year = {2017},
    }
    
    ```
    



#### DS1000Retrieval

A code retrieval task based on 1,000 data science programming problems from DS-1000. Each query is a natural language description of a data science task (e.g., 'Create a scatter plot of column A vs column B with matplotlib'), and the corpus contains Python code implementations using libraries like pandas, numpy, matplotlib, scikit-learn, and scipy. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains Python function implementations focused on data science workflows.

**Dataset:** [`embedding-benchmark/DS1000`](https://huggingface.co/datasets/embedding-benchmark/DS1000) • **License:** mit • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/DS1000)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng, python | Programming | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{lai2022ds,
      author = {Lai, Yuhang and Li, Chengxi and Wang, Yiming and Zhang, Tianyi and Zhong, Ruiqi and Zettlemoyer, Luke and Yih, Wen-tau and Fried, Daniel and Wang, Sida and Yu, Tao},
      journal = {arXiv preprint arXiv:2211.11501},
      title = {DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation},
      year = {2022},
    }
    
    ```
    



#### DanFEVER

A Danish dataset intended for misinformation research. It follows the same format as the English FEVER dataset.

**Dataset:** [`mteb/DanFEVER`](https://huggingface.co/datasets/mteb/DanFEVER) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2021.nodalida-main.47/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | dan | Encyclopaedic, Non-fiction, Spoken | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{norregaard-derczynski-2021-danfever,
      abstract = {We present a dataset, DanFEVER, intended for multilingual misinformation research. The dataset is in Danish and has the same format as the well-known English FEVER dataset. It can be used for testing methods in multilingual settings, as well as for creating models in production for the Danish language.},
      address = {Reykjavik, Iceland (Online)},
      author = {N{\o}rregaard, Jeppe  and
    Derczynski, Leon},
      booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
      editor = {Dobnik, Simon  and
    {\O}vrelid, Lilja},
      month = may # { 31--2 } # jun,
      pages = {422--428},
      publisher = {Link{\"o}ping University Electronic Press, Sweden},
      title = {{D}an{FEVER}: claim verification dataset for {D}anish},
      url = {https://aclanthology.org/2021.nodalida-main.47},
      year = {2021},
    }
    
    ```
    



#### DanFeverRetrieval

A Danish dataset intended for misinformation research. It follows the same format as the English FEVER dataset. DanFeverRetrieval fixed an issue in DanFever where some corpus entries were incorrectly removed.

**Dataset:** [`strombergnlp/danfever`](https://huggingface.co/datasets/strombergnlp/danfever) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2021.nodalida-main.47/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | dan | Encyclopaedic, Non-fiction, Spoken | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{norregaard-derczynski-2021-danfever,
      abstract = {We present a dataset, DanFEVER, intended for multilingual misinformation research. The dataset is in Danish and has the same format as the well-known English FEVER dataset. It can be used for testing methods in multilingual settings, as well as for creating models in production for the Danish language.},
      address = {Reykjavik, Iceland (Online)},
      author = {N{\o}rregaard, Jeppe  and
    Derczynski, Leon},
      booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
      editor = {Dobnik, Simon  and
    {\O}vrelid, Lilja},
      month = may # { 31--2 } # jun,
      pages = {422--428},
      publisher = {Link{\"o}ping University Electronic Press, Sweden},
      title = {{D}an{FEVER}: claim verification dataset for {D}anish},
      url = {https://aclanthology.org/2021.nodalida-main.47},
      year = {2021},
    }
    
    ```
    



#### DuRetrieval

A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine

**Dataset:** [`mteb/DuRetrieval`](https://huggingface.co/datasets/mteb/DuRetrieval) • **License:** not specified • [Learn more →](https://aclanthology.org/2022.emnlp-main.357.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | not specified | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{qiu2022dureaderretrieval,
      archiveprefix = {arXiv},
      author = {Yifu Qiu and Hongyu Li and Yingqi Qu and Ying Chen and Qiaoqiao She and Jing Liu and Hua Wu and Haifeng Wang},
      eprint = {2203.10232},
      primaryclass = {cs.CL},
      title = {DuReader_retrieval: A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine},
      year = {2022},
    }
    
    ```
    



#### EcomRetrieval

EcomRetrieval

**Dataset:** [`mteb/EcomRetrieval`](https://huggingface.co/datasets/mteb/EcomRetrieval) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2203.03367)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | not specified | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{long2022multicprmultidomainchinese,
      archiveprefix = {arXiv},
      author = {Dingkun Long and Qiong Gao and Kuan Zou and Guangwei Xu and Pengjun Xie and Ruijie Guo and Jian Xu and Guanjun Jiang and Luxi Xing and Ping Yang},
      eprint = {2203.03367},
      primaryclass = {cs.IR},
      title = {Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval},
      url = {https://arxiv.org/abs/2203.03367},
      year = {2022},
    }
    
    ```
    



#### EstQA

EstQA is an Estonian question answering dataset based on Wikipedia.

**Dataset:** [`kardosdrur/estonian-qa`](https://huggingface.co/datasets/kardosdrur/estonian-qa) • **License:** not specified • [Learn more →](https://www.semanticscholar.org/paper/Extractive-Question-Answering-for-Estonian-Language-182912IAPM-Alum%C3%A4e/ea4f60ab36cadca059c880678bc4c51e293a85d6?utm_source=direct_link)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | est | Encyclopaedic, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @mastersthesis{mastersthesis,
      author = {Anu Käver},
      school = {Tallinn University of Technology (TalTech)},
      title = {Extractive Question Answering for Estonian Language},
      year = {2021},
    }
    
    ```
    



#### FEVER

FEVER (Fact Extraction and VERification) consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from.

**Dataset:** [`mteb/fever`](https://huggingface.co/datasets/mteb/fever) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://fever.ai/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{thorne-etal-2018-fever,
      abstract = {In this paper we introduce a new publicly available dataset for verification against textual sources, FEVER: Fact Extraction and VERification. It consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. The claims are classified as Supported, Refuted or NotEnoughInfo by annotators achieving 0.6841 in Fleiss kappa. For the first two classes, the annotators also recorded the sentence(s) forming the necessary evidence for their judgment. To characterize the challenge of the dataset presented, we develop a pipeline approach and compare it to suitably designed oracles. The best accuracy we achieve on labeling a claim accompanied by the correct evidence is 31.87{\%}, while if we ignore the evidence we achieve 50.91{\%}. Thus we believe that FEVER is a challenging testbed that will help stimulate progress on claim verification against textual sources.},
      address = {New Orleans, Louisiana},
      author = {Thorne, James  and
    Vlachos, Andreas  and
    Christodoulopoulos, Christos  and
    Mittal, Arpit},
      booktitle = {Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
      doi = {10.18653/v1/N18-1074},
      editor = {Walker, Marilyn  and
    Ji, Heng  and
    Stent, Amanda},
      month = jun,
      pages = {809--819},
      publisher = {Association for Computational Linguistics},
      title = {{FEVER}: a Large-scale Dataset for Fact Extraction and {VER}ification},
      url = {https://aclanthology.org/N18-1074},
      year = {2018},
    }
    
    ```
    



#### FEVER-FaHardNegatives

FEVER-FaHardNegatives

**Dataset:** [`MCINext/FEVER_FA_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/MCINext/FEVER_FA_test_top_250_only_w_correct-v2) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/FEVER_FA_test_top_250_only_w_correct-v2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Encyclopaedic, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### FEVER-NL

FEVER (Fact Extraction and VERification) consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. FEVER-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-fever`](https://huggingface.co/datasets/clips/beir-nl-fever) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-fever)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Encyclopaedic, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



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
| text to text (t2t) | ndcg_at_10 | vie | Encyclopaedic, Written | derived | machine-translated and LM verified |



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
    



#### FEVERHardNegatives

FEVER (Fact Extraction and VERification) consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/FEVER_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/FEVER_test_top_250_only_w_correct-v2) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://fever.ai/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{thorne-etal-2018-fever,
      abstract = {In this paper we introduce a new publicly available dataset for verification against textual sources, FEVER: Fact Extraction and VERification. It consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. The claims are classified as Supported, Refuted or NotEnoughInfo by annotators achieving 0.6841 in Fleiss kappa. For the first two classes, the annotators also recorded the sentence(s) forming the necessary evidence for their judgment. To characterize the challenge of the dataset presented, we develop a pipeline approach and compare it to suitably designed oracles. The best accuracy we achieve on labeling a claim accompanied by the correct evidence is 31.87{\%}, while if we ignore the evidence we achieve 50.91{\%}. Thus we believe that FEVER is a challenging testbed that will help stimulate progress on claim verification against textual sources.},
      address = {New Orleans, Louisiana},
      author = {Thorne, James  and
    Vlachos, Andreas  and
    Christodoulopoulos, Christos  and
    Mittal, Arpit},
      booktitle = {Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
      doi = {10.18653/v1/N18-1074},
      editor = {Walker, Marilyn  and
    Ji, Heng  and
    Stent, Amanda},
      month = jun,
      pages = {809--819},
      publisher = {Association for Computational Linguistics},
      title = {{FEVER}: a Large-scale Dataset for Fact Extraction and {VER}ification},
      url = {https://aclanthology.org/N18-1074},
      year = {2018},
    }
    
    ```
    



#### FQuADRetrieval

This dataset has been built from the French SQuad dataset.

**Dataset:** [`manu/fquad2_test`](https://huggingface.co/datasets/manu/fquad2_test) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/manu/fquad2_test)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fra | Encyclopaedic, Written | human-annotated | created |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{dhoffschmidt-etal-2020-fquad,
      address = {Online},
      author = {d{'}Hoffschmidt, Martin  and
    Belblidia, Wacim  and
    Heinrich, Quentin  and
    Brendl{\'e}, Tom  and
    Vidal, Maxime},
      booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
      doi = {10.18653/v1/2020.findings-emnlp.107},
      editor = {Cohn, Trevor  and
    He, Yulan  and
    Liu, Yang},
      month = nov,
      pages = {1193--1208},
      publisher = {Association for Computational Linguistics},
      title = {{FQ}u{AD}: {F}rench Question Answering Dataset},
      url = {https://aclanthology.org/2020.findings-emnlp.107},
      year = {2020},
    }
    
    ```
    



#### FaithDial

FaithDial is a faithful knowledge-grounded dialogue benchmark.It was curated by asking annotators to amend hallucinated utterances in Wizard of Wikipedia (WoW). It consists of conversation histories along with manually labelled relevant passage. For the purpose of retrieval, we only consider the instances marked as 'Edification' in the VRM field, as the gold passage associated with these instances is non-ambiguous.

**Dataset:** [`mteb/FaithDial`](https://huggingface.co/datasets/mteb/FaithDial) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://mcgill-nlp.github.io/FaithDial)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{dziri2022faithdial,
      author = {Dziri, Nouha and Kamalloo, Ehsan and Milton, Sivan and Zaiane, Osmar and Yu, Mo and Ponti, Edoardo M and Reddy, Siva},
      doi = {10.1162/tacl_a_00529},
      journal = {Transactions of the Association for Computational Linguistics},
      month = {12},
      pages = {1473--1490},
      publisher = {MIT Press},
      title = {{FaithDial: A Faithful Benchmark for Information-Seeking Dialogue}},
      volume = {10},
      year = {2022},
    }
    
    ```
    



#### FeedbackQARetrieval

Using Interactive Feedback to Improve the Accuracy and Explainability of Question Answering Systems Post-Deployment

**Dataset:** [`lt2c/fqa`](https://huggingface.co/datasets/lt2c/fqa) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2204.03025)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | precision_at_1 | eng | Government, Medical, Web, Written | human-annotated | created |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{li-etal-2022-using,
      address = {Dublin, Ireland},
      author = {Li, Zichao  and
    Sharma, Prakhar  and
    Lu, Xing Han  and
    Cheung, Jackie  and
    Reddy, Siva},
      booktitle = {Findings of the Association for Computational Linguistics: ACL 2022},
      doi = {10.18653/v1/2022.findings-acl.75},
      editor = {Muresan, Smaranda  and
    Nakov, Preslav  and
    Villavicencio, Aline},
      month = may,
      pages = {926--937},
      publisher = {Association for Computational Linguistics},
      title = {Using Interactive Feedback to Improve the Accuracy and Explainability of Question Answering Systems Post-Deployment},
      url = {https://aclanthology.org/2022.findings-acl.75},
      year = {2022},
    }
    
    ```
    



#### FiQA-PL

Financial Opinion Mining and Question Answering

**Dataset:** [`mteb/FiQA-PL`](https://huggingface.co/datasets/mteb/FiQA-PL) • **License:** not specified • [Learn more →](https://sites.google.com/view/fiqa/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Financial, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{thakur2021beir,
      author = {Nandan Thakur and Nils Reimers and Andreas R{\"u}ckl{\'e} and Abhishek Srivastava and Iryna Gurevych},
      booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
      title = {{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
      url = {https://openreview.net/forum?id=wCu6T5xFjeJ},
      year = {2021},
    }
    
    ```
    



#### FiQA2018

Financial Opinion Mining and Question Answering

**Dataset:** [`mteb/fiqa`](https://huggingface.co/datasets/mteb/fiqa) • **License:** not specified • [Learn more →](https://sites.google.com/view/fiqa/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Financial, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{thakur2021beir,
      author = {Nandan Thakur and Nils Reimers and Andreas R{\"u}ckl{\'e} and Abhishek Srivastava and Iryna Gurevych},
      booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
      title = {{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
      url = {https://openreview.net/forum?id=wCu6T5xFjeJ},
      year = {2021},
    }
    
    ```
    



#### FiQA2018-Fa

FiQA2018-Fa

**Dataset:** [`MCINext/fiqa-fa`](https://huggingface.co/datasets/MCINext/fiqa-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/fiqa-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### FiQA2018-Fa.v2

FiQA2018-Fa.v2

**Dataset:** [`MCINext/fiqa-fa-v2`](https://huggingface.co/datasets/MCINext/fiqa-fa-v2) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/fiqa-fa-v2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### FiQA2018-NL

Financial Opinion Mining and Question Answering. FiQA2018-NL is a Dutch translation

**Dataset:** [`clips/beir-nl-fiqa`](https://huggingface.co/datasets/clips/beir-nl-fiqa) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-fiqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Non-fiction, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### FiQA2018-VN

A translated dataset from Financial Opinion Mining and Question Answering
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/fiqa-vn`](https://huggingface.co/datasets/GreenNode/fiqa-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://sites.google.com/view/fiqa/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Financial, Written | derived | machine-translated and LM verified |



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
    



#### FinQARetrieval

A financial retrieval task based on FinQA dataset containing numerical reasoning questions over financial documents. Each query is a financial question requiring numerical computation (e.g., 'What is the percentage change in operating expenses from 2019 to 2020?'), and the corpus contains financial document text with tables and numerical data. The task is to retrieve the correct financial information that enables answering the numerical question. Queries are numerical reasoning questions while the corpus contains financial text passages with embedded tables, figures, and quantitative financial data from earnings reports.

**Dataset:** [`embedding-benchmark/FinQA`](https://huggingface.co/datasets/embedding-benchmark/FinQA) • **License:** mit • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/FinQA)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Financial | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{chen2021finqa,
      author = {Chen, Zhiyu and Chen, Wenhu and Smiley, Charese and Shah, Sameena and Borova, Iana and Langdon, Dylan and Moussa, Reema and Beane, Matt and Huang, Ting-Hao and Routledge, Bryan and Wang, William Yang},
      journal = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
      title = {FinQA: A Dataset of Numerical Reasoning over Financial Data},
      year = {2021},
    }
    
    ```
    



#### FinanceBenchRetrieval

A financial retrieval task based on FinanceBench dataset containing financial questions and answers. Each query is a financial question (e.g., 'What was the total revenue in Q3 2023?'), and the corpus contains financial document excerpts and annual reports. The task is to retrieve the correct financial information that answers the question. Queries are financial questions while the corpus contains relevant excerpts from financial documents, earnings reports, and SEC filings with detailed financial data and metrics.

**Dataset:** [`embedding-benchmark/FinanceBench`](https://huggingface.co/datasets/embedding-benchmark/FinanceBench) • **License:** mit • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/FinanceBench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Financial | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{islam2023financebench,
      author = {Islam, Pranab and Kannappan, Anand and Kiela, Douwe and Fergus, Rob and Ott, Myle and Wang, Sam and Garimella, Aparna and Garcia, Nino},
      journal = {arXiv preprint arXiv:2311.11944},
      title = {FinanceBench: A New Benchmark for Financial Question Answering},
      year = {2023},
    }
    
    ```
    



#### FreshStackRetrieval

A code retrieval task based on FreshStack dataset containing programming problems across multiple languages. Each query is a natural language description of a programming task (e.g., 'Write a function to reverse a string using recursion'), and the corpus contains code implementations in Python, JavaScript, and Go. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains function implementations with proper syntax and logic across different programming languages.

**Dataset:** [`embedding-benchmark/FreshStack_mteb`](https://huggingface.co/datasets/embedding-benchmark/FreshStack_mteb) • **License:** mit • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/FreshStack_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng, go, javascript, python | Programming | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{freshstack2023,
      author = {FreshStack Authors},
      journal = {arXiv preprint arXiv:2301.12345},
      title = {FreshStack: A Multi-language Code Generation and Retrieval Benchmark},
      year = {2023},
    }
    
    ```
    



#### GeorgianFAQRetrieval

Frequently asked questions (FAQs) and answers mined from Georgian websites via Common Crawl.

**Dataset:** [`jupyterjazz/georgian-faq`](https://huggingface.co/datasets/jupyterjazz/georgian-faq) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jupyterjazz/georgian-faq)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | kat | Web, Written | derived | created |



#### GerDaLIR

GerDaLIR is a legal information retrieval dataset created from the Open Legal Data platform.

**Dataset:** [`mteb/GerDaLIRSmall`](https://huggingface.co/datasets/mteb/GerDaLIRSmall) • **License:** not specified • [Learn more →](https://github.com/lavis-nlp/GerDaLIR)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | deu | Legal | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{wrzalik-krechel-2021-gerdalir,
      abstract = {We present GerDaLIR, a German Dataset for Legal Information Retrieval based on case documents from the open legal information platform Open Legal Data. The dataset consists of 123K queries, each labelled with at least one relevant document in a collection of 131K case documents. We conduct several baseline experiments including BM25 and a state-of-the-art neural re-ranker. With our dataset, we aim to provide a standardized benchmark for German LIR and promote open research in this area. Beyond that, our dataset comprises sufficient training data to be used as a downstream task for German or multilingual language models.},
      address = {Punta Cana, Dominican Republic},
      author = {Wrzalik, Marco  and
    Krechel, Dirk},
      booktitle = {Proceedings of the Natural Legal Language Processing Workshop 2021},
      month = nov,
      pages = {123--128},
      publisher = {Association for Computational Linguistics},
      title = {{G}er{D}a{LIR}: A {G}erman Dataset for Legal Information Retrieval},
      url = {https://aclanthology.org/2021.nllp-1.13},
      year = {2021},
    }
    
    ```
    



#### GerDaLIRSmall

The dataset consists of documents, passages and relevance labels in German. In contrast to the original dataset, only documents that have corresponding queries in the query set are chosen to create a smaller corpus for evaluation purposes.

**Dataset:** [`mteb/GerDaLIRSmall`](https://huggingface.co/datasets/mteb/GerDaLIRSmall) • **License:** mit • [Learn more →](https://github.com/lavis-nlp/GerDaLIR)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | deu | Legal, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{wrzalik-krechel-2021-gerdalir,
      abstract = {We present GerDaLIR, a German Dataset for Legal Information Retrieval based on case documents from the open legal information platform Open Legal Data. The dataset consists of 123K queries, each labelled with at least one relevant document in a collection of 131K case documents. We conduct several baseline experiments including BM25 and a state-of-the-art neural re-ranker. With our dataset, we aim to provide a standardized benchmark for German LIR and promote open research in this area. Beyond that, our dataset comprises sufficient training data to be used as a downstream task for German or multilingual language models.},
      address = {Punta Cana, Dominican Republic},
      author = {Wrzalik, Marco  and
    Krechel, Dirk},
      booktitle = {Proceedings of the Natural Legal Language Processing Workshop 2021},
      month = nov,
      pages = {123--128},
      publisher = {Association for Computational Linguistics},
      title = {{G}er{D}a{LIR}: A {G}erman Dataset for Legal Information Retrieval},
      url = {https://aclanthology.org/2021.nllp-1.13},
      year = {2021},
    }
    
    ```
    



#### GermanDPR

GermanDPR is a German Question Answering dataset for open-domain QA. It associates questions with a textual context containing the answer

**Dataset:** [`mteb/GermanDPR`](https://huggingface.co/datasets/mteb/GermanDPR) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/deepset/germandpr)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | deu | Non-fiction, Web, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{möller2021germanquad,
      archiveprefix = {arXiv},
      author = {Timo Möller and Julian Risch and Malte Pietsch},
      eprint = {2104.12741},
      primaryclass = {cs.CL},
      title = {GermanQuAD and GermanDPR: Improving Non-English Question Answering and Passage Retrieval},
      year = {2021},
    }
    
    ```
    



#### GermanGovServiceRetrieval

LHM-Dienstleistungen-QA is a German question answering dataset for government services of the Munich city administration. It associates questions with a textual context containing the answer

**Dataset:** [`it-at-m/LHM-Dienstleistungen-QA`](https://huggingface.co/datasets/it-at-m/LHM-Dienstleistungen-QA) • **License:** mit • [Learn more →](https://huggingface.co/datasets/it-at-m/LHM-Dienstleistungen-QA)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_5 | deu | Government, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @software{lhm-dienstleistungen-qa,
      author = {Schröder, Leon Marius and
    Gutknecht, Clemens and
    Alkiddeh, Oubada and
    Susanne Weiß,
    Lukas, Leon},
      month = nov,
      publisher = {it@M},
      title = {LHM-Dienstleistungen-QA - german public domain question-answering dataset},
      url = {https://huggingface.co/datasets/it-at-m/LHM-Dienstleistungen-QA},
      year = {2022},
    }
    
    ```
    



#### GermanQuAD-Retrieval

Context Retrieval for German Question Answering

**Dataset:** [`mteb/germanquad-retrieval`](https://huggingface.co/datasets/mteb/germanquad-retrieval) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/deepset/germanquad)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | mrr_at_5 | deu | Non-fiction, Web, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{möller2021germanquad,
      archiveprefix = {arXiv},
      author = {Timo Möller and Julian Risch and Malte Pietsch},
      eprint = {2104.12741},
      primaryclass = {cs.CL},
      title = {GermanQuAD and GermanDPR: Improving Non-English Question Answering and Passage Retrieval},
      year = {2021},
    }
    
    ```
    



#### GovReport

A dataset for evaluating the ability of information retrieval models to retrieve lengthy US government reports from their summaries.

**Dataset:** [`isaacus/mteb-GovReport`](https://huggingface.co/datasets/isaacus/mteb-GovReport) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/launch/gov_report)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Government, Legal | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{huang-etal-2021-efficient,
      abstract = {The quadratic computational and memory complexities of large Transformers have limited their scalability for long document summarization. In this paper, we propose Hepos, a novel efficient encoder-decoder attention with head-wise positional strides to effectively pinpoint salient information from the source. We further conduct a systematic study of existing efficient self-attentions. Combined with Hepos, we are able to process ten times more tokens than existing models that use full attentions. For evaluation, we present a new dataset, GovReport, with significantly longer documents and summaries. Results show that our models produce significantly higher ROUGE scores than competitive comparisons, including new state-of-the-art results on PubMed. Human evaluation also shows that our models generate more informative summaries with fewer unfaithful errors.},
      address = {Online},
      author = {Huang, Luyang  and
    Cao, Shuyang  and
    Parulian, Nikolaus  and
    Ji, Heng  and
    Wang, Lu},
      booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
      doi = {10.18653/v1/2021.naacl-main.112},
      eprint = {2104.02112},
      month = jun,
      pages = {1419--1436},
      publisher = {Association for Computational Linguistics},
      title = {Efficient Attentions for Long Document Summarization},
      url = {https://aclanthology.org/2021.naacl-main.112},
      year = {2021},
    }
    
    ```
    



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
| text to text (t2t) | ndcg_at_10 | eng | Financial | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{guo2023hc3,
      author = {Guo, Biyang and Zhang, Xin and Wang, Zhiyuan and Jiang, Mingyuan and Nie, Jinran and Ding, Yuxuan and Yue, Jianwei and Wu, Yupeng},
      journal = {arXiv preprint arXiv:2301.07597},
      title = {How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection},
      year = {2023},
    }
    
    ```
    



#### HagridRetrieval

HAGRID (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset)is a dataset for generative information-seeking scenarios. It consists of queriesalong with a set of manually labelled relevant passages

**Dataset:** [`mteb/HagridRetrieval`](https://huggingface.co/datasets/mteb/HagridRetrieval) • **License:** apache-2.0 • [Learn more →](https://github.com/project-miracl/hagrid)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{hagrid,
      author = {Ehsan Kamalloo and Aref Jafari and Xinyu Zhang and Nandan Thakur and Jimmy Lin},
      journal = {arXiv:2307.16883},
      title = {{HAGRID}: A Human-LLM Collaborative Dataset for Generative Information-Seeking with Attribution},
      year = {2023},
    }
    
    ```
    



#### HellaSwag

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on HellaSwag.

**Dataset:** [`mteb/HellaSwag`](https://huggingface.co/datasets/mteb/HellaSwag) • **License:** mit • [Learn more →](https://rowanzellers.com/hellaswag/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    @article{zellers2019hellaswag,
      author = {Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
      journal = {arXiv preprint arXiv:1905.07830},
      title = {Hellaswag: Can a machine really finish your sentence?},
      year = {2019},
    }
    
    ```
    



#### HotpotQA

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.

**Dataset:** [`mteb/hotpotqa`](https://huggingface.co/datasets/mteb/hotpotqa) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Web, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{yang-etal-2018-hotpotqa,
      abstract = {Existing question answering (QA) datasets fail to train QA systems to perform complex reasoning and provide explanations for answers. We introduce HotpotQA, a new dataset with 113k Wikipedia-based question-answer pairs with four key features: (1) the questions require finding and reasoning over multiple supporting documents to answer; (2) the questions are diverse and not constrained to any pre-existing knowledge bases or knowledge schemas; (3) we provide sentence-level supporting facts required for reasoning, allowing QA systems to reason with strong supervision and explain the predictions; (4) we offer a new type of factoid comparison questions to test QA systems{'} ability to extract relevant facts and perform necessary comparison. We show that HotpotQA is challenging for the latest QA systems, and the supporting facts enable models to improve performance and make explainable predictions.},
      address = {Brussels, Belgium},
      author = {Yang, Zhilin  and
    Qi, Peng  and
    Zhang, Saizheng  and
    Bengio, Yoshua  and
    Cohen, William  and
    Salakhutdinov, Ruslan  and
    Manning, Christopher D.},
      booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
      doi = {10.18653/v1/D18-1259},
      editor = {Riloff, Ellen  and
    Chiang, David  and
    Hockenmaier, Julia  and
    Tsujii, Jun{'}ichi},
      month = oct # {-} # nov,
      pages = {2369--2380},
      publisher = {Association for Computational Linguistics},
      title = {{H}otpot{QA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
      url = {https://aclanthology.org/D18-1259},
      year = {2018},
    }
    
    ```
    



#### HotpotQA-Fa

HotpotQA-Fa

**Dataset:** [`MCINext/hotpotqa-fa`](https://huggingface.co/datasets/MCINext/hotpotqa-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/hotpotqa-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Encyclopaedic | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### HotpotQA-FaHardNegatives

HotpotQA-FaHardNegatives

**Dataset:** [`MCINext/HotpotQA_FA_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/MCINext/HotpotQA_FA_test_top_250_only_w_correct-v2) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/HotpotQA_FA_test_top_250_only_w_correct-v2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Encyclopaedic | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### HotpotQA-NL

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strongsupervision for supporting facts to enable more explainable question answering systems. HotpotQA-NL is a Dutch translation. 

**Dataset:** [`clips/beir-nl-hotpotqa`](https://huggingface.co/datasets/clips/beir-nl-hotpotqa) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Web, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### HotpotQA-PL

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.

**Dataset:** [`mteb/HotpotQA-PL`](https://huggingface.co/datasets/mteb/HotpotQA-PL) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web, Written | derived | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### HotpotQA-PLHardNegatives

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/HotpotQA-PLHardNegatives`](https://huggingface.co/datasets/mteb/HotpotQA-PLHardNegatives) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web, Written | derived | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



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
| text to text (t2t) | ndcg_at_10 | vie | Web, Written | derived | machine-translated and LM verified |



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
    



#### HotpotQAHardNegatives

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.  The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/HotpotQA_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/HotpotQA_test_top_250_only_w_correct-v2) • **License:** cc-by-sa-4.0 • [Learn more →](https://hotpotqa.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Web, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{yang-etal-2018-hotpotqa,
      abstract = {Existing question answering (QA) datasets fail to train QA systems to perform complex reasoning and provide explanations for answers. We introduce HotpotQA, a new dataset with 113k Wikipedia-based question-answer pairs with four key features: (1) the questions require finding and reasoning over multiple supporting documents to answer; (2) the questions are diverse and not constrained to any pre-existing knowledge bases or knowledge schemas; (3) we provide sentence-level supporting facts required for reasoning, allowing QA systems to reason with strong supervision and explain the predictions; (4) we offer a new type of factoid comparison questions to test QA systems{'} ability to extract relevant facts and perform necessary comparison. We show that HotpotQA is challenging for the latest QA systems, and the supporting facts enable models to improve performance and make explainable predictions.},
      address = {Brussels, Belgium},
      author = {Yang, Zhilin  and
    Qi, Peng  and
    Zhang, Saizheng  and
    Bengio, Yoshua  and
    Cohen, William  and
    Salakhutdinov, Ruslan  and
    Manning, Christopher D.},
      booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
      doi = {10.18653/v1/D18-1259},
      editor = {Riloff, Ellen  and
    Chiang, David  and
    Hockenmaier, Julia  and
    Tsujii, Jun{'}ichi},
      month = oct # {-} # nov,
      pages = {2369--2380},
      publisher = {Association for Computational Linguistics},
      title = {{H}otpot{QA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
      url = {https://aclanthology.org/D18-1259},
      year = {2018},
    }
    
    ```
    



#### HumanEvalRetrieval

A code retrieval task based on 164 Python programming problems from HumanEval. Each query is a natural language description of a programming task (e.g., 'Check if in given list of numbers, are any two numbers closer to each other than given threshold'), and the corpus contains Python code implementations. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains Python function implementations with proper indentation and logic.

**Dataset:** [`embedding-benchmark/HumanEval`](https://huggingface.co/datasets/embedding-benchmark/HumanEval) • **License:** mit • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/HumanEval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng, python | Programming | derived | found |



??? quote "Citation"

    
    ```bibtex
    @article{chen2021evaluating,
      archiveprefix = {arXiv},
      author = {Chen, Mark and Tworek, Jerry and Jun, Heewoo and Yuan, Qiming and Pinto, Henrique Ponde de Oliveira and Kaplan, Jared and Edwards, Harri and Burda, Yuri and Joseph, Nicholas and Brockman, Greg and Ray, Alex and Puri, Raul and Krueger, Gretchen and Petrov, Michael and Khlaaf, Heidy and Sastry, Girish and Mishkin, Pamela and Chan, Brooke and Gray, Scott and Ryder, Nick and Pavlov, Mikhail and Power, Alethea and Kaiser, Lukasz and Bavarian, Mohammad and Winter, Clemens and Tillet, Philippe and Such, Felipe Petroski and Cummings, Dave and Plappert, Matthias and Chantzis, Fotios and Barnes, Elizabeth and Herbert-Voss, Ariel and Guss, William Hebgen and Nichol, Alex and Paino, Alex and Tezak, Nikolas and Tang, Jie and Babuschkin, Igor and Balaji, Suchir and Jain, Shantanu and Saunders, William and Hesse, Christopher and Carr, Andrew N. and Leike, Jan and Achiam, Joshua and Misra, Vedant and Morikawa, Evan and Radford, Alec and Knight, Matthew and Brundage, Miles and Murati, Mira and Mayer, Katie and Welinder, Peter and McGrew, Bob and Amodei, Dario and McCandlish, Sam and Sutskever, Ilya and Zaremba, Wojciech},
      eprint = {2107.03374},
      primaryclass = {cs.LG},
      title = {Evaluating Large Language Models Trained on Code},
      year = {2021},
    }
    ```
    



#### HunSum2AbstractiveRetrieval

HunSum-2-abstractive is a Hungarian dataset containing news articles along with lead, titles and metadata.

**Dataset:** [`SZTAKI-HLT/HunSum-2-abstractive`](https://huggingface.co/datasets/SZTAKI-HLT/HunSum-2-abstractive) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2404.03555)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_1 | hun | News, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{barta2024news,
      archiveprefix = {arXiv},
      author = {Botond Barta and Dorina Lakatos and Attila Nagy and Milán Konor Nyist and Judit Ács},
      eprint = {2404.03555},
      primaryclass = {cs.CL},
      title = {From News to Summaries: Building a Hungarian Corpus for Extractive and Abstractive Summarization},
      year = {2024},
    }
    
    ```
    



#### IndicQARetrieval

IndicQA is a manually curated cloze-style reading comprehension dataset that can be used for evaluating question-answering models in 11 Indic languages. It is repurposed retrieving relevant context for each question.

**Dataset:** [`mteb/IndicQARetrieval`](https://huggingface.co/datasets/mteb/IndicQARetrieval) • **License:** cc0-1.0 • [Learn more →](https://arxiv.org/abs/2212.05409)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | asm, ben, guj, hin, kan, ... (11) | Web, Written | human-annotated | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @article{doddapaneni2022towards,
      author = {Sumanth Doddapaneni and Rahul Aralikatte and Gowtham Ramesh and Shreyansh Goyal and Mitesh M. Khapra and Anoop Kunchukuttan and Pratyush Kumar},
      doi = {10.18653/v1/2023.acl-long.693},
      journal = {Annual Meeting of the Association for Computational Linguistics},
      title = {Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages},
      year = {2022},
    }
    
    ```
    



#### JaCWIRRetrieval

JaCWIR is a small-scale Japanese information retrieval evaluation dataset consisting of
5000 question texts and approximately 500k web page titles and web page introductions or summaries
(meta descriptions, etc.). The question texts are created based on one of the 500k web pages,
and that data is used as a positive example for the question text.

**Dataset:** [`mteb/JaCWIRRetrieval`](https://huggingface.co/datasets/mteb/JaCWIRRetrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/hotchpotch/JaCWIR)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Web, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{yuichi-tateno-2024-jacwir,
      author = {Yuichi Tateno},
      title = {JaCWIR: Japanese Casual Web IR - 日本語情報検索評価のための小規模でカジュアルなWebタイトルと概要のデータセット},
      url = {https://huggingface.co/datasets/hotchpotch/JaCWIR},
    }
    
    ```
    



#### JaGovFaqsRetrieval

JaGovFaqs is a dataset consisting of FAQs manully extracted from the website of Japanese bureaus. The dataset consists of 22k FAQs, where the queries (questions) and corpus (answers) have been shuffled, and the goal is to match the answer with the question.

**Dataset:** [`mteb/JaGovFaqsRetrieval`](https://huggingface.co/datasets/mteb/JaGovFaqsRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://github.com/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Web, Written | derived | found |



#### JaQuADRetrieval

Human-annotated question-answer pairs for Japanese wikipedia pages.

**Dataset:** [`mteb/JaQuADRetrieval`](https://huggingface.co/datasets/mteb/JaQuADRetrieval) • **License:** cc-by-sa-3.0 • [Learn more →](https://arxiv.org/abs/2202.01764)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Encyclopaedic, Non-fiction, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{so2022jaquad,
      archiveprefix = {arXiv},
      author = {ByungHoon So and Kyuhong Byun and Kyungwon Kang and Seongjin Cho},
      eprint = {2202.01764},
      primaryclass = {cs.CL},
      title = {{JaQuAD: Japanese Question Answering Dataset for Machine Reading Comprehension}},
      year = {2022},
    }
    
    ```
    



#### JaqketRetrieval

JAQKET (JApanese Questions on Knowledge of EnTities) is a QA dataset that is created based on quiz questions.

**Dataset:** [`mteb/jaqket`](https://huggingface.co/datasets/mteb/jaqket) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/kumapo/JAQKET-dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Encyclopaedic, Non-fiction, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Kurihara_nlp2020,
      author = {鈴木正敏 and 鈴木潤 and 松田耕史 and ⻄田京介 and 井之上直也},
      booktitle = {言語処理学会第26回年次大会},
      note = {in Japanese},
      title = {JAQKET: クイズを題材にした日本語 QA データセットの構築},
      url = {https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf},
      year = {2020},
    }
    
    ```
    



#### Ko-StrategyQA

Ko-StrategyQA

**Dataset:** [`taeminlee/Ko-StrategyQA`](https://huggingface.co/datasets/taeminlee/Ko-StrategyQA) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/taeminlee/Ko-StrategyQA)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | kor | not specified | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @article{geva2021strategyqa,
      author = {Geva, Mor and Khashabi, Daniel and Segal, Elad and Khot, Tushar and Roth, Dan and Berant, Jonathan},
      journal = {Transactions of the Association for Computational Linguistics (TACL)},
      title = {{Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies}},
      year = {2021},
    }
    
    ```
    



#### LEMBNarrativeQARetrieval

narrativeqa subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Fiction, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{kocisky-etal-2018-narrativeqa,
      abstract = {},
      address = {Cambridge, MA},
      author = {Ko{\v{c}}isk{\'y}, Tom{\'a}{\v{s}}  and
    Schwarz, Jonathan  and
    Blunsom, Phil  and
    Dyer, Chris  and
    Hermann, Karl Moritz  and
    Melis, G{\'a}bor  and
    Grefenstette, Edward},
      doi = {10.1162/tacl_a_00023},
      editor = {Lee, Lillian  and
    Johnson, Mark  and
    Toutanova, Kristina  and
    Roark, Brian},
      journal = {Transactions of the Association for Computational Linguistics},
      pages = {317--328},
      publisher = {MIT Press},
      title = {The {N}arrative{QA} Reading Comprehension Challenge},
      url = {https://aclanthology.org/Q18-1023},
      volume = {6},
      year = {2018},
    }
    
    ```
    



#### LEMBNeedleRetrieval

needle subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_1 | eng | Academic, Blog, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{zhu2024longembed,
      author = {Zhu, Dawei and Wang, Liang and Yang, Nan and Song, Yifan and Wu, Wenhao and Wei, Furu and Li, Sujian},
      journal = {arXiv preprint arXiv:2404.12096},
      title = {LongEmbed: Extending Embedding Models for Long Context Retrieval},
      year = {2024},
    }
    
    ```
    



#### LEMBPasskeyRetrieval

passkey subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_1 | eng | Fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{zhu2024longembed,
      author = {Zhu, Dawei and Wang, Liang and Yang, Nan and Song, Yifan and Wu, Wenhao and Wei, Furu and Li, Sujian},
      journal = {arXiv preprint arXiv:2404.12096},
      title = {LongEmbed: Extending Embedding Models for Long Context Retrieval},
      year = {2024},
    }
    
    ```
    



#### LEMBQMSumRetrieval

qmsum subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Spoken, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{zhong-etal-2021-qmsum,
      abstract = {},
      address = {Online},
      author = {Zhong, Ming  and
    Yin, Da  and
    Yu, Tao  and
    Zaidi, Ahmad  and
    Mutuma, Mutethia  and
    Jha, Rahul  and
    Awadallah, Ahmed Hassan  and
    Celikyilmaz, Asli  and
    Liu, Yang  and
    Qiu, Xipeng  and
    Radev, Dragomir},
      booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
      doi = {10.18653/v1/2021.naacl-main.472},
      editor = {Toutanova, Kristina  and
    Rumshisky, Anna  and
    Zettlemoyer, Luke  and
    Hakkani-Tur, Dilek  and
    Beltagy, Iz  and
    Bethard, Steven  and
    Cotterell, Ryan  and
    Chakraborty, Tanmoy  and
    Zhou, Yichao},
      month = jun,
      pages = {5905--5921},
      publisher = {Association for Computational Linguistics},
      title = {{QMS}um: A New Benchmark for Query-based Multi-domain Meeting Summarization},
      url = {https://aclanthology.org/2021.naacl-main.472},
      year = {2021},
    }
    
    ```
    



#### LEMBSummScreenFDRetrieval

summ_screen_fd subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Spoken, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{chen-etal-2022-summscreen,
      abstract = {},
      address = {Dublin, Ireland},
      author = {Chen, Mingda  and
    Chu, Zewei  and
    Wiseman, Sam  and
    Gimpel, Kevin},
      booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      doi = {10.18653/v1/2022.acl-long.589},
      editor = {Muresan, Smaranda  and
    Nakov, Preslav  and
    Villavicencio, Aline},
      month = may,
      pages = {8602--8615},
      publisher = {Association for Computational Linguistics},
      title = {{S}umm{S}creen: A Dataset for Abstractive Screenplay Summarization},
      url = {https://aclanthology.org/2022.acl-long.589},
      year = {2022},
    }
    
    ```
    



#### LEMBWikimQARetrieval

2wikimqa subset of dwzhu/LongEmbed dataset.

**Dataset:** [`dwzhu/LongEmbed`](https://huggingface.co/datasets/dwzhu/LongEmbed) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/dwzhu/LongEmbed)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{ho2020constructing,
      author = {Ho, Xanh and Nguyen, Anh-Khoa Duong and Sugawara, Saku and Aizawa, Akiko},
      booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
      pages = {6609--6625},
      title = {Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps},
      year = {2020},
    }
    
    ```
    



#### LIMITRetrieval

A simple retrieval task designed to test all combinations of top-2 documents. This version includes all 50k docs.

**Dataset:** [`orionweller/LIMIT`](https://huggingface.co/datasets/orionweller/LIMIT) • **License:** apache-2.0 • [Learn more →](https://github.com/google-deepmind/limit)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_2 | eng | Fiction | human-annotated | created |



??? quote "Citation"

    
    ```bibtex
    
    @misc{weller2025theoreticallimit,
      archiveprefix = {arXiv},
      author = {Orion Weller and Michael Boratko and Iftekhar Naim and Jinhyuk Lee},
      eprint = {2508.21038},
      primaryclass = {cs.IR},
      title = {On the Theoretical Limitations of Embedding-Based Retrieval},
      url = {https://arxiv.org/abs/2508.21038},
      year = {2025},
    }
    ```
    



#### LIMITSmallRetrieval

A simple retrieval task designed to test all combinations of top-2 documents. This version only includes the 46 documents that are relevant to the 1000 queries.

**Dataset:** [`orionweller/LIMIT-small`](https://huggingface.co/datasets/orionweller/LIMIT-small) • **License:** apache-2.0 • [Learn more →](https://github.com/google-deepmind/limit)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_2 | eng | Fiction | human-annotated | created |



??? quote "Citation"

    
    ```bibtex
    
    @misc{weller2025theoreticallimit,
      archiveprefix = {arXiv},
      author = {Orion Weller and Michael Boratko and Iftekhar Naim and Jinhyuk Lee},
      eprint = {2508.21038},
      primaryclass = {cs.IR},
      title = {On the Theoretical Limitations of Embedding-Based Retrieval},
      url = {https://arxiv.org/abs/2508.21038},
      year = {2025},
    }
    ```
    



#### LeCaRDv2

The task involves identifying and retrieving the case document that best matches or is most relevant to the scenario described in each of the provided queries.

**Dataset:** [`mteb/LeCaRDv2`](https://huggingface.co/datasets/mteb/LeCaRDv2) • **License:** mit • [Learn more →](https://github.com/THUIR/LeCaRDv2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | zho | Legal, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{li2023lecardv2,
      archiveprefix = {arXiv},
      author = {Haitao Li and Yunqiu Shao and Yueyue Wu and Qingyao Ai and Yixiao Ma and Yiqun Liu},
      eprint = {2310.17609},
      primaryclass = {cs.CL},
      title = {LeCaRDv2: A Large-Scale Chinese Legal Case Retrieval Dataset},
      year = {2023},
    }
    
    ```
    



#### LegalBenchConsumerContractsQA

The dataset includes questions and answers related to contracts.

**Dataset:** [`mteb/legalbench_consumer_contracts_qa`](https://huggingface.co/datasets/mteb/legalbench_consumer_contracts_qa) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench/viewer/consumer_contracts_qa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Legal, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{hendrycks2021cuad,
      author = {Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
      journal = {arXiv preprint arXiv:2103.06268},
      title = {Cuad: An expert-annotated nlp dataset for legal contract review},
      year = {2021},
    }
    
    @article{koreeda2021contractnli,
      author = {Koreeda, Yuta and Manning, Christopher D},
      journal = {arXiv preprint arXiv:2110.01799},
      title = {ContractNLI: A dataset for document-level natural language inference for contracts},
      year = {2021},
    }
    
    ```
    



#### LegalBenchCorporateLobbying

The dataset includes bill titles and bill summaries related to corporate lobbying.

**Dataset:** [`mteb/legalbench_corporate_lobbying`](https://huggingface.co/datasets/mteb/legalbench_corporate_lobbying) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench/viewer/corporate_lobbying)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Legal, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{guha2023legalbench,
      archiveprefix = {arXiv},
      author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
      eprint = {2308.11462},
      primaryclass = {cs.CL},
      title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
      year = {2023},
    }
    
    @article{hendrycks2021cuad,
      author = {Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
      journal = {arXiv preprint arXiv:2103.06268},
      title = {Cuad: An expert-annotated nlp dataset for legal contract review},
      year = {2021},
    }
    
    @article{holzenberger2021factoring,
      author = {Holzenberger, Nils and Van Durme, Benjamin},
      journal = {arXiv preprint arXiv:2105.07903},
      title = {Factoring statutory reasoning as language understanding challenges},
      year = {2021},
    }
    
    @article{koreeda2021contractnli,
      author = {Koreeda, Yuta and Manning, Christopher D},
      journal = {arXiv preprint arXiv:2110.01799},
      title = {ContractNLI: A dataset for document-level natural language inference for contracts},
      year = {2021},
    }
    
    @article{lippi2019claudette,
      author = {Lippi, Marco and Pa{\l}ka, Przemys{\l}aw and Contissa, Giuseppe and Lagioia, Francesca and Micklitz, Hans-Wolfgang and Sartor, Giovanni and Torroni, Paolo},
      journal = {Artificial Intelligence and Law},
      pages = {117--139},
      publisher = {Springer},
      title = {CLAUDETTE: an automated detector of potentially unfair clauses in online terms of service},
      volume = {27},
      year = {2019},
    }
    
    @article{ravichander2019question,
      author = {Ravichander, Abhilasha and Black, Alan W and Wilson, Shomir and Norton, Thomas and Sadeh, Norman},
      journal = {arXiv preprint arXiv:1911.00841},
      title = {Question answering for privacy policies: Combining computational and legal perspectives},
      year = {2019},
    }
    
    @article{wang2023maud,
      author = {Wang, Steven H and Scardigli, Antoine and Tang, Leonard and Chen, Wei and Levkin, Dimitry and Chen, Anya and Ball, Spencer and Woodside, Thomas and Zhang, Oliver and Hendrycks, Dan},
      journal = {arXiv preprint arXiv:2301.00876},
      title = {MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding},
      year = {2023},
    }
    
    @inproceedings{wilson2016creation,
      author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
      booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      pages = {1330--1340},
      title = {The creation and analysis of a website privacy policy corpus},
      year = {2016},
    }
    
    @inproceedings{zheng2021does,
      author = {Zheng, Lucia and Guha, Neel and Anderson, Brandon R and Henderson, Peter and Ho, Daniel E},
      booktitle = {Proceedings of the eighteenth international conference on artificial intelligence and law},
      pages = {159--168},
      title = {When does pretraining help? assessing self-supervised learning for law and the casehold dataset of 53,000+ legal holdings},
      year = {2021},
    }
    
    @article{zimmeck2019maps,
      author = {Zimmeck, Sebastian and Story, Peter and Smullen, Daniel and Ravichander, Abhilasha and Wang, Ziqi and Reidenberg, Joel R and Russell, N Cameron and Sadeh, Norman},
      journal = {Proc. Priv. Enhancing Tech.},
      pages = {66},
      title = {Maps: Scaling privacy compliance analysis to a million apps},
      volume = {2019},
      year = {2019},
    }
    
    ```
    



#### LegalQuAD

The dataset consists of questions and legal documents in German.

**Dataset:** [`mteb/LegalQuAD`](https://huggingface.co/datasets/mteb/LegalQuAD) • **License:** cc-by-4.0 • [Learn more →](https://github.com/Christoph911/AIKE2021_Appendix)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | deu | Legal, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{9723721,
      author = {Hoppe, Christoph and Pelkmann, David and Migenda, Nico and Hötte, Daniel and Schenck, Wolfram},
      booktitle = {2021 IEEE Fourth International Conference on Artificial Intelligence and Knowledge Engineering (AIKE)},
      doi = {10.1109/AIKE52691.2021.00011},
      keywords = {Knowledge engineering;Law;Semantic search;Conferences;Bit error rate;NLP;knowledge extraction;question-answering;semantic search;document retrieval;German language},
      number = {},
      pages = {29-32},
      title = {Towards Intelligent Legal Advisors for Document Retrieval and Question-Answering in German Legal Documents},
      volume = {},
      year = {2021},
    }
    
    ```
    



#### LegalSummarization

The dataset consistes of 439 pairs of contracts and their summarizations from https://tldrlegal.com and https://tosdr.org/.

**Dataset:** [`mteb/legal_summarization`](https://huggingface.co/datasets/mteb/legal_summarization) • **License:** apache-2.0 • [Learn more →](https://github.com/lauramanor/legal_summarization)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Legal, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{manor-li-2019-plain,
      address = {Minneapolis, Minnesota},
      author = {Manor, Laura  and
    Li, Junyi Jessy},
      booktitle = {Proceedings of the Natural Legal Language Processing Workshop 2019},
      month = jun,
      pages = {1--11},
      publisher = {Association for Computational Linguistics},
      title = {Plain {E}nglish Summarization of Contracts},
      url = {https://www.aclweb.org/anthology/W19-2201},
      year = {2019},
    }
    
    ```
    



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



??? quote "Citation"

    
    ```bibtex
    
    @article{ajith2024litsearch,
      author = {Ajith, Anirudh and Xia, Mengzhou and Chevalier, Alexis and Goyal, Tanya and Chen, Danqi and Gao, Tianyu},
      title = {LitSearch: A Retrieval Benchmark for Scientific Literature Search},
      year = {2024},
    }
    
    ```
    



#### LoTTE

LoTTE (Long-Tail Topic-stratified Evaluation for IR) is designed to evaluate retrieval models on underrepresented, long-tail topics. Unlike MSMARCO or BEIR, LoTTE features domain-specific queries and passages from StackExchange (covering writing, recreation, science, technology, and lifestyle), providing a challenging out-of-domain generalization benchmark.

**Dataset:** [`mteb/LoTTE`](https://huggingface.co/datasets/mteb/LoTTE) • **License:** mit • [Learn more →](https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | precision_at_5 | eng | Academic, Social, Web | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{santhanam-etal-2022-colbertv2,
      address = {Seattle, United States},
      author = {Santhanam, Keshav  and
    Khattab, Omar  and
    Saad-Falcon, Jon  and
    Potts, Christopher  and
    Zaharia, Matei},
      booktitle = {Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
      doi = {10.18653/v1/2022.naacl-main.272},
      editor = {Carpuat, Marine  and
    de Marneffe, Marie-Catherine  and
    Meza Ruiz, Ivan Vladimir},
      month = jul,
      pages = {3715--3734},
      publisher = {Association for Computational Linguistics},
      title = {{C}ol{BERT}v2: Effective and Efficient Retrieval via Lightweight Late Interaction},
      url = {https://aclanthology.org/2022.naacl-main.272/},
      year = {2022},
    }
    
    ```
    



#### MBPPRetrieval

A code retrieval task based on 378 Python programming problems from MBPP (Mostly Basic Python Programming). Each query is a natural language description of a programming task (e.g., 'Write a function to find the shared elements from the given two lists'), and the corpus contains Python code implementations. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains Python function implementations with proper syntax and logic.

**Dataset:** [`embedding-benchmark/MBPP`](https://huggingface.co/datasets/embedding-benchmark/MBPP) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/MBPP)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng, python | Programming | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{austin2021program,
      author = {Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
      journal = {arXiv preprint arXiv:2108.07732},
      title = {Program Synthesis with Large Language Models},
      year = {2021},
    }
    
    ```
    



#### MIRACLRetrieval

MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages.

**Dataset:** [`mteb/MIRACLRetrieval`](https://huggingface.co/datasets/mteb/MIRACLRetrieval) • **License:** cc-by-sa-4.0 • [Learn more →](http://miracl.ai/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, ben, deu, eng, fas, ... (18) | Encyclopaedic, Written | expert-annotated | created |



??? quote "Citation"

    
    ```bibtex
    
    @article{10.1162/tacl_a_00595,
      abstract = {{MIRACL is a multilingual dataset for ad hoc retrieval across 18 languages that collectively encompass over three billion native speakers around the world. This resource is designed to support monolingual retrieval tasks, where the queries and the corpora are in the same language. In total, we have gathered over 726k high-quality relevance judgments for 78k queries over Wikipedia in these languages, where all annotations have been performed by native speakers hired by our team. MIRACL covers languages that are both typologically close as well as distant from 10 language families and 13 sub-families, associated with varying amounts of publicly available resources. Extensive automatic heuristic verification and manual assessments were performed during the annotation process to control data quality. In total, MIRACL represents an investment of around five person-years of human annotator effort. Our goal is to spur research on improving retrieval across a continuum of languages, thus enhancing information access capabilities for diverse populations around the world, particularly those that have traditionally been underserved. MIRACL is available at http://miracl.ai/.}},
      author = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
      doi = {10.1162/tacl_a_00595},
      eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00595/2157340/tacl\_a\_00595.pdf},
      issn = {2307-387X},
      journal = {Transactions of the Association for Computational Linguistics},
      month = {09},
      pages = {1114-1131},
      title = {{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}},
      url = {https://doi.org/10.1162/tacl\_a\_00595},
      volume = {11},
      year = {2023},
    }
    
    ```
    



#### MIRACLRetrievalHardNegatives

MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/MIRACLRetrievalHardNegatives`](https://huggingface.co/datasets/mteb/MIRACLRetrievalHardNegatives) • **License:** cc-by-sa-4.0 • [Learn more →](http://miracl.ai/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, ben, deu, eng, fas, ... (18) | Encyclopaedic, Written | expert-annotated | created |



??? quote "Citation"

    
    ```bibtex
    
    @article{10.1162/tacl_a_00595,
      abstract = {{MIRACL is a multilingual dataset for ad hoc retrieval across 18 languages that collectively encompass over three billion native speakers around the world. This resource is designed to support monolingual retrieval tasks, where the queries and the corpora are in the same language. In total, we have gathered over 726k high-quality relevance judgments for 78k queries over Wikipedia in these languages, where all annotations have been performed by native speakers hired by our team. MIRACL covers languages that are both typologically close as well as distant from 10 language families and 13 sub-families, associated with varying amounts of publicly available resources. Extensive automatic heuristic verification and manual assessments were performed during the annotation process to control data quality. In total, MIRACL represents an investment of around five person-years of human annotator effort. Our goal is to spur research on improving retrieval across a continuum of languages, thus enhancing information access capabilities for diverse populations around the world, particularly those that have traditionally been underserved. MIRACL is available at http://miracl.ai/.}},
      author = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
      doi = {10.1162/tacl_a_00595},
      eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00595/2157340/tacl\_a\_00595.pdf},
      issn = {2307-387X},
      journal = {Transactions of the Association for Computational Linguistics},
      month = {09},
      pages = {1114-1131},
      title = {{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}},
      url = {https://doi.org/10.1162/tacl\_a\_00595},
      volume = {11},
      year = {2023},
    }
    
    ```
    



#### MIRACLRetrievalHardNegatives.v2

MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.V2 uses a more appropriate prompt rather than the default prompt for retrieval.

**Dataset:** [`mteb/MIRACLRetrievalHardNegatives`](https://huggingface.co/datasets/mteb/MIRACLRetrievalHardNegatives) • **License:** cc-by-sa-4.0 • [Learn more →](http://miracl.ai/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, ben, deu, eng, fas, ... (18) | Encyclopaedic, Written | expert-annotated | created |



??? quote "Citation"

    
    ```bibtex
    
    @article{10.1162/tacl_a_00595,
      abstract = {{MIRACL is a multilingual dataset for ad hoc retrieval across 18 languages that collectively encompass over three billion native speakers around the world. This resource is designed to support monolingual retrieval tasks, where the queries and the corpora are in the same language. In total, we have gathered over 726k high-quality relevance judgments for 78k queries over Wikipedia in these languages, where all annotations have been performed by native speakers hired by our team. MIRACL covers languages that are both typologically close as well as distant from 10 language families and 13 sub-families, associated with varying amounts of publicly available resources. Extensive automatic heuristic verification and manual assessments were performed during the annotation process to control data quality. In total, MIRACL represents an investment of around five person-years of human annotator effort. Our goal is to spur research on improving retrieval across a continuum of languages, thus enhancing information access capabilities for diverse populations around the world, particularly those that have traditionally been underserved. MIRACL is available at http://miracl.ai/.}},
      author = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
      doi = {10.1162/tacl_a_00595},
      eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00595/2157340/tacl\_a\_00595.pdf},
      issn = {2307-387X},
      journal = {Transactions of the Association for Computational Linguistics},
      month = {09},
      pages = {1114-1131},
      title = {{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}},
      url = {https://doi.org/10.1162/tacl\_a\_00595},
      volume = {11},
      year = {2023},
    }
    
    ```
    



#### MKQARetrieval

Multilingual Knowledge Questions & Answers (MKQA)contains 10,000 queries sampled from the Google Natural Questions dataset.
        For each query we collect new passage-independent answers. These queries and answers are then human translated into 25 Non-English languages.

**Dataset:** [`mteb/MKQARetrieval`](https://huggingface.co/datasets/mteb/MKQARetrieval) • **License:** cc-by-3.0 • [Learn more →](https://github.com/apple/ml-mkqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, dan, deu, eng, fin, ... (26) | Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{mkqa,
      author = {Shayne Longpre and Yi Lu and Joachim Daiber},
      title = {MKQA: A Linguistically Diverse Benchmark for Multilingual Open Domain Question Answering},
      url = {https://arxiv.org/pdf/2007.15207.pdf},
      year = {2020},
    }
    
    ```
    



#### MLQARetrieval

MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating cross-lingual question answering performance.
        MLQA consists of over 5K extractive QA instances (12K in English) in SQuAD format in seven languages - English, Arabic,
        German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA is highly parallel, with QA instances parallel between
        4 different languages on average.

**Dataset:** [`mteb/MLQARetrieval`](https://huggingface.co/datasets/mteb/MLQARetrieval) • **License:** cc-by-sa-3.0 • [Learn more →](https://huggingface.co/datasets/mlqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, deu, eng, hin, spa, ... (7) | Encyclopaedic, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{lewis2019mlqa,
      author = {Lewis, Patrick and Oguz, Barlas and Rinott, Ruty and Riedel, Sebastian and Schwenk, Holger},
      eid = {arXiv: 1910.07475},
      journal = {arXiv preprint arXiv:1910.07475},
      title = {MLQA: Evaluating Cross-lingual Extractive Question Answering},
      year = {2019},
    }
    
    ```
    



#### MLQuestions

MLQuestions is a domain adaptation dataset for the machine learning domainIt consists of ML questions along with passages from Wikipedia machine learning pages (https://en.wikipedia.org/wiki/Category:Machine_learning)

**Dataset:** [`McGill-NLP/mlquestions`](https://huggingface.co/datasets/McGill-NLP/mlquestions) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://github.com/McGill-NLP/MLQuestions)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Encyclopaedic, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{kulshreshtha-etal-2021-back,
      abstract = {In this work, we introduce back-training, an alternative to self-training for unsupervised domain adaptation (UDA). While self-training generates synthetic training data where natural inputs are aligned with noisy outputs, back-training results in natural outputs aligned with noisy inputs. This significantly reduces the gap between target domain and synthetic data distribution, and reduces model overfitting to source domain. We run UDA experiments on question generation and passage retrieval from the Natural Questions domain to machine learning and biomedical domains. We find that back-training vastly outperforms self-training by a mean improvement of 7.8 BLEU-4 points on generation, and 17.6{\%} top-20 retrieval accuracy across both domains. We further propose consistency filters to remove low-quality synthetic data before training. We also release a new domain-adaptation dataset - MLQuestions containing 35K unaligned questions, 50K unaligned passages, and 3K aligned question-passage pairs.},
      address = {Online and Punta Cana, Dominican Republic},
      author = {Kulshreshtha, Devang  and
    Belfer, Robert  and
    Serban, Iulian Vlad  and
    Reddy, Siva},
      booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
      month = nov,
      pages = {7064--7078},
      publisher = {Association for Computational Linguistics},
      title = {Back-Training excels Self-Training at Unsupervised Domain Adaptation of Question Generation and Passage Retrieval},
      url = {https://aclanthology.org/2021.emnlp-main.566},
      year = {2021},
    }
    
    ```
    



#### MMarcoRetrieval

MMarcoRetrieval

**Dataset:** [`mteb/MMarcoRetrieval`](https://huggingface.co/datasets/mteb/MMarcoRetrieval) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2309.07597)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | not specified | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{xiao2024cpack,
      archiveprefix = {arXiv},
      author = {Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff and Defu Lian and Jian-Yun Nie},
      eprint = {2309.07597},
      primaryclass = {cs.CL},
      title = {C-Pack: Packaged Resources To Advance General Chinese Embedding},
      year = {2024},
    }
    
    ```
    



#### MSMARCO

MS MARCO is a collection of datasets focused on deep learning in search

**Dataset:** [`mteb/msmarco`](https://huggingface.co/datasets/mteb/msmarco) • **License:** msr-la-nc • [Learn more →](https://microsoft.github.io/msmarco/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Blog, Encyclopaedic, Government, Medical, ... (10) | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{DBLP:journals/corr/NguyenRSGTMD16,
      archiveprefix = {arXiv},
      author = {Tri Nguyen and
    Mir Rosenberg and
    Xia Song and
    Jianfeng Gao and
    Saurabh Tiwary and
    Rangan Majumder and
    Li Deng},
      bibsource = {dblp computer science bibliography, https://dblp.org},
      biburl = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
      eprint = {1611.09268},
      journal = {CoRR},
      timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
      title = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
      url = {http://arxiv.org/abs/1611.09268},
      volume = {abs/1611.09268},
      year = {2016},
    }
    
    ```
    



#### MSMARCO-Fa

MSMARCO-Fa

**Dataset:** [`MCINext/msmarco-fa`](https://huggingface.co/datasets/MCINext/msmarco-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/msmarco-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### MSMARCO-FaHardNegatives

MSMARCO-FaHardNegatives

**Dataset:** [`MCINext/MSMARCO_FA_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/MCINext/MSMARCO_FA_test_top_250_only_w_correct-v2) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/MSMARCO_FA_test_top_250_only_w_correct-v2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### MSMARCO-PL

MS MARCO is a collection of datasets focused on deep learning in search

**Dataset:** [`mteb/MSMARCO-PL`](https://huggingface.co/datasets/mteb/MSMARCO-PL) • **License:** https://microsoft.github.io/msmarco/ • [Learn more →](https://microsoft.github.io/msmarco/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web, Written | derived | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### MSMARCO-PLHardNegatives

MS MARCO is a collection of datasets focused on deep learning in search. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/MSMARCO-PLHardNegatives`](https://huggingface.co/datasets/mteb/MSMARCO-PLHardNegatives) • **License:** https://microsoft.github.io/msmarco/ • [Learn more →](https://microsoft.github.io/msmarco/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web, Written | derived | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### MSMARCO-VN

A translated dataset from MS MARCO is a collection of datasets focused on deep learning in search
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/msmarco-vn`](https://huggingface.co/datasets/GreenNode/msmarco-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://microsoft.github.io/msmarco/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Academic, Blog, Encyclopaedic, Government, Medical, ... (10) | derived | machine-translated and LM verified |



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
    



#### MSMARCOHardNegatives

MS MARCO is a collection of datasets focused on deep learning in search. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/MSMARCO_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/MSMARCO_test_top_250_only_w_correct-v2) • **License:** msr-la-nc • [Learn more →](https://microsoft.github.io/msmarco/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Blog, Encyclopaedic, Government, Medical, ... (10) | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{DBLP:journals/corr/NguyenRSGTMD16,
      archiveprefix = {arXiv},
      author = {Tri Nguyen and
    Mir Rosenberg and
    Xia Song and
    Jianfeng Gao and
    Saurabh Tiwary and
    Rangan Majumder and
    Li Deng},
      bibsource = {dblp computer science bibliography, https://dblp.org},
      biburl = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
      eprint = {1611.09268},
      journal = {CoRR},
      timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
      title = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
      url = {http://arxiv.org/abs/1611.09268},
      volume = {abs/1611.09268},
      year = {2016},
    }
    
    ```
    



#### MSMARCOv2

MS MARCO is a collection of datasets focused on deep learning in search. This version is derived from BEIR

**Dataset:** [`mteb/msmarco-v2`](https://huggingface.co/datasets/mteb/msmarco-v2) • **License:** msr-la-nc • [Learn more →](https://microsoft.github.io/msmarco/TREC-Deep-Learning.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Blog, Encyclopaedic, Government, Medical, ... (10) | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{DBLP:journals/corr/NguyenRSGTMD16,
      archiveprefix = {arXiv},
      author = {Tri Nguyen and
    Mir Rosenberg and
    Xia Song and
    Jianfeng Gao and
    Saurabh Tiwary and
    Rangan Majumder and
    Li Deng},
      bibsource = {dblp computer science bibliography, https://dblp.org},
      biburl = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
      eprint = {1611.09268},
      journal = {CoRR},
      timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
      title = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
      url = {http://arxiv.org/abs/1611.09268},
      volume = {abs/1611.09268},
      year = {2016},
    }
    
    ```
    



#### MedicalQARetrieval

The dataset consists 2048 medical question and answer pairs.

**Dataset:** [`mteb/medical_qa`](https://huggingface.co/datasets/mteb/medical_qa) • **License:** cc0-1.0 • [Learn more →](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{BenAbacha-BMC-2019,
      author = {Asma, Ben Abacha and Dina, Demner{-}Fushman},
      journal = {{BMC} Bioinform.},
      number = {1},
      pages = {511:1--511:23},
      title = {A Question-Entailment Approach to Question Answering},
      url = {https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4},
      volume = {20},
      year = {2019},
    }
    
    ```
    



#### MedicalRetrieval

MedicalRetrieval

**Dataset:** [`mteb/MedicalRetrieval`](https://huggingface.co/datasets/mteb/MedicalRetrieval) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2203.03367)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | not specified | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{long2022multicprmultidomainchinese,
      archiveprefix = {arXiv},
      author = {Dingkun Long and Qiong Gao and Kuan Zou and Guangwei Xu and Pengjun Xie and Ruijie Guo and Jian Xu and Guanjun Jiang and Luxi Xing and Ping Yang},
      eprint = {2203.03367},
      primaryclass = {cs.IR},
      title = {Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval},
      url = {https://arxiv.org/abs/2203.03367},
      year = {2022},
    }
    
    ```
    



#### MintakaRetrieval

We introduce Mintaka, a complex, natural, and multilingual dataset designed for experimenting with end-to-end question-answering models. Mintaka is composed of 20,000 question-answer pairs collected in English, annotated with Wikidata entities, and translated into Arabic, French, German, Hindi, Italian, Japanese, Portuguese, and Spanish for a total of 180,000 samples. Mintaka includes 8 types of complex questions, including superlative, intersection, and multi-hop questions, which were naturally elicited from crowd workers. 

**Dataset:** [`mteb/MintakaRetrieval`](https://huggingface.co/datasets/mteb/MintakaRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/mteb/MintakaRetrieval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, deu, fra, hin, ita, ... (8) | Encyclopaedic, Written | derived | human-translated |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{sen-etal-2022-mintaka,
      address = {Gyeongju, Republic of Korea},
      author = {Sen, Priyanka  and
    Aji, Alham Fikri  and
    Saffari, Amir},
      booktitle = {Proceedings of the 29th International Conference on Computational Linguistics},
      month = oct,
      pages = {1604--1619},
      publisher = {International Committee on Computational Linguistics},
      title = {Mintaka: A Complex, Natural, and Multilingual Dataset for End-to-End Question Answering},
      url = {https://aclanthology.org/2022.coling-1.138},
      year = {2022},
    }
    
    ```
    



#### MrTidyRetrieval

Mr. TyDi is a multi-lingual benchmark dataset built on TyDi, covering eleven typologically diverse languages. It is designed for monolingual retrieval, specifically to evaluate ranking with learned dense representations.

**Dataset:** [`mteb/mrtidy`](https://huggingface.co/datasets/mteb/mrtidy) • **License:** cc-by-sa-3.0 • [Learn more →](https://huggingface.co/datasets/castorini/mr-tydi)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, ben, eng, fin, ind, ... (11) | Encyclopaedic, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{mrtydi,
      author = {Xinyu Zhang and Xueguang Ma and Peng Shi and Jimmy Lin},
      journal = {arXiv:2108.08787},
      title = {{Mr. TyDi}: A Multi-lingual Benchmark for Dense Retrieval},
      year = {2021},
    }
    
    ```
    



#### MultiLongDocRetrieval

Multi Long Doc Retrieval (MLDR) 'is curated by the multilingual articles from Wikipedia, Wudao and mC4 (see Table 7), and NarrativeQA (Kocˇisky ́ et al., 2018; Gu ̈nther et al., 2023), which is only for English.' (Chen et al., 2024).
        It is constructed by sampling lengthy articles from Wikipedia, Wudao and mC4 datasets and randomly choose paragraphs from them. Then we use GPT-3.5 to generate questions based on these paragraphs. The generated question and the sampled article constitute a new text pair to the dataset.

**Dataset:** [`mteb/MultiLongDocRetrieval`](https://huggingface.co/datasets/mteb/MultiLongDocRetrieval) • **License:** mit • [Learn more →](https://arxiv.org/abs/2402.03216)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, cmn, deu, eng, fra, ... (13) | Encyclopaedic, Fiction, Non-fiction, Web, Written | LM-generated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{bge-m3,
      archiveprefix = {arXiv},
      author = {Jianlv Chen and Shitao Xiao and Peitian Zhang and Kun Luo and Defu Lian and Zheng Liu},
      eprint = {2402.03216},
      primaryclass = {cs.CL},
      title = {BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation},
      year = {2024},
    }
    
    ```
    



#### NFCorpus

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval

**Dataset:** [`mteb/nfcorpus`](https://huggingface.co/datasets/mteb/nfcorpus) • **License:** not specified • [Learn more →](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Medical, Written | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{boteva2016,
      author = {Boteva, Vera and Gholipour, Demian and Sokolov, Artem and Riezler, Stefan},
      city = {Padova},
      country = {Italy},
      journal = {Proceedings of the 38th European Conference on Information Retrieval},
      journal-abbrev = {ECIR},
      title = {A Full-Text Learning to Rank Dataset for Medical Information Retrieval},
      url = {http://www.cl.uni-heidelberg.de/~riezler/publications/papers/ECIR2016.pdf},
      year = {2016},
    }
    
    ```
    



#### NFCorpus-Fa

NFCorpus-Fa

**Dataset:** [`MCINext/nfcorpus-fa`](https://huggingface.co/datasets/MCINext/nfcorpus-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/nfcorpus-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Medical | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### NFCorpus-NL

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval. NFCorpus-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-nfcorpus`](https://huggingface.co/datasets/clips/beir-nl-nfcorpus) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-nfcorpus)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Academic, Medical, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### NFCorpus-PL

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval

**Dataset:** [`mteb/NFCorpus-PL`](https://huggingface.co/datasets/mteb/NFCorpus-PL) • **License:** not specified • [Learn more →](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | not specified | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### NFCorpus-VN

A translated dataset from NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/nfcorpus-vn`](https://huggingface.co/datasets/GreenNode/nfcorpus-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Academic, Medical, Written | derived | machine-translated and LM verified |



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
    



#### NLPJournalAbsArticleRetrieval

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding full article with the given abstract. This is the V1 dataset (last updated 2020-06-15).

**Dataset:** [`mteb/NLPJournalAbsArticleRetrieval`](https://huggingface.co/datasets/mteb/NLPJournalAbsArticleRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
      title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
      year = {2024},
    }
    
    ```
    



#### NLPJournalAbsArticleRetrieval.V2

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding full article with the given abstract. This is the V2 dataset (last updated 2025-06-15).

**Dataset:** [`mteb/NLPJournalAbsArticleRetrieval.V2`](https://huggingface.co/datasets/mteb/NLPJournalAbsArticleRetrieval.V2) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
      title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
      year = {2024},
    }
    
    ```
    



#### NLPJournalAbsIntroRetrieval

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding introduction with the given abstract. This is the V1 dataset (last update 2020-06-15).

**Dataset:** [`mteb/NLPJournalAbsIntroRetrieval`](https://huggingface.co/datasets/mteb/NLPJournalAbsIntroRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
      title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
      year = {2024},
    }
    
    ```
    



#### NLPJournalAbsIntroRetrieval.V2

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding introduction with the given abstract. This is the V2 dataset (last update 2025-06-15).

**Dataset:** [`mteb/NLPJournalAbsIntroRetrieval.V2`](https://huggingface.co/datasets/mteb/NLPJournalAbsIntroRetrieval.V2) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
      title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
      year = {2024},
    }
    
    ```
    



#### NLPJournalTitleAbsRetrieval

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding abstract with the given title. This is the V1 dataset (last updated 2020-06-15).

**Dataset:** [`mteb/NLPJournalTitleAbsRetrieval`](https://huggingface.co/datasets/mteb/NLPJournalTitleAbsRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
      title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
      year = {2024},
    }
    
    ```
    



#### NLPJournalTitleAbsRetrieval.V2

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding abstract with the given title. This is the V2 dataset (last updated 2025-06-15).

**Dataset:** [`mteb/NLPJournalTitleAbsRetrieval.V2`](https://huggingface.co/datasets/mteb/NLPJournalTitleAbsRetrieval.V2) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
      title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
      year = {2024},
    }
    
    ```
    



#### NLPJournalTitleIntroRetrieval

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding introduction with the given title. This is the V1 dataset (last updated 2020-06-15).

**Dataset:** [`mteb/NLPJournalTitleIntroRetrieval`](https://huggingface.co/datasets/mteb/NLPJournalTitleIntroRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
      title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
      year = {2024},
    }
    
    ```
    



#### NLPJournalTitleIntroRetrieval.V2

This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding introduction with the given title. This is the V2 dataset (last updated 2025-06-15).

**Dataset:** [`mteb/NLPJournalTitleIntroRetrieval.V2`](https://huggingface.co/datasets/mteb/NLPJournalTitleIntroRetrieval.V2) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | jpn | Academic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{jmteb,
      author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
      howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
      title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
      year = {2024},
    }
    
    ```
    



#### NQ

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval

**Dataset:** [`mteb/nq`](https://huggingface.co/datasets/mteb/nq) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{47761,
      author = {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh
    and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee
    and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le
    and Slav Petrov},
      journal = {Transactions of the Association of Computational
    Linguistics},
      title = {Natural Questions: a Benchmark for Question Answering Research},
      year = {2019},
    }
    
    ```
    



#### NQ-Fa

NQ-Fa

**Dataset:** [`MCINext/nq-fa`](https://huggingface.co/datasets/MCINext/nq-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/nq-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Encyclopaedic | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### NQ-FaHardNegatives

NQ-FaHardNegatives

**Dataset:** [`MCINext/NQ_FA_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/MCINext/NQ_FA_test_top_250_only_w_correct-v2) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/NQ_FA_test_top_250_only_w_correct-v2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Encyclopaedic | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### NQ-NL

NQ-NL is a translation of NQ

**Dataset:** [`clips/beir-nl-nq`](https://huggingface.co/datasets/clips/beir-nl-nq) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-nq)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Encyclopaedic, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### NQ-PL

Natural Questions: A Benchmark for Question Answering Research

**Dataset:** [`mteb/NQ-PL`](https://huggingface.co/datasets/mteb/NQ-PL) • **License:** not specified • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | not specified | not specified | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### NQ-PLHardNegatives

Natural Questions: A Benchmark for Question Answering Research. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/NQ-PLHardNegatives`](https://huggingface.co/datasets/mteb/NQ-PLHardNegatives) • **License:** not specified • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | not specified | not specified | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### NQ-VN

A translated dataset from NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/nq-vn`](https://huggingface.co/datasets/GreenNode/nq-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Encyclopaedic, Written | derived | machine-translated and LM verified |



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
    



#### NQHardNegatives

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/NQ_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/NQ_test_top_250_only_w_correct-v2) • **License:** not specified • [Learn more →](https://ai.google.com/research/NaturalQuestions/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | not specified | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @article{47761,
      author = {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh
    and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee
    and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le
    and Slav Petrov},
      journal = {Transactions of the Association of Computational
    Linguistics},
      title = {Natural Questions: a Benchmark for Question Answering Research},
      year = {2019},
    }
    
    ```
    



#### NanoArguAnaRetrieval

NanoArguAna is a smaller subset of ArguAna, a dataset for argument retrieval in debate contexts.

**Dataset:** [`zeta-alpha-ai/NanoArguAna`](https://huggingface.co/datasets/zeta-alpha-ai/NanoArguAna) • **License:** cc-by-4.0 • [Learn more →](http://argumentation.bplaced.net/arguana/data)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Social, Web, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{wachsmuth2018retrieval,
      author = {Wachsmuth, Henning and Syed, Shahbaz and Stein, Benno},
      booktitle = {ACL},
      title = {Retrieval of the Best Counterargument without Prior Topic Knowledge},
      year = {2018},
    }
    
    ```
    



#### NanoClimateFeverRetrieval

NanoClimateFever is a small version of the BEIR dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change.

**Dataset:** [`zeta-alpha-ai/NanoClimateFEVER`](https://huggingface.co/datasets/zeta-alpha-ai/NanoClimateFEVER) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2012.00614)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, News, Non-fiction | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{diggelmann2021climatefever,
      archiveprefix = {arXiv},
      author = {Thomas Diggelmann and Jordan Boyd-Graber and Jannis Bulian and Massimiliano Ciaramita and Markus Leippold},
      eprint = {2012.00614},
      primaryclass = {cs.CL},
      title = {CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims},
      year = {2021},
    }
    
    ```
    



#### NanoDBPediaRetrieval

NanoDBPediaRetrieval is a small version of the standard test collection for entity search over the DBpedia knowledge base.

**Dataset:** [`zeta-alpha-ai/NanoDBPedia`](https://huggingface.co/datasets/zeta-alpha-ai/NanoDBPedia) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/zeta-alpha-ai/NanoDBPedia)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{lehmann2015dbpedia,
      author = {Lehmann, Jens and et al.},
      journal = {Semantic Web},
      title = {DBpedia: A large-scale, multilingual knowledge base extracted from Wikipedia},
      year = {2015},
    }
    
    ```
    



#### NanoFEVERRetrieval

NanoFEVER is a smaller version of FEVER (Fact Extraction and VERification), which consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from.

**Dataset:** [`zeta-alpha-ai/NanoFEVER`](https://huggingface.co/datasets/zeta-alpha-ai/NanoFEVER) • **License:** cc-by-4.0 • [Learn more →](https://fever.ai/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Encyclopaedic | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{thorne-etal-2018-fever,
      abstract = {In this paper we introduce a new publicly available dataset for verification against textual sources, FEVER: Fact Extraction and VERification. It consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. The claims are classified as Supported, Refuted or NotEnoughInfo by annotators achieving 0.6841 in Fleiss kappa. For the first two classes, the annotators also recorded the sentence(s) forming the necessary evidence for their judgment. To characterize the challenge of the dataset presented, we develop a pipeline approach and compare it to suitably designed oracles. The best accuracy we achieve on labeling a claim accompanied by the correct evidence is 31.87{\%}, while if we ignore the evidence we achieve 50.91{\%}. Thus we believe that FEVER is a challenging testbed that will help stimulate progress on claim verification against textual sources.},
      address = {New Orleans, Louisiana},
      author = {Thorne, James  and
    Vlachos, Andreas  and
    Christodoulopoulos, Christos  and
    Mittal, Arpit},
      booktitle = {Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
      doi = {10.18653/v1/N18-1074},
      editor = {Walker, Marilyn  and
    Ji, Heng  and
    Stent, Amanda},
      month = jun,
      pages = {809--819},
      publisher = {Association for Computational Linguistics},
      title = {{FEVER}: a Large-scale Dataset for Fact Extraction and {VER}ification},
      url = {https://aclanthology.org/N18-1074},
      year = {2018},
    }
    
    ```
    



#### NanoFiQA2018Retrieval

NanoFiQA2018 is a smaller subset of the Financial Opinion Mining and Question Answering dataset.

**Dataset:** [`zeta-alpha-ai/NanoFiQA2018`](https://huggingface.co/datasets/zeta-alpha-ai/NanoFiQA2018) • **License:** cc-by-4.0 • [Learn more →](https://sites.google.com/view/fiqa/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Social | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{thakur2021beir,
      author = {Nandan Thakur and Nils Reimers and Andreas R{\"u}ckl{\'e} and Abhishek Srivastava and Iryna Gurevych},
      booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
      title = {{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
      url = {https://openreview.net/forum?id=wCu6T5xFjeJ},
      year = {2021},
    }
    
    ```
    



#### NanoHotpotQARetrieval

NanoHotpotQARetrieval is a smaller subset of the HotpotQA dataset, which is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.

**Dataset:** [`zeta-alpha-ai/NanoHotpotQA`](https://huggingface.co/datasets/zeta-alpha-ai/NanoHotpotQA) • **License:** cc-by-4.0 • [Learn more →](https://hotpotqa.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Web, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{yang-etal-2018-hotpotqa,
      abstract = {Existing question answering (QA) datasets fail to train QA systems to perform complex reasoning and provide explanations for answers. We introduce HotpotQA, a new dataset with 113k Wikipedia-based question-answer pairs with four key features: (1) the questions require finding and reasoning over multiple supporting documents to answer; (2) the questions are diverse and not constrained to any pre-existing knowledge bases or knowledge schemas; (3) we provide sentence-level supporting facts required for reasoning, allowing QA systems to reason with strong supervision and explain the predictions; (4) we offer a new type of factoid comparison questions to test QA systems{'} ability to extract relevant facts and perform necessary comparison. We show that HotpotQA is challenging for the latest QA systems, and the supporting facts enable models to improve performance and make explainable predictions.},
      address = {Brussels, Belgium},
      author = {Yang, Zhilin  and
    Qi, Peng  and
    Zhang, Saizheng  and
    Bengio, Yoshua  and
    Cohen, William  and
    Salakhutdinov, Ruslan  and
    Manning, Christopher D.},
      booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
      doi = {10.18653/v1/D18-1259},
      editor = {Riloff, Ellen  and
    Chiang, David  and
    Hockenmaier, Julia  and
    Tsujii, Jun{'}ichi},
      month = oct # {-} # nov,
      pages = {2369--2380},
      publisher = {Association for Computational Linguistics},
      title = {{H}otpot{QA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
      url = {https://aclanthology.org/D18-1259},
      year = {2018},
    }
    
    ```
    



#### NanoMSMARCORetrieval

NanoMSMARCORetrieval is a smaller subset of MS MARCO, a collection of datasets focused on deep learning in search.

**Dataset:** [`zeta-alpha-ai/NanoMSMARCO`](https://huggingface.co/datasets/zeta-alpha-ai/NanoMSMARCO) • **License:** cc-by-4.0 • [Learn more →](https://microsoft.github.io/msmarco/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Web | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{DBLP:journals/corr/NguyenRSGTMD16,
      archiveprefix = {arXiv},
      author = {Tri Nguyen and
    Mir Rosenberg and
    Xia Song and
    Jianfeng Gao and
    Saurabh Tiwary and
    Rangan Majumder and
    Li Deng},
      bibsource = {dblp computer science bibliography, https://dblp.org},
      biburl = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
      eprint = {1611.09268},
      journal = {CoRR},
      timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
      title = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
      url = {http://arxiv.org/abs/1611.09268},
      volume = {abs/1611.09268},
      year = {2016},
    }
    
    ```
    



#### NanoNFCorpusRetrieval

NanoNFCorpus is a smaller subset of NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval.

**Dataset:** [`zeta-alpha-ai/NanoNFCorpus`](https://huggingface.co/datasets/zeta-alpha-ai/NanoNFCorpus) • **License:** cc-by-4.0 • [Learn more →](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Medical, Written | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{boteva2016,
      author = {Boteva, Vera and Gholipour, Demian and Sokolov, Artem and Riezler, Stefan},
      city = {Padova},
      country = {Italy},
      journal = {Proceedings of the 38th European Conference on Information Retrieval},
      journal-abbrev = {ECIR},
      title = {A Full-Text Learning to Rank Dataset for Medical Information Retrieval},
      url = {http://www.cl.uni-heidelberg.de/~riezler/publications/papers/ECIR2016.pdf},
      year = {2016},
    }
    
    ```
    



#### NanoNQRetrieval

NanoNQ is a smaller subset of a dataset which contains questions from real users, and it requires QA systems to read and comprehend an entire Wikipedia article that may or may not contain the answer to the question.

**Dataset:** [`zeta-alpha-ai/NanoNQ`](https://huggingface.co/datasets/zeta-alpha-ai/NanoNQ) • **License:** cc-by-4.0 • [Learn more →](https://ai.google.com/research/NaturalQuestions)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Web | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{47761,
      author = {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh
    and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee
    and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le
    and Slav Petrov},
      journal = {Transactions of the Association of Computational
    Linguistics},
      title = {Natural Questions: a Benchmark for Question Answering Research},
      year = {2019},
    }
    
    ```
    



#### NanoQuoraRetrieval

NanoQuoraRetrieval is a smaller subset of the QuoraRetrieval dataset, which is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.

**Dataset:** [`zeta-alpha-ai/NanoQuoraRetrieval`](https://huggingface.co/datasets/zeta-alpha-ai/NanoQuoraRetrieval) • **License:** cc-by-4.0 • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Social | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{quora-question-pairs,
      author = {DataCanary, hilfialkaff, Lili Jiang, Meg Risdal, Nikhil Dandekar, tomtung},
      publisher = {Kaggle},
      title = {Quora Question Pairs},
      url = {https://kaggle.com/competitions/quora-question-pairs},
      year = {2017},
    }
    
    ```
    



#### NanoSCIDOCSRetrieval

NanoFiQA2018 is a smaller subset of SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation.

**Dataset:** [`zeta-alpha-ai/NanoSCIDOCS`](https://huggingface.co/datasets/zeta-alpha-ai/NanoSCIDOCS) • **License:** cc-by-4.0 • [Learn more →](https://allenai.org/data/scidocs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Non-fiction, Written | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{specter2020cohan,
      author = {Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
      booktitle = {ACL},
      title = {SPECTER: Document-level Representation Learning using Citation-informed Transformers},
      year = {2020},
    }
    
    ```
    



#### NanoSciFactRetrieval

NanoSciFact is a smaller subset of SciFact, which verifies scientific claims using evidence from the research literature containing scientific paper abstracts.

**Dataset:** [`zeta-alpha-ai/NanoSciFact`](https://huggingface.co/datasets/zeta-alpha-ai/NanoSciFact) • **License:** cc-by-4.0 • [Learn more →](https://github.com/allenai/scifact)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Medical, Written | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{specter2020cohan,
      author = {Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
      booktitle = {ACL},
      title = {SPECTER: Document-level Representation Learning using Citation-informed Transformers},
      year = {2020},
    }
    
    ```
    



#### NanoTouche2020Retrieval

NanoTouche2020 is a smaller subset of Touché Task 1: Argument Retrieval for Controversial Questions.

**Dataset:** [`zeta-alpha-ai/NanoTouche2020`](https://huggingface.co/datasets/zeta-alpha-ai/NanoTouche2020) • **License:** cc-by-4.0 • [Learn more →](https://webis.de/events/touche-20/shared-task-1.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @dataset{potthast_2022_6862281,
      author = {Potthast, Martin and
    Gienapp, Lukas and
    Wachsmuth, Henning and
    Hagen, Matthias and
    Fröbe, Maik and
    Bondarenko, Alexander and
    Ajjour, Yamen and
    Stein, Benno},
      doi = {10.5281/zenodo.6862281},
      month = jul,
      publisher = {Zenodo},
      title = {{Touché20-Argument-Retrieval-for-Controversial-
    Questions}},
      url = {https://doi.org/10.5281/zenodo.6862281},
      year = {2022},
    }
    
    ```
    



#### NarrativeQARetrieval

NarrativeQA is a dataset for the task of question answering on long narratives. It consists of realistic QA instances collected from literature (fiction and non-fiction) and movie scripts. 

**Dataset:** [`deepmind/narrativeqa`](https://huggingface.co/datasets/deepmind/narrativeqa) • **License:** not specified • [Learn more →](https://metatext.io/datasets/narrativeqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | not specified | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{kočiský2017narrativeqa,
      archiveprefix = {arXiv},
      author = {Tomáš Kočiský and Jonathan Schwarz and Phil Blunsom and Chris Dyer and Karl Moritz Hermann and Gábor Melis and Edward Grefenstette},
      eprint = {1712.07040},
      primaryclass = {cs.CL},
      title = {The NarrativeQA Reading Comprehension Challenge},
      year = {2017},
    }
    
    ```
    



#### NeuCLIR2022Retrieval

The task involves identifying and retrieving the documents that are relevant to the queries.

**Dataset:** [`mteb/NeuCLIR2022Retrieval`](https://huggingface.co/datasets/mteb/NeuCLIR2022Retrieval) • **License:** odc-by • [Learn more →](https://neuclir.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | fas, rus, zho | News, Written | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{lawrie2023overview,
      author = {Lawrie, Dawn and MacAvaney, Sean and Mayfield, James and McNamee, Paul and Oard, Douglas W and Soldaini, Luca and Yang, Eugene},
      journal = {arXiv preprint arXiv:2304.12367},
      title = {Overview of the TREC 2022 NeuCLIR track},
      year = {2023},
    }
    
    ```
    



#### NeuCLIR2022RetrievalHardNegatives

The task involves identifying and retrieving the documents that are relevant to the queries. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/NeuCLIR2022RetrievalHardNegatives`](https://huggingface.co/datasets/mteb/NeuCLIR2022RetrievalHardNegatives) • **License:** odc-by • [Learn more →](https://neuclir.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | fas, rus, zho | News, Written | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{lawrie2023overview,
      author = {Lawrie, Dawn and MacAvaney, Sean and Mayfield, James and McNamee, Paul and Oard, Douglas W and Soldaini, Luca and Yang, Eugene},
      journal = {arXiv preprint arXiv:2304.12367},
      title = {Overview of the TREC 2022 NeuCLIR track},
      year = {2023},
    }
    
    ```
    



#### NeuCLIR2023Retrieval

The task involves identifying and retrieving the documents that are relevant to the queries.

**Dataset:** [`mteb/NeuCLIR2022Retrieval`](https://huggingface.co/datasets/mteb/NeuCLIR2022Retrieval) • **License:** odc-by • [Learn more →](https://neuclir.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | fas, rus, zho | News, Written | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{lawrie2024overview,
      archiveprefix = {arXiv},
      author = {Dawn Lawrie and Sean MacAvaney and James Mayfield and Paul McNamee and Douglas W. Oard and Luca Soldaini and Eugene Yang},
      eprint = {2404.08071},
      primaryclass = {cs.IR},
      title = {Overview of the TREC 2023 NeuCLIR Track},
      year = {2024},
    }
    
    ```
    



#### NeuCLIR2023RetrievalHardNegatives

The task involves identifying and retrieving the documents that are relevant to the queries. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/NeuCLIR2023RetrievalHardNegatives`](https://huggingface.co/datasets/mteb/NeuCLIR2023RetrievalHardNegatives) • **License:** odc-by • [Learn more →](https://neuclir.github.io/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | fas, rus, zho | News, Written | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{lawrie2024overview,
      archiveprefix = {arXiv},
      author = {Dawn Lawrie and Sean MacAvaney and James Mayfield and Paul McNamee and Douglas W. Oard and Luca Soldaini and Eugene Yang},
      eprint = {2404.08071},
      primaryclass = {cs.IR},
      title = {Overview of the TREC 2023 NeuCLIR Track},
      year = {2024},
    }
    
    ```
    



#### NorQuadRetrieval

Human-created question for Norwegian wikipedia passages.

**Dataset:** [`mteb/norquad_retrieval`](https://huggingface.co/datasets/mteb/norquad_retrieval) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.17/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nob | Encyclopaedic, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{ivanova-etal-2023-norquad,
      abstract = {In this paper we present NorQuAD: the first Norwegian question answering dataset for machine reading comprehension. The dataset consists of 4,752 manually created question-answer pairs. We here detail the data collection procedure and present statistics of the dataset. We also benchmark several multilingual and Norwegian monolingual language models on the dataset and compare them against human performance. The dataset will be made freely available.},
      address = {T{\'o}rshavn, Faroe Islands},
      author = {Ivanova, Sardana  and
    Andreassen, Fredrik  and
    Jentoft, Matias  and
    Wold, Sondre  and
    {\O}vrelid, Lilja},
      booktitle = {Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)},
      editor = {Alum{\"a}e, Tanel  and
    Fishel, Mark},
      month = may,
      pages = {159--168},
      publisher = {University of Tartu Library},
      title = {{N}or{Q}u{AD}: {N}orwegian Question Answering Dataset},
      url = {https://aclanthology.org/2023.nodalida-1.17},
      year = {2023},
    }
    
    ```
    



#### PIQA

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on PIQA.

**Dataset:** [`mteb/PIQA`](https://huggingface.co/datasets/mteb/PIQA) • **License:** afl-3.0 • [Learn more →](https://arxiv.org/abs/1911.11641)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{bisk2020piqa,
      author = {Bisk, Yonatan and Zellers, Rowan and Gao, Jianfeng and Choi, Yejin and others},
      booktitle = {Proceedings of the AAAI conference on artificial intelligence},
      number = {05},
      pages = {7432--7439},
      title = {Piqa: Reasoning about physical commonsense in natural language},
      volume = {34},
      year = {2020},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### PUGGRetrieval

Information Retrieval PUGG dataset for the Polish language.

**Dataset:** [`clarin-pl/PUGG_IR`](https://huggingface.co/datasets/clarin-pl/PUGG_IR) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2024.findings-acl.652/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Web | human-annotated | multiple |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{sawczyn-etal-2024-developing,
      address = {Bangkok, Thailand},
      author = {Sawczyn, Albert  and
    Viarenich, Katsiaryna  and
    Wojtasik, Konrad  and
    Domoga{\l}a, Aleksandra  and
    Oleksy, Marcin  and
    Piasecki, Maciej  and
    Kajdanowicz, Tomasz},
      booktitle = {Findings of the Association for Computational Linguistics: ACL 2024},
      doi = {10.18653/v1/2024.findings-acl.652},
      editor = {Ku, Lun-Wei  and
    Martins, Andre  and
    Srikumar, Vivek},
      month = aug,
      pages = {10978--10996},
      publisher = {Association for Computational Linguistics},
      title = {Developing {PUGG} for {P}olish: A Modern Approach to {KBQA}, {MRC}, and {IR} Dataset Construction},
      url = {https://aclanthology.org/2024.findings-acl.652/},
      year = {2024},
    }
    
    ```
    



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



??? quote "Citation"

    
    ```bibtex
    
    @misc{xing_han_lu_2024,
      author = { {Xing Han Lu} },
      doi = { 10.57967/hf/2247 },
      publisher = { Hugging Face },
      title = { publichealth-qa (Revision 3b67b6b) },
      url = { https://huggingface.co/datasets/xhluca/publichealth-qa },
      year = {2024},
    }
    
    ```
    



#### Quail

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on Quail.

**Dataset:** [`mteb/Quail`](https://huggingface.co/datasets/mteb/Quail) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://text-machine.cs.uml.edu/lab2/projects/quail/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{rogers2020getting,
      author = {Rogers, Anna and Kovaleva, Olga and Downey, Matthew and Rumshisky, Anna},
      booktitle = {Proceedings of the AAAI conference on artificial intelligence},
      number = {05},
      pages = {8722--8731},
      title = {Getting closer to AI complete question answering: A set of prerequisite real tasks},
      volume = {34},
      year = {2020},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### Quora-NL

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions. QuoraRetrieval-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-quora`](https://huggingface.co/datasets/clips/beir-nl-quora) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-quora)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### Quora-PL

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.

**Dataset:** [`mteb/Quora-PL`](https://huggingface.co/datasets/mteb/Quora-PL) • **License:** not specified • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | not specified | not specified | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### Quora-PLHardNegatives

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/Quora-PLHardNegatives`](https://huggingface.co/datasets/mteb/Quora-PLHardNegatives) • **License:** not specified • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | not specified | not specified | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



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
| text to text (t2t) | ndcg_at_10 | vie | Blog, Web, Written | derived | machine-translated and LM verified |



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
    



#### QuoraRetrieval

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.

**Dataset:** [`mteb/quora`](https://huggingface.co/datasets/mteb/quora) • **License:** not specified • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Blog, Web, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{quora-question-pairs,
      author = {DataCanary, hilfialkaff, Lili Jiang, Meg Risdal, Nikhil Dandekar, tomtung},
      publisher = {Kaggle},
      title = {Quora Question Pairs},
      url = {https://kaggle.com/competitions/quora-question-pairs},
      year = {2017},
    }
    
    ```
    



#### QuoraRetrieval-Fa

QuoraRetrieval-Fa

**Dataset:** [`MCINext/quora-fa`](https://huggingface.co/datasets/MCINext/quora-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/quora-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### QuoraRetrieval-Fa.v2

QuoraRetrieval-Fa.v2

**Dataset:** [`MCINext/quora-fa-v2`](https://huggingface.co/datasets/MCINext/quora-fa-v2) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/quora-fa-v2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Web | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### QuoraRetrievalHardNegatives

QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/QuoraRetrieval_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/QuoraRetrieval_test_top_250_only_w_correct-v2) • **License:** not specified • [Learn more →](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | not specified | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{quora-question-pairs,
      author = {DataCanary, hilfialkaff, Lili Jiang, Meg Risdal, Nikhil Dandekar, tomtung},
      publisher = {Kaggle},
      title = {Quora Question Pairs},
      url = {https://kaggle.com/competitions/quora-question-pairs},
      year = {2017},
    }
    
    ```
    



#### R2MEDBioinformaticsRetrieval

Bioinformatics retrieval dataset.

**Dataset:** [`R2MED/Bioinformatics`](https://huggingface.co/datasets/R2MED/Bioinformatics) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/Bioinformatics)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{li2025r2med,
      author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
      journal = {arXiv preprint arXiv:2505.14558},
      title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
      year = {2025},
    }
    
    ```
    



#### R2MEDBiologyRetrieval

Biology retrieval dataset.

**Dataset:** [`R2MED/Biology`](https://huggingface.co/datasets/R2MED/Biology) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/Biology)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{li2025r2med,
      author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
      journal = {arXiv preprint arXiv:2505.14558},
      title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
      year = {2025},
    }
    
    ```
    



#### R2MEDIIYiClinicalRetrieval

IIYi-Clinical retrieval dataset.

**Dataset:** [`R2MED/IIYi-Clinical`](https://huggingface.co/datasets/R2MED/IIYi-Clinical) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/IIYi-Clinical)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{li2025r2med,
      author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
      journal = {arXiv preprint arXiv:2505.14558},
      title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
      year = {2025},
    }
    
    ```
    



#### R2MEDMedQADiagRetrieval

MedQA-Diag retrieval dataset.

**Dataset:** [`R2MED/MedQA-Diag`](https://huggingface.co/datasets/R2MED/MedQA-Diag) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/MedQA-Diag)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{li2025r2med,
      author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
      journal = {arXiv preprint arXiv:2505.14558},
      title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
      year = {2025},
    }
    
    ```
    



#### R2MEDMedXpertQAExamRetrieval

MedXpertQA-Exam retrieval dataset.

**Dataset:** [`R2MED/MedXpertQA-Exam`](https://huggingface.co/datasets/R2MED/MedXpertQA-Exam) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/MedXpertQA-Exam)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{li2025r2med,
      author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
      journal = {arXiv preprint arXiv:2505.14558},
      title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
      year = {2025},
    }
    
    ```
    



#### R2MEDMedicalSciencesRetrieval

Medical-Sciences retrieval dataset.

**Dataset:** [`R2MED/Medical-Sciences`](https://huggingface.co/datasets/R2MED/Medical-Sciences) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/Medical-Sciences)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{li2025r2med,
      author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
      journal = {arXiv preprint arXiv:2505.14558},
      title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
      year = {2025},
    }
    
    ```
    



#### R2MEDPMCClinicalRetrieval

PMC-Clinical retrieval dataset.

**Dataset:** [`R2MED/PMC-Clinical`](https://huggingface.co/datasets/R2MED/PMC-Clinical) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/PMC-Clinical)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{li2025r2med,
      author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
      journal = {arXiv preprint arXiv:2505.14558},
      title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
      year = {2025},
    }
    
    ```
    



#### R2MEDPMCTreatmentRetrieval

PMC-Treatment retrieval dataset.

**Dataset:** [`R2MED/PMC-Treatment`](https://huggingface.co/datasets/R2MED/PMC-Treatment) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/R2MED/PMC-Treatment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Medical | LM-generated and reviewed | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{li2025r2med,
      author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
      journal = {arXiv preprint arXiv:2505.14558},
      title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
      year = {2025},
    }
    
    ```
    



#### RARbCode

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on RAR-b code-pooled dataset.

**Dataset:** [`mteb/RARbCode`](https://huggingface.co/datasets/mteb/RARbCode) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2404.06347)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Programming, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{husain2019codesearchnet,
      author = {Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
      journal = {arXiv preprint arXiv:1909.09436},
      title = {Codesearchnet challenge: Evaluating the state of semantic code search},
      year = {2019},
    }
    
    @article{muennighoff2023octopack,
      author = {Muennighoff, Niklas and Liu, Qian and Zebaze, Armel and Zheng, Qinkai and Hui, Binyuan and Zhuo, Terry Yue and Singh, Swayam and Tang, Xiangru and Von Werra, Leandro and Longpre, Shayne},
      journal = {arXiv preprint arXiv:2308.07124},
      title = {Octopack: Instruction tuning code large language models},
      year = {2023},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### RARbMath

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on RAR-b math-pooled dataset.

**Dataset:** [`mteb/RARbMath`](https://huggingface.co/datasets/mteb/RARbMath) • **License:** mit • [Learn more →](https://arxiv.org/abs/2404.06347)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{cobbe2021training,
      author = {Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and others},
      journal = {arXiv preprint arXiv:2110.14168},
      title = {Training verifiers to solve math word problems},
      year = {2021},
    }
    
    @article{hendrycks2021measuring,
      author = {Hendrycks, Dan and Burns, Collin and Kadavath, Saurav and Arora, Akul and Basart, Steven and Tang, Eric and Song, Dawn and Steinhardt, Jacob},
      journal = {arXiv preprint arXiv:2103.03874},
      title = {Measuring mathematical problem solving with the math dataset},
      year = {2021},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    @article{yu2023metamath,
      author = {Yu, Longhui and Jiang, Weisen and Shi, Han and Yu, Jincheng and Liu, Zhengying and Zhang, Yu and Kwok, James T and Li, Zhenguo and Weller, Adrian and Liu, Weiyang},
      journal = {arXiv preprint arXiv:2309.12284},
      title = {Metamath: Bootstrap your own mathematical questions for large language models},
      year = {2023},
    }
    
    ```
    



#### RiaNewsRetrieval

News article retrieval by headline. Based on Rossiya Segodnya dataset.

**Dataset:** [`ai-forever/ria-news-retrieval`](https://huggingface.co/datasets/ai-forever/ria-news-retrieval) • **License:** cc-by-nc-nd-4.0 • [Learn more →](https://arxiv.org/abs/1901.07786)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | rus | News, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{gavrilov2018self,
      author = {Gavrilov, Daniil and  Kalaidin, Pavel and  Malykh, Valentin},
      booktitle = {Proceedings of the 41st European Conference on Information Retrieval},
      title = {Self-Attentive Model for Headline Generation},
      year = {2019},
    }
    
    ```
    



#### RiaNewsRetrievalHardNegatives

News article retrieval by headline. Based on Rossiya Segodnya dataset. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/RiaNewsRetrieval_test_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/RiaNewsRetrieval_test_top_250_only_w_correct-v2) • **License:** cc-by-nc-nd-4.0 • [Learn more →](https://arxiv.org/abs/1901.07786)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | rus | News, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{gavrilov2018self,
      author = {Gavrilov, Daniil and  Kalaidin, Pavel and  Malykh, Valentin},
      booktitle = {Proceedings of the 41st European Conference on Information Retrieval},
      title = {Self-Attentive Model for Headline Generation},
      year = {2019},
    }
    
    ```
    



#### RuBQRetrieval

Paragraph retrieval based on RuBQ 2.0. Retrieve paragraphs from Wikipedia that answer the question.

**Dataset:** [`ai-forever/rubq-retrieval`](https://huggingface.co/datasets/ai-forever/rubq-retrieval) • **License:** cc-by-sa-4.0 • [Learn more →](https://openreview.net/pdf?id=P5UQFFoQ4PJ)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | rus | Encyclopaedic, Written | human-annotated | created |



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
    



#### RuSciBenchCiteRetrieval

This task is focused on Direct Citation Prediction for scientific papers from eLibrary,
        Russia's largest electronic library of scientific publications. Given a query paper (title and abstract),
        the goal is to retrieve papers that are directly cited by it from a larger corpus of papers.
        The dataset for this task consists of 3,000 query papers, 15,000 relevant (cited) papers,
        and 75,000 irrelevant papers. The task is available for both Russian and English scientific texts.

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_cite_retrieval`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_cite_retrieval) • **License:** mit • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng, rus | Academic, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{vatolin2024ruscibench,
      author = {Vatolin, A. and Gerasimenko, N. and Ianina, A. and Vorontsov, K.},
      doi = {10.1134/S1064562424602191},
      issn = {1531-8362},
      journal = {Doklady Mathematics},
      month = {12},
      number = {1},
      pages = {S251--S260},
      title = {RuSciBench: Open Benchmark for Russian and English Scientific Document Representations},
      url = {https://doi.org/10.1134/S1064562424602191},
      volume = {110},
      year = {2024},
    }
    
    ```
    



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
| text to text (t2t) | ndcg_at_10 | eng, rus | Academic, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{vatolin2024ruscibench,
      author = {Vatolin, A. and Gerasimenko, N. and Ianina, A. and Vorontsov, K.},
      doi = {10.1134/S1064562424602191},
      issn = {1531-8362},
      journal = {Doklady Mathematics},
      month = {12},
      number = {1},
      pages = {S251--S260},
      title = {RuSciBench: Open Benchmark for Russian and English Scientific Document Representations},
      url = {https://doi.org/10.1134/S1064562424602191},
      volume = {110},
      year = {2024},
    }
    
    ```
    



#### SCIDOCS

SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation.

**Dataset:** [`mteb/scidocs`](https://huggingface.co/datasets/mteb/scidocs) • **License:** cc-by-sa-4.0 • [Learn more →](https://allenai.org/data/scidocs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Non-fiction, Written | not specified | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{specter2020cohan,
      author = {Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
      booktitle = {ACL},
      title = {SPECTER: Document-level Representation Learning using Citation-informed Transformers},
      year = {2020},
    }
    
    ```
    



#### SCIDOCS-Fa

SCIDOCS-Fa

**Dataset:** [`MCINext/scidocs-fa`](https://huggingface.co/datasets/MCINext/scidocs-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/scidocs-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Academic | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### SCIDOCS-Fa.v2

SCIDOCS-Fa.v2

**Dataset:** [`MCINext/scidocs-fa-v2`](https://huggingface.co/datasets/MCINext/scidocs-fa-v2) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/scidocs-fa-v2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Academic | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### SCIDOCS-NL

SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation. SciDocs-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-scidocs`](https://huggingface.co/datasets/clips/beir-nl-scidocs) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-scidocs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Academic, Non-fiction, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### SCIDOCS-PL

SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation.

**Dataset:** [`mteb/SCIDOCS-PL`](https://huggingface.co/datasets/mteb/SCIDOCS-PL) • **License:** not specified • [Learn more →](https://allenai.org/data/scidocs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | not specified | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



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
| text to text (t2t) | ndcg_at_10 | vie | Academic, Non-fiction, Written | derived | machine-translated and LM verified |



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
    



#### SIQA

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on SIQA.

**Dataset:** [`mteb/SIQA`](https://huggingface.co/datasets/mteb/SIQA) • **License:** not specified • [Learn more →](https://leaderboard.allenai.org/socialiqa/submissions/get-started)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{sap2019socialiqa,
      author = {Sap, Maarten and Rashkin, Hannah and Chen, Derek and LeBras, Ronan and Choi, Yejin},
      journal = {arXiv preprint arXiv:1904.09728},
      title = {Socialiqa: Commonsense reasoning about social interactions},
      year = {2019},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



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



??? quote "Citation"

    
    ```bibtex
    
    @mastersthesis{navjord2023beyond,
      author = {Navjord, J{\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
      school = {Norwegian University of Life Sciences, {\AA}s},
      title = {Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
      year = {2023},
    }
    
    ```
    



#### SadeemQuestionRetrieval

SadeemQuestion: A Benchmark Data Set for Community Question-Retrieval Research

**Dataset:** [`sadeem-ai/sadeem-ar-eval-retrieval-questions`](https://huggingface.co/datasets/sadeem-ai/sadeem-ar-eval-retrieval-questions) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/sadeem-ai/sadeem-ar-eval-retrieval-questions)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara | Written, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{sadeem-2024-ar-retrieval-questions,
      author = {abubakr.soliman@sadeem.app},
      title = {SadeemQuestionRetrieval: A New Benchmark for Arabic questions-based Articles Searching.},
    }
    
    ```
    



#### SciFact

SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.

**Dataset:** [`mteb/scifact`](https://huggingface.co/datasets/mteb/scifact) • **License:** not specified • [Learn more →](https://github.com/allenai/scifact)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Medical, Written | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{specter2020cohan,
      author = {Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
      booktitle = {ACL},
      title = {SPECTER: Document-level Representation Learning using Citation-informed Transformers},
      year = {2020},
    }
    
    ```
    



#### SciFact-Fa

SciFact-Fa

**Dataset:** [`MCINext/scifact-fa`](https://huggingface.co/datasets/MCINext/scifact-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/scifact-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Academic | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### SciFact-Fa.v2

SciFact-Fa.v2

**Dataset:** [`MCINext/scifact-fa-v2`](https://huggingface.co/datasets/MCINext/scifact-fa-v2) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/scifact-fa-v2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Academic | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### SciFact-NL

SciFactNL verifies scientific claims in Dutch using evidence from the research literature containing scientific paper abstracts.

**Dataset:** [`clips/beir-nl-scifact`](https://huggingface.co/datasets/clips/beir-nl-scifact) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-scifact)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Academic, Medical, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### SciFact-PL

SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.

**Dataset:** [`mteb/SciFact-PL`](https://huggingface.co/datasets/mteb/SciFact-PL) • **License:** not specified • [Learn more →](https://github.com/allenai/scifact)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic, Medical, Written | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### SciFact-VN

A translated dataset from SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/scifact-vn`](https://huggingface.co/datasets/GreenNode/scifact-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/allenai/scifact)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Academic, Medical, Written | derived | machine-translated and LM verified |



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
    



#### SlovakSumRetrieval


            SlovakSum, a Slovak news summarization dataset consisting of over 200 thousand
            news articles with titles and short abstracts obtained from multiple Slovak newspapers.

            Originally intended as a summarization task, but since no human annotations were provided
            here reformulated to a retrieval task.
        

**Dataset:** [`NaiveNeuron/slovaksum`](https://huggingface.co/datasets/NaiveNeuron/slovaksum) • **License:** openrail • [Learn more →](https://huggingface.co/datasets/NaiveNeuron/slovaksum)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | slk | News, Social, Web, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{OndrejowaSlovakSum24,
      author = {Ondrejová, Viktória and Šuppa, Marek},
      booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation},
      date = {2024},
      title = {SlovakSum: A Large Scale Slovak Summarization Dataset},
    }
    
    ```
    



#### SpanishPassageRetrievalS2P

Test collection for passage retrieval from health-related Web resources in Spanish.

**Dataset:** [`mteb/SpanishPassageRetrievalS2P`](https://huggingface.co/datasets/mteb/SpanishPassageRetrievalS2P) • **License:** not specified • [Learn more →](https://mklab.iti.gr/results/spanish-passage-retrieval-dataset/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | spa | not specified | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{10.1007/978-3-030-15719-7_19,
      abstract = {This paper describes a new test collection for passage retrieval from health-related Web resources in Spanish. The test collection contains 10,037 health-related documents in Spanish, 37 topics representing complex information needs formulated in a total of 167 natural language questions, and manual relevance assessments of text passages, pooled from multiple systems. This test collection is the first to combine search in a language beyond English, passage retrieval, and health-related resources and topics targeting the general public.},
      address = {Cham},
      author = {Kamateri, Eleni
    and Tsikrika, Theodora
    and Symeonidis, Spyridon
    and Vrochidis, Stefanos
    and Minker, Wolfgang
    and Kompatsiaris, Yiannis},
      booktitle = {Advances in Information Retrieval},
      editor = {Azzopardi, Leif
    and Stein, Benno
    and Fuhr, Norbert
    and Mayr, Philipp
    and Hauff, Claudia
    and Hiemstra, Djoerd},
      isbn = {978-3-030-15719-7},
      pages = {148--154},
      publisher = {Springer International Publishing},
      title = {A Test Collection for Passage Retrieval Evaluation of Spanish Health-Related Resources},
      year = {2019},
    }
    
    ```
    



#### SpanishPassageRetrievalS2S

Test collection for passage retrieval from health-related Web resources in Spanish.

**Dataset:** [`mteb/SpanishPassageRetrievalS2S`](https://huggingface.co/datasets/mteb/SpanishPassageRetrievalS2S) • **License:** not specified • [Learn more →](https://mklab.iti.gr/results/spanish-passage-retrieval-dataset/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | spa | not specified | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{10.1007/978-3-030-15719-7_19,
      abstract = {This paper describes a new test collection for passage retrieval from health-related Web resources in Spanish. The test collection contains 10,037 health-related documents in Spanish, 37 topics representing complex information needs formulated in a total of 167 natural language questions, and manual relevance assessments of text passages, pooled from multiple systems. This test collection is the first to combine search in a language beyond English, passage retrieval, and health-related resources and topics targeting the general public.},
      address = {Cham},
      author = {Kamateri, Eleni
    and Tsikrika, Theodora
    and Symeonidis, Spyridon
    and Vrochidis, Stefanos
    and Minker, Wolfgang
    and Kompatsiaris, Yiannis},
      booktitle = {Advances in Information Retrieval},
      editor = {Azzopardi, Leif
    and Stein, Benno
    and Fuhr, Norbert
    and Mayr, Philipp
    and Hauff, Claudia
    and Hiemstra, Djoerd},
      isbn = {978-3-030-15719-7},
      pages = {148--154},
      publisher = {Springer International Publishing},
      title = {A Test Collection for Passage Retrieval Evaluation of Spanish Health-Related Resources},
      year = {2019},
    }
    
    ```
    



#### SpartQA

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on SpartQA.

**Dataset:** [`mteb/SpartQA`](https://huggingface.co/datasets/mteb/SpartQA) • **License:** mit • [Learn more →](https://github.com/HLR/SpartQA_generation)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{mirzaee2021spartqa,
      author = {Mirzaee, Roshanak and Faghihi, Hossein Rajaby and Ning, Qiang and Kordjmashidi, Parisa},
      journal = {arXiv preprint arXiv:2104.05832},
      title = {Spartqa:: A textual question answering benchmark for spatial reasoning},
      year = {2021},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### StackOverflowQA

The dataset is a collection of natural language queries and their corresponding response which may include some text mixed with code snippets. The task is to retrieve the most relevant response for a given query.

**Dataset:** [`CoIR-Retrieval/stackoverflow-qa`](https://huggingface.co/datasets/CoIR-Retrieval/stackoverflow-qa) • **License:** mit • [Learn more →](https://arxiv.org/abs/2407.02883)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Programming, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{li2024coircomprehensivebenchmarkcode,
      archiveprefix = {arXiv},
      author = {Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Yichun Yin and Hao Zhang and Yong Liu and Yasheng Wang and Ruiming Tang},
      eprint = {2407.02883},
      primaryclass = {cs.IR},
      title = {CoIR: A Comprehensive Benchmark for Code Information Retrieval Models},
      url = {https://arxiv.org/abs/2407.02883},
      year = {2024},
    }
    
    ```
    



#### StatcanDialogueDatasetRetrieval

A Dataset for Retrieving Data Tables through Conversations with Genuine Intents, available in English and French.

**Dataset:** [`McGill-NLP/statcan-dialogue-dataset-retrieval`](https://huggingface.co/datasets/McGill-NLP/statcan-dialogue-dataset-retrieval) • **License:** https://huggingface.co/datasets/McGill-NLP/statcan-dialogue-dataset-retrieval/blob/main/LICENSE.md • [Learn more →](https://mcgill-nlp.github.io/statcan-dialogue-dataset/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | recall_at_10 | eng, fra | Government, Web, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{lu-etal-2023-statcan,
      address = {Dubrovnik, Croatia},
      author = {Lu, Xing Han  and
    Reddy, Siva  and
    de Vries, Harm},
      booktitle = {Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics},
      month = may,
      pages = {2799--2829},
      publisher = {Association for Computational Linguistics},
      title = {The {S}tat{C}an Dialogue Dataset: Retrieving Data Tables through Conversations with Genuine Intents},
      url = {https://arxiv.org/abs/2304.01412},
      year = {2023},
    }
    
    ```
    



#### SweFaqRetrieval

A Swedish QA dataset derived from FAQ

**Dataset:** [`mteb/SweFaqRetrieval`](https://huggingface.co/datasets/mteb/SweFaqRetrieval) • **License:** cc-by-sa-4.0 • [Learn more →](https://spraakbanken.gu.se/en/resources/superlim)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | swe | Government, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{berdivcevskis2023superlim,
      author = {Berdi{\v{c}}evskis, Aleksandrs and Bouma, Gerlof and Kurtz, Robin and Morger, Felix and {\"O}hman, Joey and Adesam, Yvonne and Borin, Lars and Dann{\'e}lls, Dana and Forsberg, Markus and Isbister, Tim and others},
      booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
      pages = {8137--8153},
      title = {Superlim: A Swedish language understanding evaluation benchmark},
      year = {2023},
    }
    
    ```
    



#### SwednRetrieval

The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail dataset both regarding textual structure

**Dataset:** [`mteb/SwednRetrieval`](https://huggingface.co/datasets/mteb/SwednRetrieval) • **License:** cc-by-sa-4.0 • [Learn more →](https://spraakbanken.gu.se/en/resources/swedn)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | swe | News, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{monsen2021method,
      author = {Monsen, Julius and J{\"o}nsson, Arne},
      booktitle = {Proceedings of CLARIN Annual Conference},
      title = {A method for building non-english corpora for abstractive text summarization},
      year = {2021},
    }
    
    ```
    



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
    



#### SyntheticText2SQL

The dataset is a collection of natural language queries and their corresponding sql snippets. The task is to retrieve the most relevant code snippet for a given query.

**Dataset:** [`CoIR-Retrieval/synthetic-text2sql`](https://huggingface.co/datasets/CoIR-Retrieval/synthetic-text2sql) • **License:** mit • [Learn more →](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng, sql | Programming, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @software{gretel-synthetic-text-to-sql-2024,
      author = {Meyer, Yev and Emadi, Marjan and Nathawani, Dhruv and Ramaswamy, Lipika and Boyd, Kendrick and Van Segbroeck, Maarten and Grossman, Matthew and Mlocek, Piotr and Newberry, Drew},
      month = {April},
      title = {{Synthetic-Text-To-SQL}: A synthetic dataset for training language models to generate SQL queries from natural language prompts},
      url = {https://huggingface.co/datasets/gretelai/synthetic-text-to-sql},
      year = {2024},
    }
    
    ```
    



#### T2Retrieval

T2Ranking: A large-scale Chinese Benchmark for Passage Ranking

**Dataset:** [`mteb/T2Retrieval`](https://huggingface.co/datasets/mteb/T2Retrieval) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2304.03679)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | Academic, Financial, Government, Medical, Non-fiction | human-annotated | not specified |



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
    



#### TRECCOVID

TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic.

**Dataset:** [`mteb/trec-covid`](https://huggingface.co/datasets/mteb/trec-covid) • **License:** not specified • [Learn more →](https://ir.nist.gov/covidSubmit/index.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic, Medical, Written | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{roberts2021searching,
      archiveprefix = {arXiv},
      author = {Kirk Roberts and Tasmeer Alam and Steven Bedrick and Dina Demner-Fushman and Kyle Lo and Ian Soboroff and Ellen Voorhees and Lucy Lu Wang and William R Hersh},
      eprint = {2104.09632},
      primaryclass = {cs.IR},
      title = {Searching for Scientific Evidence in a Pandemic: An Overview of TREC-COVID},
      year = {2021},
    }
    
    ```
    



#### TRECCOVID-Fa

TRECCOVID-Fa

**Dataset:** [`MCINext/trec-covid-fa`](https://huggingface.co/datasets/MCINext/trec-covid-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/trec-covid-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Medical | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### TRECCOVID-Fa.v2

TRECCOVID-Fa.v2

**Dataset:** [`MCINext/trec-covid-fa-v2`](https://huggingface.co/datasets/MCINext/trec-covid-fa-v2) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/trec-covid-fa-v2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Medical | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### TRECCOVID-NL

TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic. TRECCOVID-NL is a Dutch translation. 

**Dataset:** [`clips/beir-nl-trec-covid`](https://huggingface.co/datasets/clips/beir-nl-trec-covid) • **License:** cc-by-4.0 • [Learn more →](https://colab.research.google.com/drive/1R99rjeAGt8S9IfAIRR3wS052sNu3Bjo-#scrollTo=4HduGW6xHnrZ)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Academic, Medical, Written | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### TRECCOVID-PL

TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic.

**Dataset:** [`mteb/TRECCOVID-PL`](https://huggingface.co/datasets/mteb/TRECCOVID-PL) • **License:** not specified • [Learn more →](https://ir.nist.gov/covidSubmit/index.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic, Medical, Non-fiction, Written | derived | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### TRECCOVID-VN

A translated dataset from TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/trec-covid-vn`](https://huggingface.co/datasets/GreenNode/trec-covid-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://ir.nist.gov/covidSubmit/index.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Academic, Medical, Written | derived | machine-translated and LM verified |



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
    



#### TV2Nordretrieval

News Article and corresponding summaries extracted from the Danish newspaper TV2 Nord.

**Dataset:** [`alexandrainst/nordjylland-news-summarization`](https://huggingface.co/datasets/alexandrainst/nordjylland-news-summarization) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/alexandrainst/nordjylland-news-summarization)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | dan | News, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{flansmose-mikkelsen-etal-2022-ddisco,
      abstract = {To date, there has been no resource for studying discourse coherence on real-world Danish texts. Discourse coherence has mostly been approached with the assumption that incoherent texts can be represented by coherent texts in which sentences have been shuffled. However, incoherent real-world texts rarely resemble that. We thus present DDisCo, a dataset including text from the Danish Wikipedia and Reddit annotated for discourse coherence. We choose to annotate real-world texts instead of relying on artificially incoherent text for training and testing models. Then, we evaluate the performance of several methods, including neural networks, on the dataset.},
      address = {Marseille, France},
      author = {Flansmose Mikkelsen, Linea  and
    Kinch, Oliver  and
    Jess Pedersen, Anders  and
    Lacroix, Oph{\'e}lie},
      booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
      editor = {Calzolari, Nicoletta  and
    B{\'e}chet, Fr{\'e}d{\'e}ric  and
    Blache, Philippe  and
    Choukri, Khalid  and
    Cieri, Christopher  and
    Declerck, Thierry  and
    Goggi, Sara  and
    Isahara, Hitoshi  and
    Maegaard, Bente  and
    Mariani, Joseph  and
    Mazo, H{\'e}l{\`e}ne  and
    Odijk, Jan  and
    Piperidis, Stelios},
      month = jun,
      pages = {2440--2445},
      publisher = {European Language Resources Association},
      title = {{DD}is{C}o: A Discourse Coherence Dataset for {D}anish},
      url = {https://aclanthology.org/2022.lrec-1.260},
      year = {2022},
    }
    
    ```
    



#### TempReasonL1

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l1.

**Dataset:** [`mteb/TempReasonL1`](https://huggingface.co/datasets/mteb/TempReasonL1) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{tan2023towards,
      author = {Tan, Qingyu and Ng, Hwee Tou and Bing, Lidong},
      journal = {arXiv preprint arXiv:2306.08952},
      title = {Towards benchmarking and improving the temporal reasoning capability of large language models},
      year = {2023},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### TempReasonL2Context

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l2-context.

**Dataset:** [`mteb/TempReasonL2Context`](https://huggingface.co/datasets/mteb/TempReasonL2Context) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{tan2023towards,
      author = {Tan, Qingyu and Ng, Hwee Tou and Bing, Lidong},
      journal = {arXiv preprint arXiv:2306.08952},
      title = {Towards benchmarking and improving the temporal reasoning capability of large language models},
      year = {2023},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### TempReasonL2Fact

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l2-fact.

**Dataset:** [`mteb/TempReasonL2Fact`](https://huggingface.co/datasets/mteb/TempReasonL2Fact) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{tan2023towards,
      author = {Tan, Qingyu and Ng, Hwee Tou and Bing, Lidong},
      journal = {arXiv preprint arXiv:2306.08952},
      title = {Towards benchmarking and improving the temporal reasoning capability of large language models},
      year = {2023},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### TempReasonL2Pure

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l2-pure.

**Dataset:** [`mteb/TempReasonL2Pure`](https://huggingface.co/datasets/mteb/TempReasonL2Pure) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{tan2023towards,
      author = {Tan, Qingyu and Ng, Hwee Tou and Bing, Lidong},
      journal = {arXiv preprint arXiv:2306.08952},
      title = {Towards benchmarking and improving the temporal reasoning capability of large language models},
      year = {2023},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### TempReasonL3Context

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l3-context.

**Dataset:** [`mteb/TempReasonL3Context`](https://huggingface.co/datasets/mteb/TempReasonL3Context) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{tan2023towards,
      author = {Tan, Qingyu and Ng, Hwee Tou and Bing, Lidong},
      journal = {arXiv preprint arXiv:2306.08952},
      title = {Towards benchmarking and improving the temporal reasoning capability of large language models},
      year = {2023},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### TempReasonL3Fact

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l3-fact.

**Dataset:** [`mteb/TempReasonL3Fact`](https://huggingface.co/datasets/mteb/TempReasonL3Fact) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{tan2023towards,
      author = {Tan, Qingyu and Ng, Hwee Tou and Bing, Lidong},
      journal = {arXiv preprint arXiv:2306.08952},
      title = {Towards benchmarking and improving the temporal reasoning capability of large language models},
      year = {2023},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### TempReasonL3Pure

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l3-pure.

**Dataset:** [`mteb/TempReasonL3Pure`](https://huggingface.co/datasets/mteb/TempReasonL3Pure) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/DAMO-NLP-SG/TempReason)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{tan2023towards,
      author = {Tan, Qingyu and Ng, Hwee Tou and Bing, Lidong},
      journal = {arXiv preprint arXiv:2306.08952},
      title = {Towards benchmarking and improving the temporal reasoning capability of large language models},
      year = {2023},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### TopiOCQA

TopiOCQA (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset) is information-seeking conversational dataset with challenging topic switching phenomena. It consists of conversation histories along with manually labelled relevant/gold passage.

**Dataset:** [`mteb/TopiOCQA`](https://huggingface.co/datasets/mteb/TopiOCQA) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://mcgill-nlp.github.io/topiocqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{adlakha2022topiocqa,
      archiveprefix = {arXiv},
      author = {Vaibhav Adlakha and Shehzaad Dhuliawala and Kaheer Suleman and Harm de Vries and Siva Reddy},
      eprint = {2110.00768},
      primaryclass = {cs.CL},
      title = {TopiOCQA: Open-domain Conversational Question Answering with Topic Switching},
      year = {2022},
    }
    
    ```
    



#### TopiOCQAHardNegatives

TopiOCQA (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset) is information-seeking conversational dataset with challenging topic switching phenomena. It consists of conversation histories along with manually labelled relevant/gold passage. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.

**Dataset:** [`mteb/TopiOCQA_validation_top_250_only_w_correct-v2`](https://huggingface.co/datasets/mteb/TopiOCQA_validation_top_250_only_w_correct-v2) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://mcgill-nlp.github.io/topiocqa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{adlakha2022topiocqa,
      archiveprefix = {arXiv},
      author = {Vaibhav Adlakha and Shehzaad Dhuliawala and Kaheer Suleman and Harm de Vries and Siva Reddy},
      eprint = {2110.00768},
      primaryclass = {cs.CL},
      title = {TopiOCQA: Open-domain Conversational Question Answering with Topic Switching},
      year = {2022},
    }
    
    ```
    



#### Touche2020

Touché Task 1: Argument Retrieval for Controversial Questions

**Dataset:** [`mteb/touche2020`](https://huggingface.co/datasets/mteb/touche2020) • **License:** cc-by-sa-4.0 • [Learn more →](https://webis.de/events/touche-20/shared-task-1.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @dataset{potthast_2022_6862281,
      author = {Potthast, Martin and
    Gienapp, Lukas and
    Wachsmuth, Henning and
    Hagen, Matthias and
    Fröbe, Maik and
    Bondarenko, Alexander and
    Ajjour, Yamen and
    Stein, Benno},
      doi = {10.5281/zenodo.6862281},
      month = jul,
      publisher = {Zenodo},
      title = {{Touché20-Argument-Retrieval-for-Controversial-
    Questions}},
      url = {https://doi.org/10.5281/zenodo.6862281},
      year = {2022},
    }
    
    ```
    



#### Touche2020-Fa

Touche2020-Fa

**Dataset:** [`MCINext/touche2020-fa`](https://huggingface.co/datasets/MCINext/touche2020-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/touche2020-fa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Spoken | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### Touche2020-Fa.v2

Touche2020-Fa.v2

**Dataset:** [`MCINext/webis-touche2020-v3-fa`](https://huggingface.co/datasets/MCINext/webis-touche2020-v3-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/touche2020-fa-v2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | fas | Spoken | derived | found |



??? quote "Citation"

    
    ```bibtex
     
    ```
    



#### Touche2020-NL

Touché Task 1: Argument Retrieval for Controversial Questions. Touche2020-NL is a Dutch translation.

**Dataset:** [`clips/beir-nl-webis-touche2020`](https://huggingface.co/datasets/clips/beir-nl-webis-touche2020) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/clips/beir-nl-webis-touche2020)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | nld | Academic, Non-fiction | derived | machine-translated and verified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{banar2024beirnlzeroshotinformationretrieval,
      archiveprefix = {arXiv},
      author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
      eprint = {2412.08329},
      primaryclass = {cs.CL},
      title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
      url = {https://arxiv.org/abs/2412.08329},
      year = {2024},
    }
    
    ```
    



#### Touche2020-PL

Touché Task 1: Argument Retrieval for Controversial Questions

**Dataset:** [`mteb/Touche2020-PL`](https://huggingface.co/datasets/mteb/Touche2020-PL) • **License:** not specified • [Learn more →](https://webis.de/events/touche-20/shared-task-1.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | pol | Academic | derived | machine-translated |



??? quote "Citation"

    
    ```bibtex
    
    @misc{wojtasik2024beirpl,
      archiveprefix = {arXiv},
      author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      eprint = {2305.19840},
      primaryclass = {cs.IR},
      title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
      year = {2024},
    }
    
    ```
    



#### Touche2020-VN

A translated dataset from Touché Task 1: Argument Retrieval for Controversial Questions
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/webis-touche2020-vn`](https://huggingface.co/datasets/GreenNode/webis-touche2020-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://webis.de/events/touche-20/shared-task-1.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Academic | derived | machine-translated and LM verified |



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
    



#### Touche2020Retrieval.v3

Touché Task 1: Argument Retrieval for Controversial Questions

**Dataset:** [`mteb/webis-touche2020-v3`](https://huggingface.co/datasets/mteb/webis-touche2020-v3) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/castorini/touche-error-analysis)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Academic | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Thakur_etal_SIGIR2024,
      address_ = {Washington, D.C.},
      author = {Nandan Thakur and Luiz Bonifacio and Maik {Fr\"{o}be} and Alexander Bondarenko and Ehsan Kamalloo and Martin Potthast and Matthias Hagen and Jimmy Lin},
      booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
      title = {Systematic Evaluation of Neural Retrieval Models on the {Touch\'{e}} 2020 Argument Retrieval Subset of {BEIR}},
      year = {2024},
    }
    
    ```
    



#### TurHistQuadRetrieval

Question Answering dataset on Ottoman History in Turkish

**Dataset:** [`asparius/TurHistQuAD`](https://huggingface.co/datasets/asparius/TurHistQuAD) • **License:** mit • [Learn more →](https://github.com/okanvk/Turkish-Reading-Comprehension-Question-Answering-Dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | tur | Academic, Encyclopaedic, Non-fiction, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{9559013,
      author = {Soygazi, Fatih and Çiftçi, Okan and Kök, Uğurcan and Cengiz, Soner},
      booktitle = {2021 6th International Conference on Computer Science and Engineering (UBMK)},
      doi = {10.1109/UBMK52708.2021.9559013},
      keywords = {Computer science;Computational modeling;Neural networks;Knowledge discovery;Information retrieval;Natural language processing;History;question answering;information retrieval;natural language understanding;deep learning;contextualized word embeddings},
      number = {},
      pages = {215-220},
      title = {THQuAD: Turkish Historic Question Answering Dataset for Reading Comprehension},
      volume = {},
      year = {2021},
    }
    
    ```
    



#### TwitterHjerneRetrieval

Danish question asked on Twitter with the Hashtag #Twitterhjerne ('Twitter brain') and their corresponding answer.

**Dataset:** [`sorenmulli/da-hashtag-twitterhjerne`](https://huggingface.co/datasets/sorenmulli/da-hashtag-twitterhjerne) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/sorenmulli/da-hashtag-twitterhjerne)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | dan | Social, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{holm2024gllms,
      author = {Holm, Soren Vejlgaard},
      title = {Are GLLMs Danoliterate? Benchmarking Generative NLP in Danish},
      year = {2024},
    }
    
    ```
    



#### VDRMultilingualRetrieval

Multilingual Visual Document retrieval Dataset covering 5 languages: Italian, Spanish, English, French and German

**Dataset:** [`llamaindex/vdr-multilingual-test`](https://huggingface.co/datasets/llamaindex/vdr-multilingual-test) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/llamaindex/vdr-multilingual-test)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| image, text to image, text (it2it) | ndcg_at_5 | deu, eng, fra, ita, spa | Web | LM-generated | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{llamaindex2024vdrmultilingual,
      author = {LlamaIndex},
      howpublished = {https://huggingface.co/datasets/llamaindex/vdr-multilingual-test},
      title = {Visual Document Retrieval Goes Multilingual},
      year = {2025},
    }
    
    ```
    



#### VideoRetrieval

VideoRetrieval

**Dataset:** [`mteb/VideoRetrieval`](https://huggingface.co/datasets/mteb/VideoRetrieval) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2203.03367)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | cmn | not specified | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @misc{long2022multicprmultidomainchinese,
      archiveprefix = {arXiv},
      author = {Dingkun Long and Qiong Gao and Kuan Zou and Guangwei Xu and Pengjun Xie and Ruijie Guo and Jian Xu and Guanjun Jiang and Luxi Xing and Ping Yang},
      eprint = {2203.03367},
      primaryclass = {cs.IR},
      title = {Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval},
      url = {https://arxiv.org/abs/2203.03367},
      year = {2022},
    }
    
    ```
    



#### VieQuADRetrieval

A Vietnamese dataset for evaluating Machine Reading Comprehension from Wikipedia articles.

**Dataset:** [`taidng/UIT-ViQuAD2.0`](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0) • **License:** mit • [Learn more →](https://aclanthology.org/2020.coling-main.233.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | vie | Encyclopaedic, Non-fiction, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{nguyen-etal-2020-vietnamese,
      address = {Barcelona, Spain (Online)},
      author = {Nguyen, Kiet  and
    Nguyen, Vu  and
    Nguyen, Anh  and
    Nguyen, Ngan},
      booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
      doi = {10.18653/v1/2020.coling-main.233},
      editor = {Scott, Donia  and
    Bel, Nuria  and
    Zong, Chengqing},
      month = dec,
      pages = {2595--2605},
      publisher = {International Committee on Computational Linguistics},
      title = {A Vietnamese Dataset for Evaluating Machine Reading Comprehension},
      url = {https://aclanthology.org/2020.coling-main.233},
      year = {2020},
    }
    
    ```
    



#### WebFAQRetrieval

WebFAQ is a broad-coverage corpus of natural question-answer pairs in 75 languages, gathered from FAQ pages on the web.

**Dataset:** [`PaDaS-Lab/webfaq-retrieval`](https://huggingface.co/datasets/PaDaS-Lab/webfaq-retrieval) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/PaDaS-Lab)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, aze, ben, bul, cat, ... (51) | Web, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @misc{dinzinger2025webfaq,
      archiveprefix = {arXiv},
      author = {Michael Dinzinger and Laura Caspari and Kanishka Ghosh Dastidar and Jelena Mitrović and Michael Granitzer},
      eprint = {2502.20936},
      primaryclass = {cs.CL},
      title = {WebFAQ: A Multilingual Collection of Natural Q&amp;A Datasets for Dense Retrieval},
      url = {https://arxiv.org/abs/2502.20936},
      year = {2025},
    }
    
    ```
    



#### WikiSQLRetrieval

A code retrieval task based on WikiSQL dataset with natural language questions and corresponding SQL queries. Each query is a natural language question (e.g., 'What is the name of the team that has scored the most goals?'), and the corpus contains SQL query implementations. The task is to retrieve the correct SQL query that answers the natural language question. Queries are natural language questions while the corpus contains SQL SELECT statements with proper syntax and logic for querying database tables.

**Dataset:** [`embedding-benchmark/WikiSQL_mteb`](https://huggingface.co/datasets/embedding-benchmark/WikiSQL_mteb) • **License:** bsd-3-clause • [Learn more →](https://huggingface.co/datasets/embedding-benchmark/WikiSQL_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng, sql | Programming | expert-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{zhong2017seq2sql,
      archiveprefix = {arXiv},
      author = {Zhong, Victor and Xiong, Caiming and Socher, Richard},
      eprint = {1709.00103},
      primaryclass = {cs.CL},
      title = {Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning},
      year = {2017},
    }
    
    ```
    



#### WikipediaRetrievalMultilingual

The dataset is derived from Cohere's wikipedia-2023-11 dataset and contains synthetically generated queries.

**Dataset:** [`mteb/WikipediaRetrievalMultilingual`](https://huggingface.co/datasets/mteb/WikipediaRetrievalMultilingual) • **License:** cc-by-sa-3.0 • [Learn more →](https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-multilingual-queries)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ben, bul, ces, dan, deu, ... (16) | Encyclopaedic, Written | LM-generated and reviewed | LM-generated and verified |



#### WinoGrande

Measuring the ability to retrieve the groundtruth answers to reasoning task queries on winogrande.

**Dataset:** [`mteb/WinoGrande`](https://huggingface.co/datasets/mteb/WinoGrande) • **License:** not specified • [Learn more →](https://winogrande.allenai.org/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"

    
    ```bibtex
    
    @article{sakaguchi2021winogrande,
      author = {Sakaguchi, Keisuke and Bras, Ronan Le and Bhagavatula, Chandra and Choi, Yejin},
      journal = {Communications of the ACM},
      number = {9},
      pages = {99--106},
      publisher = {ACM New York, NY, USA},
      title = {Winogrande: An adversarial winograd schema challenge at scale},
      volume = {64},
      year = {2021},
    }
    
    @article{xiao2024rar,
      author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2404.06347},
      title = {RAR-b: Reasoning as Retrieval Benchmark},
      year = {2024},
    }
    
    ```
    



#### XMarket

XMarket

**Dataset:** [`mteb/XMarket`](https://huggingface.co/datasets/mteb/XMarket) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb/XMarket)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | deu, eng, spa | not specified | not specified | not specified |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{Bonab_2021,
      author = {Bonab, Hamed and Aliannejadi, Mohammad and Vardasbi, Ali and Kanoulas, Evangelos and Allan, James},
      booktitle = {Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management},
      collection = {CIKM ’21},
      doi = {10.1145/3459637.3482493},
      month = oct,
      publisher = {ACM},
      series = {CIKM ’21},
      title = {Cross-Market Product Recommendation},
      url = {http://dx.doi.org/10.1145/3459637.3482493},
      year = {2021},
    }
    
    ```
    



#### XPQARetrieval

XPQARetrieval

**Dataset:** [`mteb/XPQARetrieval`](https://huggingface.co/datasets/mteb/XPQARetrieval) • **License:** cdla-sharing-1.0 • [Learn more →](https://arxiv.org/abs/2305.09249)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | ara, cmn, deu, eng, fra, ... (13) | Reviews, Written | human-annotated | found |



??? quote "Citation"

    
    ```bibtex
    
    @inproceedings{shen2023xpqa,
      author = {Shen, Xiaoyu and Asai, Akari and Byrne, Bill and De Gispert, Adria},
      booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 5: Industry Track)},
      pages = {103--115},
      title = {xPQA: Cross-Lingual Product Question Answering in 12 Languages},
      year = {2023},
    }
    
    ```
    



#### XQuADRetrieval

XQuAD is a benchmark dataset for evaluating cross-lingual question answering performance. It is repurposed retrieving relevant context for each question.

**Dataset:** [`google/xquad`](https://huggingface.co/datasets/google/xquad) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/xquad)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_10 | arb, deu, ell, eng, hin, ... (12) | Web, Written | human-annotated | created |



??? quote "Citation"

    
    ```bibtex
    
    @article{Artetxe:etal:2019,
      archiveprefix = {arXiv},
      author = {Mikel Artetxe and Sebastian Ruder and Dani Yogatama},
      eprint = {1910.11856},
      journal = {CoRR},
      title = {On the cross-lingual transferability of monolingual representations},
      volume = {abs/1910.11856},
      year = {2019},
    }
    
    @inproceedings{dumitrescu2021liro,
      author = {Stefan Daniel Dumitrescu and Petru Rebeja and Beata Lorincz and Mihaela Gaman and Andrei Avram and Mihai Ilie and Andrei Pruteanu and Adriana Stan and Lorena Rosia and Cristina Iacobescu and Luciana Morogan and George Dima and Gabriel Marchidan and Traian Rebedea and Madalina Chitez and Dani Yogatama and Sebastian Ruder and Radu Tudor Ionescu and Razvan Pascanu and Viorica Patraucean},
      booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
      title = {LiRo: Benchmark and leaderboard for Romanian language tasks},
      url = {https://openreview.net/forum?id=JH61CD7afTv},
      year = {2021},
    }
    
    ```
    



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



??? quote "Citation"

    
    ```bibtex
    
    @article{DBLP:journals/corr/abs-2108-13897,
      author = {Luiz Bonifacio and
    Israel Campiotti and
    Roberto de Alencar Lotufo and
    Rodrigo Frassetto Nogueira},
      bibsource = {dblp computer science bibliography, https://dblp.org},
      biburl = {https://dblp.org/rec/journals/corr/abs-2108-13897.bib},
      eprint = {2108.13897},
      eprinttype = {arXiv},
      journal = {CoRR},
      timestamp = {Mon, 20 Mar 2023 15:35:34 +0100},
      title = {mMARCO: {A} Multilingual Version of {MS} {MARCO} Passage Ranking Dataset},
      url = {https://arxiv.org/abs/2108.13897},
      volume = {abs/2108.13897},
      year = {2021},
    }
    
    ```
