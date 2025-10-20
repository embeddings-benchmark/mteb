
# STS

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 47

#### AFQMC

A Chinese dataset for textual relatedness

**Dataset:** [`C-MTEB/AFQMC`](https://huggingface.co/datasets/C-MTEB/AFQMC) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.emnlp-main.357)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | cmn | not specified | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @inproceedings{raghu-etal-2021-end,
      address = {Online and Punta Cana, Dominican Republic},
      author = {Raghu, Dinesh  and
    Agarwal, Shantanu  and
    Joshi, Sachindra  and
    {Mausam}},
      booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
      doi = {10.18653/v1/2021.emnlp-main.357},
      editor = {Moens, Marie-Francine  and
    Huang, Xuanjing  and
    Specia, Lucia  and
    Yih, Scott Wen-tau},
      month = nov,
      pages = {4348--4366},
      publisher = {Association for Computational Linguistics},
      title = {End-to-End Learning of Flowchart Grounded Task-Oriented Dialogs},
      url = {https://aclanthology.org/2021.emnlp-main.357},
      year = {2021},
    }

    ```




#### ATEC

A Chinese dataset for textual relatedness

**Dataset:** [`C-MTEB/ATEC`](https://huggingface.co/datasets/C-MTEB/ATEC) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.emnlp-main.357)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | cmn | not specified | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @inproceedings{raghu-etal-2021-end,
      address = {Online and Punta Cana, Dominican Republic},
      author = {Raghu, Dinesh  and
    Agarwal, Shantanu  and
    Joshi, Sachindra  and
    {Mausam}},
      booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
      doi = {10.18653/v1/2021.emnlp-main.357},
      editor = {Moens, Marie-Francine  and
    Huang, Xuanjing  and
    Specia, Lucia  and
    Yih, Scott Wen-tau},
      month = nov,
      pages = {4348--4366},
      publisher = {Association for Computational Linguistics},
      title = {End-to-End Learning of Flowchart Grounded Task-Oriented Dialogs},
      url = {https://aclanthology.org/2021.emnlp-main.357},
      year = {2021},
    }

    ```




#### Assin2STS

Semantic Textual Similarity part of the ASSIN 2, an evaluation shared task collocated with STIL 2019.

**Dataset:** [`nilc-nlp/assin2`](https://huggingface.co/datasets/nilc-nlp/assin2) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-030-41505-1_39)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | por | Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{real2020assin,
      author = {Real, Livy and Fonseca, Erick and Oliveira, Hugo Goncalo},
      booktitle = {International Conference on Computational Processing of the Portuguese Language},
      organization = {Springer},
      pages = {406--412},
      title = {The assin 2 shared task: a quick overview},
      year = {2020},
    }

    ```




#### BIOSSES

Biomedical Semantic Similarity Estimation.

**Dataset:** [`mteb/biosses-sts`](https://huggingface.co/datasets/mteb/biosses-sts) • **License:** not specified • [Learn more →](https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | Medical | derived | found |



??? quote "Citation"


    ```bibtex

    @article{10.1093/bioinformatics/btx238,
      author = {Soğancıoğlu, Gizem and Öztürk, Hakime and Özgür, Arzucan},
      doi = {10.1093/bioinformatics/btx238},
      eprint = {https://academic.oup.com/bioinformatics/article-pdf/33/14/i49/50315066/bioinformatics\_33\_14\_i49.pdf},
      issn = {1367-4803},
      journal = {Bioinformatics},
      month = {07},
      number = {14},
      pages = {i49-i58},
      title = {{BIOSSES: a semantic sentence similarity estimation system for the biomedical domain}},
      url = {https://doi.org/10.1093/bioinformatics/btx238},
      volume = {33},
      year = {2017},
    }

    ```




#### BIOSSES-VN

A translated dataset from Biomedical Semantic Similarity Estimation.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/biosses-sts-vn`](https://huggingface.co/datasets/GreenNode/biosses-sts-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | cosine_spearman | vie | Medical | derived | machine-translated and LM verified |



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




#### BQ

A Chinese dataset for textual relatedness

**Dataset:** [`C-MTEB/BQ`](https://huggingface.co/datasets/C-MTEB/BQ) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.emnlp-main.357)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | cmn | not specified | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @misc{xiao2024cpackpackagedresourcesadvance,
      archiveprefix = {arXiv},
      author = {Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff and Defu Lian and Jian-Yun Nie},
      eprint = {2309.07597},
      primaryclass = {cs.CL},
      title = {C-Pack: Packaged Resources To Advance General Chinese Embedding},
      url = {https://arxiv.org/abs/2309.07597},
      year = {2024},
    }

    ```




#### CDSC-R

Compositional Distributional Semantics Corpus for textual relatedness.

**Dataset:** [`PL-MTEB/cdscr-sts`](https://huggingface.co/datasets/PL-MTEB/cdscr-sts) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/P17-1073.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | pol | Web, Written | human-annotated | human-translated and localized |



??? quote "Citation"


    ```bibtex

    @inproceedings{wroblewska-krasnowska-kieras-2017-polish,
      address = {Vancouver, Canada},
      author = {Wr{\'o}blewska, Alina  and
    Krasnowska-Kiera{\'s}, Katarzyna},
      booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      doi = {10.18653/v1/P17-1073},
      editor = {Barzilay, Regina  and
    Kan, Min-Yen},
      month = jul,
      pages = {784--792},
      publisher = {Association for Computational Linguistics},
      title = {{P}olish evaluation dataset for compositional distributional semantics models},
      url = {https://aclanthology.org/P17-1073},
      year = {2017},
    }

    ```




#### FaroeseSTS

Semantic Text Similarity (STS) corpus for Faroese.

**Dataset:** [`vesteinn/faroese-sts`](https://huggingface.co/datasets/vesteinn/faroese-sts) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.74.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | fao | News, Web, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{snaebjarnarson-etal-2023-transfer,
      address = {Tórshavn, Faroe Islands},
      author = {Snæbjarnarson, Vésteinn  and
    Simonsen, Annika  and
    Glavaš, Goran  and
    Vulić, Ivan},
      booktitle = {Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)},
      month = {may 22--24},
      publisher = {Link{\"o}ping University Electronic Press, Sweden},
      title = {{T}ransfer to a Low-Resource Language via Close Relatives: The Case Study on Faroese},
      year = {2023},
    }

    ```




#### Farsick

A Persian Semantic Textual Similarity And Natural Language Inference Dataset

**Dataset:** [`MCINext/farsick-sts`](https://huggingface.co/datasets/MCINext/farsick-sts) • **License:** not specified • [Learn more →](https://github.com/ZahraGhasemi-AI/FarSick)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | fas | not specified | derived | found |



??? quote "Citation"


    ```bibtex

    ```




#### FinParaSTS

Finnish paraphrase-based semantic similarity corpus

**Dataset:** [`mteb/FinParaSTS`](https://huggingface.co/datasets/mteb/FinParaSTS) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/TurkuNLP/turku_paraphrase_corpus)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | fin | News, Subtitles, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{kanerva-etal-2021-finnish,
      address = {Reykjavik, Iceland (Online)},
      author = {Kanerva, Jenna  and
    Ginter, Filip  and
    Chang, Li-Hsin  and
    Rastas, Iiro  and
    Skantsi, Valtteri  and
    Kilpel{\"a}inen, Jemina  and
    Kupari, Hanna-Mari  and
    Saarni, Jenna  and
    Sev{\'o}n, Maija  and
    Tarkka, Otto},
      booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
      editor = {Dobnik, Simon  and
    {\\O}vrelid, Lilja},
      month = may # { 31--2 } # jun,
      pages = {288--298},
      publisher = {Link{\"o}ping University Electronic Press, Sweden},
      title = {{F}innish Paraphrase Corpus},
      url = {https://aclanthology.org/2021.nodalida-main.29},
      year = {2021},
    }

    ```




#### GermanSTSBenchmark

Semantic Textual Similarity Benchmark (STSbenchmark) dataset translated into German. Translations were originally done by T-Systems on site services GmbH.

**Dataset:** [`jinaai/german-STSbenchmark`](https://huggingface.co/datasets/jinaai/german-STSbenchmark) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/t-systems-on-site-services-gmbh/german-STSbenchmark)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | deu | not specified | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @inproceedings{huggingface:dataset:stsb_multi_mt,
      author = {Philip May},
      title = {Machine translated multilingual STS benchmark dataset.},
      url = {https://github.com/PhilipMay/stsb-multi-mt},
      year = {2021},
    }

    ```




#### HUMESICK-R

Human evaluation subset of Semantic Textual Similarity SICK-R dataset

**Dataset:** [`mteb/mteb-human-sickr-sts`](https://huggingface.co/datasets/mteb/mteb-human-sickr-sts) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://aclanthology.org/L14-1314/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | Web, Written | human-annotated | not specified |



??? quote "Citation"


    ```bibtex

    @inproceedings{marelli-etal-2014-sick,
      address = {Reykjavik, Iceland},
      author = {Marelli, Marco  and
    Menini, Stefano  and
    Baroni, Marco  and
    Bentivogli, Luisa  and
    Bernardi, Raffaella  and
    Zamparelli, Roberto},
      booktitle = {Proceedings of the Ninth International Conference on Language Resources and Evaluation ({LREC}'14)},
      editor = {Calzolari, Nicoletta  and
    Choukri, Khalid  and
    Declerck, Thierry  and
    Loftsson, Hrafn  and
    Maegaard, Bente  and
    Mariani, Joseph  and
    Moreno, Asuncion  and
    Odijk, Jan  and
    Piperidis, Stelios},
      month = may,
      pages = {216--223},
      publisher = {European Language Resources Association (ELRA)},
      title = {A {SICK} cure for the evaluation of compositional distributional semantic models},
      url = {http://www.lrec-conf.org/proceedings/lrec2014/pdf/363_Paper.pdf},
      year = {2014},
    }

    ```




#### HUMESTS12

Human evaluation subset of SemEval-2012 Task 6.

**Dataset:** [`mteb/mteb-human-sts12-sts`](https://huggingface.co/datasets/mteb/mteb-human-sts12-sts) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/S12-1051.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | Encyclopaedic, News, Written | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{10.5555/2387636.2387697,
      address = {USA},
      author = {Agirre, Eneko and Diab, Mona and Cer, Daniel and Gonzalez-Agirre, Aitor},
      booktitle = {Proceedings of the First Joint Conference on Lexical and Computational Semantics - Volume 1: Proceedings of the Main Conference and the Shared Task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation},
      location = {Montr\'{e}al, Canada},
      numpages = {9},
      pages = {385–393},
      publisher = {Association for Computational Linguistics},
      series = {SemEval '12},
      title = {SemEval-2012 task 6: a pilot on semantic textual similarity},
      year = {2012},
    }

    ```




#### HUMESTS22

Human evaluation subset of SemEval 2022 Task 8: Multilingual News Article Similarity

**Dataset:** [`mteb/mteb-human-sts22-sts`](https://huggingface.co/datasets/mteb/mteb-human-sts22-sts) • **License:** not specified • [Learn more →](https://competitions.codalab.org/competitions/33835)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | ara, eng, fra, rus | News, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{chen-etal-2022-semeval,
      address = {Seattle, United States},
      author = {Chen, Xi  and
    Zeynali, Ali  and
    Camargo, Chico  and
    Fl{\"o}ck, Fabian  and
    Gaffney, Devin  and
    Grabowicz, Przemyslaw  and
    Hale, Scott  and
    Jurgens, David  and
    Samory, Mattia},
      booktitle = {Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)},
      doi = {10.18653/v1/2022.semeval-1.155},
      editor = {Emerson, Guy  and
    Schluter, Natalie  and
    Stanovsky, Gabriel  and
    Kumar, Ritesh  and
    Palmer, Alexis  and
    Schneider, Nathan  and
    Singh, Siddharth  and
    Ratan, Shyam},
      month = jul,
      pages = {1094--1106},
      publisher = {Association for Computational Linguistics},
      title = {{S}em{E}val-2022 Task 8: Multilingual news article similarity},
      url = {https://aclanthology.org/2022.semeval-1.155},
      year = {2022},
    }

    ```




#### HUMESTSBenchmark

Human evaluation subset of Semantic Textual Similarity Benchmark (STSbenchmark) dataset.

**Dataset:** [`mteb/mteb-human-stsbenchmark-sts`](https://huggingface.co/datasets/mteb/mteb-human-stsbenchmark-sts) • **License:** not specified • [Learn more →](https://github.com/PhilipMay/stsb-multi-mt/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | Blog, News, Written | human-annotated | machine-translated and verified |



??? quote "Citation"


    ```bibtex

    @inproceedings{huggingface:dataset:stsb_multi_mt,
      author = {Philip May},
      title = {Machine translated multilingual STS benchmark dataset.},
      url = {https://github.com/PhilipMay/stsb-multi-mt},
      year = {2021},
    }

    ```




#### IndicCrosslingualSTS

This is a Semantic Textual Similarity testset between English and 12 high-resource Indic languages.

**Dataset:** [`mteb/IndicCrosslingualSTS`](https://huggingface.co/datasets/mteb/IndicCrosslingualSTS) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jaygala24/indic_sts)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | asm, ben, eng, guj, hin, ... (13) | Government, News, Non-fiction, Spoken, Spoken, ... (7) | expert-annotated | created |



??? quote "Citation"


    ```bibtex

    @article{10.1162/tacl_a_00452,
      author = {Ramesh, Gowtham and Doddapaneni, Sumanth and Bheemaraj, Aravinth and Jobanputra, Mayank and AK, Raghavan and Sharma, Ajitesh and Sahoo, Sujit and Diddee, Harshita and J, Mahalakshmi and Kakwani, Divyanshu and Kumar, Navneet and Pradeep, Aswin and Nagaraj, Srihari and Deepak, Kumar and Raghavan, Vivek and Kunchukuttan, Anoop and Kumar, Pratyush and Khapra, Mitesh Shantadevi},
      doi = {10.1162/tacl_a_00452},
      eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\\_a\\_00452/1987010/tacl\\_a\\_00452.pdf},
      issn = {2307-387X},
      journal = {Transactions of the Association for Computational Linguistics},
      month = {02},
      pages = {145-162},
      title = {{Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages}},
      url = {https://doi.org/10.1162/tacl\\_a\\_00452},
      volume = {10},
      year = {2022},
    }

    ```




#### JSICK

JSICK is the Japanese NLI and STS dataset by manually translating the English dataset SICK (Marelli et al., 2014) into Japanese.

**Dataset:** [`mteb/JSICK`](https://huggingface.co/datasets/mteb/JSICK) • **License:** cc-by-4.0 • [Learn more →](https://github.com/sbintuitions/JMTEB)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | jpn | Web, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{yanaka2022compositional,
      author = {Yanaka, Hitomi and Mineshima, Koji},
      journal = {Transactions of the Association for Computational Linguistics},
      pages = {1266--1284},
      publisher = {MIT Press One Broadway, 12th Floor, Cambridge, Massachusetts 02142, USA~…},
      title = {Compositional Evaluation on Japanese Textual Entailment and Similarity},
      volume = {10},
      year = {2022},
    }

    ```




#### JSTS

Japanese Semantic Textual Similarity Benchmark dataset construct from YJ Image Captions Dataset (Miyazaki and Shimizu, 2016) and annotated by crowdsource annotators.

**Dataset:** [`mteb/JSTS`](https://huggingface.co/datasets/mteb/JSTS) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2022.lrec-1.317.pdf#page=2.00)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | jpn | Web, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{kurihara-etal-2022-jglue,
      address = {Marseille, France},
      author = {Kurihara, Kentaro  and
    Kawahara, Daisuke  and
    Shibata, Tomohide},
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
      pages = {2957--2966},
      publisher = {European Language Resources Association},
      title = {{JGLUE}: {J}apanese General Language Understanding Evaluation},
      url = {https://aclanthology.org/2022.lrec-1.317},
      year = {2022},
    }

    ```




#### KLUE-STS

Human-annotated STS dataset of Korean reviews, news, and spoken word sets. Part of the Korean Language Understanding Evaluation (KLUE).

**Dataset:** [`klue/klue`](https://huggingface.co/datasets/klue/klue) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2105.09680)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | kor | News, Reviews, Spoken, Spoken, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{park2021klue,
      archiveprefix = {arXiv},
      author = {Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
      eprint = {2105.09680},
      primaryclass = {cs.CL},
      title = {KLUE: Korean Language Understanding Evaluation},
      year = {2021},
    }

    ```




#### KorSTS

Benchmark dataset for STS in Korean. Created by machine translation and human post editing of the STS-B dataset.

**Dataset:** [`dkoterwa/kor-sts`](https://huggingface.co/datasets/dkoterwa/kor-sts) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2004.03289)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | kor | News, Web | not specified | machine-translated and localized |



??? quote "Citation"


    ```bibtex

    @article{ham2020kornli,
      author = {Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
      journal = {arXiv preprint arXiv:2004.03289},
      title = {KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
      year = {2020},
    }

    ```




#### LCQMC

A Chinese dataset for textual relatedness

**Dataset:** [`C-MTEB/LCQMC`](https://huggingface.co/datasets/C-MTEB/LCQMC) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.emnlp-main.357)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | cmn | not specified | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @misc{xiao2024cpackpackagedresourcesadvance,
      archiveprefix = {arXiv},
      author = {Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff and Defu Lian and Jian-Yun Nie},
      eprint = {2309.07597},
      primaryclass = {cs.CL},
      title = {C-Pack: Packaged Resources To Advance General Chinese Embedding},
      url = {https://arxiv.org/abs/2309.07597},
      year = {2024},
    }

    ```




#### PAWSX

A Chinese dataset for textual relatedness

**Dataset:** [`mteb/PAWSX`](https://huggingface.co/datasets/mteb/PAWSX) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.emnlp-main.357)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | cmn | not specified | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @misc{xiao2024cpackpackagedresourcesadvance,
      archiveprefix = {arXiv},
      author = {Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff and Defu Lian and Jian-Yun Nie},
      eprint = {2309.07597},
      primaryclass = {cs.CL},
      title = {C-Pack: Packaged Resources To Advance General Chinese Embedding},
      url = {https://arxiv.org/abs/2309.07597},
      year = {2024},
    }

    ```




#### QBQTC



**Dataset:** [`C-MTEB/QBQTC`](https://huggingface.co/datasets/C-MTEB/QBQTC) • **License:** not specified • [Learn more →](https://github.com/CLUEbenchmark/QBQTC/tree/main/dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | cmn | not specified | not specified | not specified |



#### Query2Query

Query to Query Datasets.

**Dataset:** [`MCINext/query-to-query-sts`](https://huggingface.co/datasets/MCINext/query-to-query-sts) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | fas | not specified | derived | found |



??? quote "Citation"


    ```bibtex

    ```




#### RUParaPhraserSTS

ParaPhraser is a news headlines corpus with precise, near and non-paraphrases.

**Dataset:** [`merionum/ru_paraphraser`](https://huggingface.co/datasets/merionum/ru_paraphraser) • **License:** mit • [Learn more →](https://aclanthology.org/2020.ngt-1.6)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | rus | News, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{gudkov-etal-2020-automatically,
      address = {Online},
      author = {Gudkov, Vadim  and
    Mitrofanova, Olga  and
    Filippskikh, Elizaveta},
      booktitle = {Proceedings of the Fourth Workshop on Neural Generation and Translation},
      doi = {10.18653/v1/2020.ngt-1.6},
      month = jul,
      pages = {54--59},
      publisher = {Association for Computational Linguistics},
      title = {Automatically Ranked {R}ussian Paraphrase Corpus for Text Generation},
      url = {https://aclanthology.org/2020.ngt-1.6},
      year = {2020},
    }

    @inproceedings{pivovarova2017paraphraser,
      author = {Pivovarova, Lidia and Pronoza, Ekaterina and Yagunova, Elena and Pronoza, Anton},
      booktitle = {Conference on artificial intelligence and natural language},
      organization = {Springer},
      pages = {211--225},
      title = {ParaPhraser: Russian paraphrase corpus and shared task},
      year = {2017},
    }

    ```




#### RonSTS

High-quality Romanian translation of STSBenchmark.

**Dataset:** [`mteb/RonSTS`](https://huggingface.co/datasets/mteb/RonSTS) • **License:** cc-by-4.0 • [Learn more →](https://openreview.net/forum?id=JH61CD7afTv)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | ron | News, Social, Web, Written | human-annotated | machine-translated and verified |



??? quote "Citation"


    ```bibtex

    @inproceedings{dumitrescu2021liro,
      author = {Dumitrescu, Stefan Daniel and Rebeja, Petru and Lorincz, Beata and Gaman, Mihaela and Avram, Andrei and Ilie, Mihai and Pruteanu, Andrei and Stan, Adriana and Rosia, Lorena and Iacobescu, Cristina and others},
      booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
      title = {LiRo: Benchmark and leaderboard for Romanian language tasks},
      year = {2021},
    }

    ```




#### RuSTSBenchmarkSTS

Semantic Textual Similarity Benchmark (STSbenchmark) dataset translated into Russian and verified. The dataset was checked with RuCOLA model to ensure that the translation is good and filtered.

**Dataset:** [`ai-forever/ru-stsbenchmark-sts`](https://huggingface.co/datasets/ai-forever/ru-stsbenchmark-sts) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/PhilipMay/stsb-multi-mt/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | rus | News, Social, Web, Written | human-annotated | machine-translated and verified |



??? quote "Citation"


    ```bibtex

    @inproceedings{huggingface:dataset:stsb_multi_mt,
      author = {Philip May},
      title = {Machine translated multilingual STS benchmark dataset.},
      url = {https://github.com/PhilipMay/stsb-multi-mt},
      year = {2021},
    }

    ```




#### SICK-BR-STS

SICK-BR is a Portuguese inference corpus, human translated from SICK

**Dataset:** [`eduagarcia/sick-br`](https://huggingface.co/datasets/eduagarcia/sick-br) • **License:** not specified • [Learn more →](https://linux.ime.usp.br/~thalen/SICK_PT.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | por | Web, Written | human-annotated | human-translated and localized |



??? quote "Citation"


    ```bibtex

    @inproceedings{real18,
      author = {Real, Livy
    and Rodrigues, Ana
    and Vieira e Silva, Andressa
    and Albiero, Beatriz
    and Thalenberg, Bruna
    and Guide, Bruno
    and Silva, Cindy
    and de Oliveira Lima, Guilherme
    and Camara, Igor C. S.
    and Stanojevi{\'{c}}, Milo{\v{s}}
    and Souza, Rodrigo
    and de Paiva, Valeria},
      booktitle = {{Computational Processing of the Portuguese Language. PROPOR 2018.}},
      doi = {10.1007/978-3-319-99722-3_31},
      isbn = {978-3-319-99722-3},
      title = {{SICK-BR: A Portuguese Corpus for Inference}},
      year = {2018},
    }

    ```




#### SICK-R

Semantic Textual Similarity SICK-R dataset

**Dataset:** [`mteb/sickr-sts`](https://huggingface.co/datasets/mteb/sickr-sts) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://aclanthology.org/L14-1314/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | Web, Written | human-annotated | not specified |



??? quote "Citation"


    ```bibtex

    @inproceedings{marelli-etal-2014-sick,
      address = {Reykjavik, Iceland},
      author = {Marelli, Marco  and
    Menini, Stefano  and
    Baroni, Marco  and
    Bentivogli, Luisa  and
    Bernardi, Raffaella  and
    Zamparelli, Roberto},
      booktitle = {Proceedings of the Ninth International Conference on Language Resources and Evaluation ({LREC}'14)},
      editor = {Calzolari, Nicoletta  and
    Choukri, Khalid  and
    Declerck, Thierry  and
    Loftsson, Hrafn  and
    Maegaard, Bente  and
    Mariani, Joseph  and
    Moreno, Asuncion  and
    Odijk, Jan  and
    Piperidis, Stelios},
      month = may,
      pages = {216--223},
      publisher = {European Language Resources Association (ELRA)},
      title = {A {SICK} cure for the evaluation of compositional distributional semantic models},
      url = {http://www.lrec-conf.org/proceedings/lrec2014/pdf/363_Paper.pdf},
      year = {2014},
    }

    ```




#### SICK-R-PL

Polish version of SICK dataset for textual relatedness.

**Dataset:** [`PL-MTEB/sickr-pl-sts`](https://huggingface.co/datasets/PL-MTEB/sickr-pl-sts) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://aclanthology.org/2020.lrec-1.207)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | pol | Web, Written | human-annotated | human-translated and localized |



??? quote "Citation"


    ```bibtex

    @inproceedings{dadas-etal-2020-evaluation,
      address = {Marseille, France},
      author = {Dadas, Slawomir  and
    Perelkiewicz, Michal  and
    Poswiata, Rafal},
      booktitle = {Proceedings of the Twelfth Language Resources and Evaluation Conference},
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
    Mazo, Helene  and
    Moreno, Asuncion  and
    Odijk, Jan  and
    Piperidis, Stelios},
      isbn = {979-10-95546-34-4},
      language = {English},
      month = may,
      pages = {1674--1680},
      publisher = {European Language Resources Association},
      title = {Evaluation of Sentence Representations in {P}olish},
      url = {https://aclanthology.org/2020.lrec-1.207},
      year = {2020},
    }

    ```




#### SICK-R-VN

A translated dataset from Semantic Textual Similarity SICK-R dataset as described here:
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/sickr-sts-vn`](https://huggingface.co/datasets/GreenNode/sickr-sts-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2020.lrec-1.207)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | cosine_spearman | vie | Web, Written | derived | machine-translated and LM verified |



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




#### SICKFr

SICK dataset french version

**Dataset:** [`Lajavaness/SICK-fr`](https://huggingface.co/datasets/Lajavaness/SICK-fr) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/Lajavaness/SICK-fr)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | fra | not specified | not specified | not specified |



#### STS12

SemEval-2012 Task 6.

**Dataset:** [`mteb/sts12-sts`](https://huggingface.co/datasets/mteb/sts12-sts) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/S12-1051.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | Encyclopaedic, News, Written | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{10.5555/2387636.2387697,
      address = {USA},
      author = {Agirre, Eneko and Diab, Mona and Cer, Daniel and Gonzalez-Agirre, Aitor},
      booktitle = {Proceedings of the First Joint Conference on Lexical and Computational Semantics - Volume 1: Proceedings of the Main Conference and the Shared Task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation},
      location = {Montr\'{e}al, Canada},
      numpages = {9},
      pages = {385–393},
      publisher = {Association for Computational Linguistics},
      series = {SemEval '12},
      title = {SemEval-2012 task 6: a pilot on semantic textual similarity},
      year = {2012},
    }

    ```




#### STS13

SemEval STS 2013 dataset.

**Dataset:** [`mteb/sts13-sts`](https://huggingface.co/datasets/mteb/sts13-sts) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/S13-1004/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | News, Non-fiction, Web, Written | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{Agirre2013SEM2S,
      author = {Eneko Agirre and Daniel Matthew Cer and Mona T. Diab and Aitor Gonzalez-Agirre and Weiwei Guo},
      booktitle = {International Workshop on Semantic Evaluation},
      title = {*SEM 2013 shared task: Semantic Textual Similarity},
      url = {https://api.semanticscholar.org/CorpusID:10241043},
      year = {2013},
    }

    ```




#### STS14

SemEval STS 2014 dataset. Currently only the English dataset

**Dataset:** [`mteb/sts14-sts`](https://huggingface.co/datasets/mteb/sts14-sts) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/S14-1002)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | Blog, Spoken, Web | derived | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{bandhakavi-etal-2014-generating,
      address = {Dublin, Ireland},
      author = {Bandhakavi, Anil  and
    Wiratunga, Nirmalie  and
    P, Deepak  and
    Massie, Stewart},
      booktitle = {Proceedings of the Third Joint Conference on Lexical and Computational Semantics (*{SEM} 2014)},
      doi = {10.3115/v1/S14-1002},
      editor = {Bos, Johan  and
    Frank, Anette  and
    Navigli, Roberto},
      month = aug,
      pages = {12--21},
      publisher = {Association for Computational Linguistics and Dublin City University},
      title = {Generating a Word-Emotion Lexicon from {\#}Emotional Tweets},
      url = {https://aclanthology.org/S14-1002},
      year = {2014},
    }

    ```




#### STS15

SemEval STS 2015 dataset

**Dataset:** [`mteb/sts15-sts`](https://huggingface.co/datasets/mteb/sts15-sts) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/S15-2010)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | Blog, News, Spoken, Web, Written | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{bicici-2015-rtm,
      address = {Denver, Colorado},
      author = {Bi{\c{c}}ici, Ergun},
      booktitle = {Proceedings of the 9th International Workshop on Semantic Evaluation ({S}em{E}val 2015)},
      doi = {10.18653/v1/S15-2010},
      editor = {Nakov, Preslav  and
    Zesch, Torsten  and
    Cer, Daniel  and
    Jurgens, David},
      month = jun,
      pages = {56--63},
      publisher = {Association for Computational Linguistics},
      title = {{RTM}-{DCU}: Predicting Semantic Similarity with Referential Translation Machines},
      url = {https://aclanthology.org/S15-2010},
      year = {2015},
    }

    ```




#### STS16

SemEval-2016 Task 4

**Dataset:** [`mteb/sts16-sts`](https://huggingface.co/datasets/mteb/sts16-sts) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/S16-1001)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | Blog, Spoken, Web | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{nakov-etal-2016-semeval,
      address = {San Diego, California},
      author = {Nakov, Preslav  and
    Ritter, Alan  and
    Rosenthal, Sara  and
    Sebastiani, Fabrizio  and
    Stoyanov, Veselin},
      booktitle = {Proceedings of the 10th International Workshop on Semantic Evaluation ({S}em{E}val-2016)},
      doi = {10.18653/v1/S16-1001},
      editor = {Bethard, Steven  and
    Carpuat, Marine  and
    Cer, Daniel  and
    Jurgens, David  and
    Nakov, Preslav  and
    Zesch, Torsten},
      month = jun,
      pages = {1--18},
      publisher = {Association for Computational Linguistics},
      title = {{S}em{E}val-2016 Task 4: Sentiment Analysis in {T}witter},
      url = {https://aclanthology.org/S16-1001},
      year = {2016},
    }

    ```




#### STS17

Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation

**Dataset:** [`mteb/sts17-crosslingual-sts`](https://huggingface.co/datasets/mteb/sts17-crosslingual-sts) • **License:** not specified • [Learn more →](https://alt.qcri.org/semeval2017/task1/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | ara, deu, eng, fra, ita, ... (9) | News, Web, Written | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{cer-etal-2017-semeval,
      address = {Vancouver, Canada},
      author = {Cer, Daniel  and
    Diab, Mona  and
    Agirre, Eneko  and
    Lopez-Gazpio, I{\\~n}igo  and
    Specia, Lucia},
      booktitle = {Proceedings of the 11th International Workshop on Semantic Evaluation ({S}em{E}val-2017)},
      doi = {10.18653/v1/S17-2001},
      editor = {Bethard, Steven  and
    Carpuat, Marine  and
    Apidianaki, Marianna  and
    Mohammad, Saif M.  and
    Cer, Daniel  and
    Jurgens, David},
      month = aug,
      pages = {1--14},
      publisher = {Association for Computational Linguistics},
      title = {{S}em{E}val-2017 Task 1: Semantic Textual Similarity Multilingual and Crosslingual Focused Evaluation},
      url = {https://aclanthology.org/S17-2001},
      year = {2017},
    }

    ```




#### STS22

SemEval 2022 Task 8: Multilingual News Article Similarity

**Dataset:** [`mteb/sts22-crosslingual-sts`](https://huggingface.co/datasets/mteb/sts22-crosslingual-sts) • **License:** not specified • [Learn more →](https://competitions.codalab.org/competitions/33835)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | ara, cmn, deu, eng, fra, ... (10) | News, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{chen-etal-2022-semeval,
      address = {Seattle, United States},
      author = {Chen, Xi  and
    Zeynali, Ali  and
    Camargo, Chico  and
    Fl{\"o}ck, Fabian  and
    Gaffney, Devin  and
    Grabowicz, Przemyslaw  and
    Hale, Scott  and
    Jurgens, David  and
    Samory, Mattia},
      booktitle = {Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)},
      doi = {10.18653/v1/2022.semeval-1.155},
      editor = {Emerson, Guy  and
    Schluter, Natalie  and
    Stanovsky, Gabriel  and
    Kumar, Ritesh  and
    Palmer, Alexis  and
    Schneider, Nathan  and
    Singh, Siddharth  and
    Ratan, Shyam},
      month = jul,
      pages = {1094--1106},
      publisher = {Association for Computational Linguistics},
      title = {{S}em{E}val-2022 Task 8: Multilingual news article similarity},
      url = {https://aclanthology.org/2022.semeval-1.155},
      year = {2022},
    }

    ```




#### STS22.v2

SemEval 2022 Task 8: Multilingual News Article Similarity. Version 2 filters updated on STS22 by removing pairs where one of entries contain empty sentences.

**Dataset:** [`mteb/sts22-crosslingual-sts`](https://huggingface.co/datasets/mteb/sts22-crosslingual-sts) • **License:** not specified • [Learn more →](https://competitions.codalab.org/competitions/33835)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | ara, cmn, deu, eng, fra, ... (10) | News, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{chen-etal-2022-semeval,
      address = {Seattle, United States},
      author = {Chen, Xi  and
    Zeynali, Ali  and
    Camargo, Chico  and
    Fl{\"o}ck, Fabian  and
    Gaffney, Devin  and
    Grabowicz, Przemyslaw  and
    Hale, Scott  and
    Jurgens, David  and
    Samory, Mattia},
      booktitle = {Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)},
      doi = {10.18653/v1/2022.semeval-1.155},
      editor = {Emerson, Guy  and
    Schluter, Natalie  and
    Stanovsky, Gabriel  and
    Kumar, Ritesh  and
    Palmer, Alexis  and
    Schneider, Nathan  and
    Singh, Siddharth  and
    Ratan, Shyam},
      month = jul,
      pages = {1094--1106},
      publisher = {Association for Computational Linguistics},
      title = {{S}em{E}val-2022 Task 8: Multilingual news article similarity},
      url = {https://aclanthology.org/2022.semeval-1.155},
      year = {2022},
    }

    ```




#### STSB

A Chinese dataset for textual relatedness

**Dataset:** [`C-MTEB/STSB`](https://huggingface.co/datasets/C-MTEB/STSB) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.emnlp-main.357)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | cmn | not specified | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @misc{xiao2024cpackpackagedresourcesadvance,
      archiveprefix = {arXiv},
      author = {Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff and Defu Lian and Jian-Yun Nie},
      eprint = {2309.07597},
      primaryclass = {cs.CL},
      title = {C-Pack: Packaged Resources To Advance General Chinese Embedding},
      url = {https://arxiv.org/abs/2309.07597},
      year = {2024},
    }

    ```




#### STSBenchmark

Semantic Textual Similarity Benchmark (STSbenchmark) dataset.

**Dataset:** [`mteb/stsbenchmark-sts`](https://huggingface.co/datasets/mteb/stsbenchmark-sts) • **License:** not specified • [Learn more →](https://github.com/PhilipMay/stsb-multi-mt/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | eng | Blog, News, Written | human-annotated | machine-translated and verified |



??? quote "Citation"


    ```bibtex

    @inproceedings{huggingface:dataset:stsb_multi_mt,
      author = {Philip May},
      title = {Machine translated multilingual STS benchmark dataset.},
      url = {https://github.com/PhilipMay/stsb-multi-mt},
      year = {2021},
    }

    ```




#### STSBenchmark-VN

A translated dataset from Semantic Textual Similarity Benchmark (STSbenchmark) dataset.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/stsbenchmark-sts-vn`](https://huggingface.co/datasets/GreenNode/stsbenchmark-sts-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/PhilipMay/stsb-multi-mt/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | cosine_spearman | vie | Blog, News, Written | derived | machine-translated and LM verified |



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




#### STSBenchmarkMultilingualSTS

Semantic Textual Similarity Benchmark (STSbenchmark) dataset, but translated using DeepL API.

**Dataset:** [`mteb/stsb_multi_mt`](https://huggingface.co/datasets/mteb/stsb_multi_mt) • **License:** not specified • [Learn more →](https://github.com/PhilipMay/stsb-multi-mt/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | cmn, deu, eng, fra, ita, ... (10) | News, Social, Spoken, Web, Written | human-annotated | machine-translated |



??? quote "Citation"


    ```bibtex

    @inproceedings{huggingface:dataset:stsb_multi_mt,
      author = {Philip May},
      title = {Machine translated multilingual STS benchmark dataset.},
      url = {https://github.com/PhilipMay/stsb-multi-mt},
      year = {2021},
    }

    ```




#### STSES

Spanish test sets from SemEval-2014 (Agirre et al., 2014) and SemEval-2015 (Agirre et al., 2015)

**Dataset:** [`mteb/STSES`](https://huggingface.co/datasets/mteb/STSES) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/PlanTL-GOB-ES/sts-es)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | spa | Written | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @inproceedings{agirre2014semeval,
      author = {Agirre, Eneko and Banea, Carmen and Cardie, Claire and Cer, Daniel M and Diab, Mona T and Gonzalez-Agirre, Aitor and Guo, Weiwei and Mihalcea, Rada and Rigau, German and Wiebe, Janyce},
      booktitle = {SemEval@ COLING},
      pages = {81--91},
      title = {SemEval-2014 Task 10: Multilingual Semantic Textual Similarity.},
      year = {2014},
    }

    @inproceedings{agirre2015semeval,
      author = {Agirre, Eneko and Banea, Carmen and Cardie, Claire and Cer, Daniel and Diab, Mona and Gonzalez-Agirre, Aitor and Guo, Weiwei and Lopez-Gazpio, Inigo and Maritxalar, Montse and Mihalcea, Rada and others},
      booktitle = {Proceedings of the 9th international workshop on semantic evaluation (SemEval 2015)},
      pages = {252--263},
      title = {Semeval-2015 task 2: Semantic textual similarity, english, spanish and pilot on interpretability},
      year = {2015},
    }

    ```




#### SemRel24STS

SemRel2024 is a collection of Semantic Textual Relatedness (STR) datasets for 14 languages, including African and Asian languages. The datasets are composed of sentence pairs, each assigned a relatedness score between 0 (completely) unrelated and 1 (maximally related) with a large range of expected relatedness values.

**Dataset:** [`SemRel/SemRel2024`](https://huggingface.co/datasets/SemRel/SemRel2024) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/SemRel/SemRel2024)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | afr, amh, arb, arq, ary, ... (12) | Spoken, Written | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @misc{ousidhoum2024semrel2024,
      archiveprefix = {arXiv},
      author = {Nedjma Ousidhoum and Shamsuddeen Hassan Muhammad and Mohamed Abdalla and Idris Abdulmumin and Ibrahim Said Ahmad and
    Sanchit Ahuja and Alham Fikri Aji and Vladimir Araujo and Abinew Ali Ayele and Pavan Baswani and Meriem Beloucif and
    Chris Biemann and Sofia Bourhim and Christine De Kock and Genet Shanko Dekebo and
    Oumaima Hourrane and Gopichand Kanumolu and Lokesh Madasu and Samuel Rutunda and Manish Shrivastava and
    Thamar Solorio and Nirmal Surange and Hailegnaw Getaneh Tilaye and Krishnapriya Vishnubhotla and Genta Winata and
    Seid Muhie Yimam and Saif M. Mohammad},
      eprint = {2402.08638},
      primaryclass = {cs.CL},
      title = {SemRel2024: A Collection of Semantic Textual Relatedness Datasets for 14 Languages},
      year = {2024},
    }

    ```




#### SynPerSTS

Synthetic Persian Semantic Textual Similarity Dataset

**Dataset:** [`MCINext/synthetic-persian-sts`](https://huggingface.co/datasets/MCINext/synthetic-persian-sts) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | cosine_spearman | fas | Blog, News, Religious, Web | LM-generated | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    ```
