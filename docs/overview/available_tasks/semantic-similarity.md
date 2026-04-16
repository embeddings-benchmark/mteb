---
icon: lucide/spline
title: "Semantic Similarity"
---

<style>
.nowrap-table th {
  white-space: nowrap;
}
</style>

# Semantic Similarity

Tasks that evaluated examines the semantic similarity between correctly and incorrectly paired items.

<!-- The following sections are auto-generated, please edit the construction script -->

<!-- START-TASKS -->


## BitextMining

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 31

#### `BUCC` { .model-copy }

BUCC bitext mining dataset train split.

**Dataset:** [`mteb/BUCC`](https://huggingface.co/datasets/mteb/BUCC) • **License:** not specified • [Learn more →](https://comparable.limsi.fr/bucc2018/bucc2018-task.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn, deu, eng, fra, rus | Written | human-annotated | human-translated | f1 |



??? quote "Citation"


    ```bibtex

    @inproceedings{zweigenbaum-etal-2017-overview,
      address = {Vancouver, Canada},
      author = {Zweigenbaum, Pierre  and
    Sharoff, Serge  and
    Rapp, Reinhard},
      booktitle = {Proceedings of the 10th Workshop on Building and Using Comparable Corpora},
      doi = {10.18653/v1/W17-2512},
      editor = {Sharoff, Serge  and
    Zweigenbaum, Pierre  and
    Rapp, Reinhard},
      month = aug,
      pages = {60--67},
      publisher = {Association for Computational Linguistics},
      title = {Overview of the Second {BUCC} Shared Task: Spotting Parallel Sentences in Comparable Corpora},
      url = {https://aclanthology.org/W17-2512},
      year = {2017},
    }

    ```




#### `BUCC.v2` { .model-copy }

BUCC bitext mining dataset train split, gold set only.

**Dataset:** [`mteb/bucc-bitext-mining`](https://huggingface.co/datasets/mteb/bucc-bitext-mining) • **License:** not specified • [Learn more →](https://comparable.limsi.fr/bucc2018/bucc2018-task.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn, deu, eng, fra, rus | Written | human-annotated | human-translated | f1 |



??? quote "Citation"


    ```bibtex

    @inproceedings{zweigenbaum-etal-2017-overview,
      address = {Vancouver, Canada},
      author = {Zweigenbaum, Pierre  and
    Sharoff, Serge  and
    Rapp, Reinhard},
      booktitle = {Proceedings of the 10th Workshop on Building and Using Comparable Corpora},
      doi = {10.18653/v1/W17-2512},
      editor = {Sharoff, Serge  and
    Zweigenbaum, Pierre  and
    Rapp, Reinhard},
      month = aug,
      pages = {60--67},
      publisher = {Association for Computational Linguistics},
      title = {Overview of the Second {BUCC} Shared Task: Spotting Parallel Sentences in Comparable Corpora},
      url = {https://aclanthology.org/W17-2512},
      year = {2017},
    }

    ```




#### `BibleNLPBitextMining` { .model-copy }

Partial Bible translations in 829 languages, aligned by verse.

**Dataset:** [`mteb/biblenlp-corpus`](https://huggingface.co/datasets/mteb/biblenlp-corpus) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2304.09919)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | aai, aak, aau, aaz, abt, ... (829) | Religious, Written | expert-annotated | created | f1 |



??? quote "Citation"


    ```bibtex

    @article{akerman2023ebible,
      author = {Akerman, Vesa and Baines, David and Daspit, Damien and Hermjakob, Ulf and Jang, Taeho and Leong, Colin and Martin, Michael and Mathew, Joel and Robie, Jonathan and Schwarting, Marcus},
      journal = {arXiv preprint arXiv:2304.09919},
      title = {The eBible Corpus: Data and Model Benchmarks for Bible Translation for Low-Resource Languages},
      year = {2023},
    }

    ```




#### `BornholmBitextMining` { .model-copy }

Danish Bornholmsk Parallel Corpus. Bornholmsk is a Danish dialect spoken on the island of Bornholm, Denmark. Historically it is a part of east Danish which was also spoken in Scania and Halland, Sweden.

**Dataset:** [`mteb/BornholmBitextMining`](https://huggingface.co/datasets/mteb/BornholmBitextMining) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/W19-6138/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | dan | Fiction, Social, Web, Written | expert-annotated | created | f1 |



??? quote "Citation"


    ```bibtex

    @inproceedings{derczynskiBornholmskNaturalLanguage2019,
      author = {Derczynski, Leon and Kjeldsen, Alex Speed},
      booktitle = {Proceedings of the Nordic Conference of Computational Linguistics (2019)},
      date = {2019},
      file = {Available Version (via Google Scholar):/Users/au554730/Zotero/storage/FBQ73ZYN/Derczynski and Kjeldsen - 2019 - Bornholmsk natural language processing Resources .pdf:application/pdf},
      pages = {338--344},
      publisher = {Linköping University Electronic Press},
      shorttitle = {Bornholmsk natural language processing},
      title = {Bornholmsk natural language processing: Resources and tools},
      url = {https://pure.itu.dk/ws/files/84551091/W19_6138.pdf},
      urldate = {2024-04-24},
    }

    ```




#### `DanishMedicinesAgencyBitextMining` { .model-copy }

A Bilingual English-Danish parallel corpus from The Danish Medicines Agency.

**Dataset:** [`mteb/english-danish-parallel-corpus`](https://huggingface.co/datasets/mteb/english-danish-parallel-corpus) • **License:** https://opendefinition.org/od/2.1/en/ • [Learn more →](https://sprogteknologi.dk/dataset/bilingual-english-danish-parallel-corpus-from-the-danish-medicines-agency)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | dan, eng | Medical, Written | human-annotated | found | f1 |



??? quote "Citation"


    ```bibtex

    @misc{elrc_danish_medicines_agency_2018,
      author = {Rozis, Roberts},
      institution = {European Union},
      license = {Open Under-PSI},
      note = {Dataset created within the European Language Resource Coordination (ELRC) project under the Connecting Europe Facility - Automated Translation (CEF.AT) actions SMART 2014/1074 and SMART 2015/1091.},
      title = {Bilingual English-Danish Parallel Corpus from the Danish Medicines Agency},
      url = {https://sprogteknologi.dk/dataset/bilingual-english-danish-parallel-corpus-from-the-danish-medicines-agency},
      year = {2019},
    }

    ```




#### `DiaBlaBitextMining` { .model-copy }

English-French Parallel Corpus. DiaBLa is an English-French dataset for the evaluation of Machine Translation (MT) for informal, written bilingual dialogue.

**Dataset:** [`mteb/DiaBlaBitextMining`](https://huggingface.co/datasets/mteb/DiaBlaBitextMining) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://inria.hal.science/hal-03021633)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, fra | Social, Written | human-annotated | created | f1 |



??? quote "Citation"


    ```bibtex

    @inproceedings{gonzalez2019diabla,
      author = {González, Matilde and García, Clara and Sánchez, Lucía},
      booktitle = {Proceedings of the 12th Language Resources and Evaluation Conference},
      pages = {4192--4198},
      title = {DiaBLa: A Corpus of Bilingual Spontaneous Written Dialogues for Machine Translation},
      year = {2019},
    }

    ```




#### `FloresBitextMining` { .model-copy }

FLORES is a benchmark dataset for machine translation between English and low-resource languages.

**Dataset:** [`mteb/FloresBitextMining`](https://huggingface.co/datasets/mteb/FloresBitextMining) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/facebook/flores)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ace, acm, acq, aeb, afr, ... (196) | Encyclopaedic, Non-fiction, Written | human-annotated | created | f1 |



??? quote "Citation"


    ```bibtex

    @inproceedings{goyal2022flores,
      author = {Goyal, Naman and Gao, Cynthia and Chaudhary, Vishrav and Chen, Peng-Jen and Wenzek, Guillaume and Ju, Da and Krishnan, Sanjana and Ranzato, Marc'Aurelio and Guzm{\'a}n, Francisco},
      booktitle = {Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
      pages = {19--35},
      title = {The FLORES-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation},
      year = {2022},
    }

    ```




#### `IN22ConvBitextMining` { .model-copy }

IN22-Conv is a n-way parallel conversation domain benchmark dataset for machine translation spanning English and 22 Indic languages.

**Dataset:** [`mteb/IN22ConvBitextMining`](https://huggingface.co/datasets/mteb/IN22ConvBitextMining) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/ai4bharat/IN22-Conv)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | asm, ben, brx, doi, eng, ... (23) | Fiction, Social, Spoken, Spoken | expert-annotated | created | f1 |



??? quote "Citation"


    ```bibtex

    @article{gala2023indictrans,
      author = {Jay Gala and Pranjal A Chitale and A K Raghavan and Varun Gumma and Sumanth Doddapaneni and Aswanth Kumar M and Janki Atul Nawale and Anupama Sujatha and Ratish Puduppully and Vivek Raghavan and Pratyush Kumar and Mitesh M Khapra and Raj Dabre and Anoop Kunchukuttan},
      issn = {2835-8856},
      journal = {Transactions on Machine Learning Research},
      note = {},
      title = {IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages},
      url = {https://openreview.net/forum?id=vfT4YuzAYA},
      year = {2023},
    }

    ```




#### `IN22GenBitextMining` { .model-copy }

IN22-Gen is a n-way parallel general-purpose multi-domain benchmark dataset for machine translation spanning English and 22 Indic languages.

**Dataset:** [`mteb/IN22GenBitextMining`](https://huggingface.co/datasets/mteb/IN22GenBitextMining) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/ai4bharat/IN22-Gen)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | asm, ben, brx, doi, eng, ... (23) | Government, Legal, News, Non-fiction, Religious, ... (7) | expert-annotated | created | f1 |



??? quote "Citation"


    ```bibtex

    @article{gala2023indictrans,
      author = {Jay Gala and Pranjal A Chitale and A K Raghavan and Varun Gumma and Sumanth Doddapaneni and Aswanth Kumar M and Janki Atul Nawale and Anupama Sujatha and Ratish Puduppully and Vivek Raghavan and Pratyush Kumar and Mitesh M Khapra and Raj Dabre and Anoop Kunchukuttan},
      issn = {2835-8856},
      journal = {Transactions on Machine Learning Research},
      note = {},
      title = {IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages},
      url = {https://openreview.net/forum?id=vfT4YuzAYA},
      year = {2023},
    }

    ```




#### `IWSLT2017BitextMining` { .model-copy }

The IWSLT 2017 Multilingual Task addresses text translation, including zero-shot translation, with a single MT system across all directions including English, German, Dutch, Italian and Romanian.

**Dataset:** [`mteb/IWSLT2017BitextMining`](https://huggingface.co/datasets/mteb/IWSLT2017BitextMining) • **License:** cc-by-nc-nd-4.0 • [Learn more →](https://aclanthology.org/2017.iwslt-1.1/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara, cmn, deu, eng, fra, ... (10) | Fiction, Non-fiction, Written | expert-annotated | found | f1 |



??? quote "Citation"


    ```bibtex

    @inproceedings{cettolo-etal-2017-overview,
      address = {Tokyo, Japan},
      author = {Cettolo, Mauro  and
    Federico, Marcello  and
    Bentivogli, Luisa  and
    Niehues, Jan  and
    St{\"u}ker, Sebastian  and
    Sudoh, Katsuhito  and
    Yoshino, Koichiro  and
    Federmann, Christian},
      booktitle = {Proceedings of the 14th International Conference on Spoken Language Translation},
      editor = {Sakti, Sakriani  and
    Utiyama, Masao},
      month = dec # { 14-15},
      pages = {2--14},
      publisher = {International Workshop on Spoken Language Translation},
      title = {Overview of the {IWSLT} 2017 Evaluation Campaign},
      url = {https://aclanthology.org/2017.iwslt-1.1},
      year = {2017},
    }

    ```




#### `IndicGenBenchFloresBitextMining` { .model-copy }

Flores-IN dataset is an extension of Flores dataset released as a part of the IndicGenBench by Google

**Dataset:** [`mteb/IndicGenBenchFloresBitextMining`](https://huggingface.co/datasets/mteb/IndicGenBenchFloresBitextMining) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/google-research-datasets/indic-gen-bench/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | asm, awa, ben, bgc, bho, ... (30) | News, Web, Written | expert-annotated | human-translated and localized | f1 |



??? quote "Citation"


    ```bibtex

    @misc{singh2024indicgenbench,
      archiveprefix = {arXiv},
      author = {Harman Singh and Nitish Gupta and Shikhar Bharadwaj and Dinesh Tewari and Partha Talukdar},
      eprint = {2404.16816},
      primaryclass = {cs.CL},
      title = {IndicGenBench: A Multilingual Benchmark to Evaluate Generation Capabilities of LLMs on Indic Languages},
      year = {2024},
    }

    ```




#### `LinceMTBitextMining` { .model-copy }

LinceMT is a parallel corpus for machine translation pairing code-mixed Hinglish (a fusion of Hindi and English commonly used in modern India) with human-generated English translations.

**Dataset:** [`gentaiscool/bitext_lincemt_miners`](https://huggingface.co/datasets/gentaiscool/bitext_lincemt_miners) • **License:** not specified • [Learn more →](https://ritual.uh.edu/lince/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, hin | Social, Written | human-annotated | found | f1 |



??? quote "Citation"


    ```bibtex

    @inproceedings{aguilar2020lince,
      author = {Aguilar, Gustavo and Kar, Sudipta and Solorio, Thamar},
      booktitle = {Proceedings of the Twelfth Language Resources and Evaluation Conference},
      pages = {1803--1813},
      title = {LinCE: A Centralized Benchmark for Linguistic Code-switching Evaluation},
      year = {2020},
    }

    ```




#### `NTREXBitextMining` { .model-copy }

NTREX is a News Test References dataset for Machine Translation Evaluation, covering translation from English into 128 languages. We select language pairs according to the M2M-100 language grouping strategy, resulting in 1916 directions.

**Dataset:** [`mteb/NTREXBitextMining`](https://huggingface.co/datasets/mteb/NTREXBitextMining) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/davidstap/NTREX)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | afr, amh, arb, aze, bak, ... (119) | News, Written | expert-annotated | human-translated and localized | f1 |



??? quote "Citation"


    ```bibtex

    @inproceedings{federmann-etal-2022-ntrex,
      address = {Online},
      author = {Federmann, Christian and Kocmi, Tom and Xin, Ying},
      booktitle = {Proceedings of the First Workshop on Scaling Up Multilingual Evaluation},
      month = {nov},
      pages = {21--24},
      publisher = {Association for Computational Linguistics},
      title = {{NTREX}-128 {--} News Test References for {MT} Evaluation of 128 Languages},
      url = {https://aclanthology.org/2022.sumeval-1.4},
      year = {2022},
    }

    ```




#### `NollySentiBitextMining` { .model-copy }

NollySenti is Nollywood movie reviews for five languages widely spoken in Nigeria (English, Hausa, Igbo, Nigerian-Pidgin, and Yoruba.

**Dataset:** [`mteb/NollySentiBitextMining`](https://huggingface.co/datasets/mteb/NollySentiBitextMining) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/IyanuSh/NollySenti)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, hau, ibo, pcm, yor | Reviews, Social, Written | human-annotated | found | f1 |



??? quote "Citation"


    ```bibtex

    @inproceedings{shode2023nollysenti,
      author = {Shode, Iyanuoluwa and Adelani, David Ifeoluwa and Peng, Jing and Feldman, Anna},
      booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
      pages = {986--998},
      title = {NollySenti: Leveraging Transfer Learning and Machine Translation for Nigerian Movie Sentiment Classification},
      year = {2023},
    }

    ```




#### `NorwegianCourtsBitextMining` { .model-copy }

Nynorsk and Bokmål parallel corpus from Norwegian courts. Norwegian courts have two standardised written languages. Bokmål is a variant closer to Danish, while Nynorsk was created to resemble regional dialects of Norwegian.

**Dataset:** [`mteb/NorwegianCourtsBitextMining`](https://huggingface.co/datasets/mteb/NorwegianCourtsBitextMining) • **License:** cc-by-4.0 • [Learn more →](https://opus.nlpl.eu/index.php)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nno, nob | Legal, Written | human-annotated | found | f1 |



??? quote "Citation"


    ```bibtex

    @inproceedings{opus4,
      author = {Tiedemann, J{\"o}rg and Thottingal, Santhosh},
      booktitle = {Proceedings of the 22nd Annual Conference of the European Association for Machine Translation (EAMT)},
      title = {OPUS-MT — Building open translation services for the World},
      year = {2020},
    }

    ```




#### `NusaTranslationBitextMining` { .model-copy }

NusaTranslation is a parallel dataset for machine translation on 11 Indonesia languages and English.

**Dataset:** [`mteb/NusaTranslationBitextMining`](https://huggingface.co/datasets/mteb/NusaTranslationBitextMining) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/indonlp/nusatranslation_mt)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | abs, bbc, bew, bhp, ind, ... (12) | Social, Written | human-annotated | created | f1 |



??? quote "Citation"


    ```bibtex

    @inproceedings{cahyawijaya-etal-2023-nusawrites,
      address = {Nusa Dua, Bali},
      author = {Cahyawijaya, Samuel  and  Lovenia, Holy  and Koto, Fajri  and  Adhista, Dea  and  Dave, Emmanuel  and  Oktavianti, Sarah  and  Akbar, Salsabil  and  Lee, Jhonson  and  Shadieq, Nuur  and  Cenggoro, Tjeng Wawan  and  Linuwih, Hanung  and  Wilie, Bryan  and  Muridan, Galih  and  Winata, Genta  and  Moeljadi, David  and  Aji, Alham Fikri  and  Purwarianti, Ayu  and  Fung, Pascale},
      booktitle = {Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)},
      editor = {Park, Jong C.  and  Arase, Yuki  and  Hu, Baotian  and  Lu, Wei  and  Wijaya, Derry  and  Purwarianti, Ayu  and  Krisnadhi, Adila Alfa},
      month = nov,
      pages = {921--945},
      publisher = {Association for Computational Linguistics},
      title = {NusaWrites: Constructing High-Quality Corpora for Underrepresented and Extremely Low-Resource Languages},
      url = {https://aclanthology.org/2023.ijcnlp-main.60},
      year = {2023},
    }

    ```




#### `NusaXBitextMining` { .model-copy }

NusaX is a parallel dataset for machine translation and sentiment analysis on 11 Indonesia languages and English.

**Dataset:** [`mteb/NusaXBitextMining`](https://huggingface.co/datasets/mteb/NusaXBitextMining) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/indonlp/NusaX-senti/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ace, ban, bbc, bjn, bug, ... (12) | Reviews, Written | human-annotated | created | f1 |



??? quote "Citation"


    ```bibtex

    @inproceedings{winata2023nusax,
      author = {Winata, Genta Indra and Aji, Alham Fikri and Cahyawijaya, Samuel and Mahendra, Rahmad and Koto, Fajri and Romadhony, Ade and Kurniawan, Kemal and Moeljadi, David and Prasojo, Radityo Eko and Fung, Pascale and others},
      booktitle = {Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics},
      pages = {815--834},
      title = {NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages},
      year = {2023},
    }

    @misc{winata2024miners,
      archiveprefix = {arXiv},
      author = {Genta Indra Winata and Ruochen Zhang and David Ifeoluwa Adelani},
      eprint = {2406.07424},
      primaryclass = {cs.CL},
      title = {MINERS: Multilingual Language Models as Semantic Retrievers},
      year = {2024},
    }

    ```




#### `PhincBitextMining` { .model-copy }

Phinc is a parallel corpus for machine translation pairing code-mixed Hinglish (a fusion of Hindi and English commonly used in modern India) with human-generated English translations.

**Dataset:** [`gentaiscool/bitext_phinc_miners`](https://huggingface.co/datasets/gentaiscool/bitext_phinc_miners) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/veezbo/phinc)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, hin | Social, Written | human-annotated | found | f1 |



??? quote "Citation"


    ```bibtex

    @inproceedings{srivastava2020phinc,
      author = {Srivastava, Vivek and Singh, Mayank},
      booktitle = {Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020)},
      pages = {41--49},
      title = {PHINC: A Parallel Hinglish Social Media Code-Mixed Corpus for Machine Translation},
      year = {2020},
    }

    ```




#### `PubChemSMILESBitextMining` { .model-copy }

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/PubChemSMILESBitextMining`](https://huggingface.co/datasets/BASF-AI/PubChemSMILESBitextMining) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Chemistry | derived | created | f1 |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    @article{kim2023pubchem,
      author = {Kim, Sunghwan and Chen, Jie and Cheng, Tiejun and Gindulyte, Asta and He, Jia and He, Siqian and Li, Qingliang and Shoemaker, Benjamin A and Thiessen, Paul A and Yu, Bo and others},
      journal = {Nucleic acids research},
      number = {D1},
      pages = {D1373--D1380},
      publisher = {Oxford University Press},
      title = {PubChem 2023 update},
      volume = {51},
      year = {2023},
    }

    ```




#### `RomaTalesBitextMining` { .model-copy }

Parallel corpus of Roma Tales in Lovari with Hungarian translations.

**Dataset:** [`kardosdrur/roma-tales`](https://huggingface.co/datasets/kardosdrur/roma-tales) • **License:** not specified • [Learn more →](https://idoc.pub/documents/idocpub-zpnxm9g35ylv)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | hun, rom | Fiction, Written | expert-annotated | created | f1 |



#### `RuSciBenchBitextMining` { .model-copy }

This task focuses on finding translations of scientific articles. The dataset is sourced from eLibrary, Russia's largest electronic library of scientific publications. Russian authors often provide English translations for their abstracts and titles, and the data consists of these paired titles and abstracts. The task evaluates a model's ability to match an article's Russian title and abstract to its English counterpart, or vice versa.

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_bitext_mining`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_bitext_mining) • **License:** not specified • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to category (t2c) | eng, rus | Academic, Non-fiction, Written | derived | found | f1 |



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




#### `RuSciBenchBitextMining.v2` { .model-copy }

This task focuses on finding translations of scientific articles. The dataset is sourced from eLibrary, Russia's largest electronic library of scientific publications. Russian authors often provide English translations for their abstracts and titles, and the data consists of these paired titles and abstracts. The task evaluates a model's ability to match an article's Russian title and abstract to its English counterpart, or vice versa. Compared to the previous version, 6 erroneous examples have been removed.

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_bitext_mining`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_bitext_mining) • **License:** not specified • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to category (t2c) | eng, rus | Academic, Non-fiction, Written | derived | found | f1 |



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




#### `SAMSumFa` { .model-copy }

Translated Version of SAMSum Dataset for summary retrieval.

**Dataset:** [`MCINext/samsum-fa`](https://huggingface.co/datasets/MCINext/samsum-fa) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/samsum-fa)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Spoken | LM-generated | machine-translated | f1 |



#### `SRNCorpusBitextMining` { .model-copy }

SRNCorpus is a machine translation corpus for creole language Sranantongo and Dutch.

**Dataset:** [`mteb/SRNCorpusBitextMining`](https://huggingface.co/datasets/mteb/SRNCorpusBitextMining) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2212.06383)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld, srn | Social, Web, Written | human-annotated | found | f1 |



??? quote "Citation"


    ```bibtex

    @article{zwennicker2022towards,
      author = {Zwennicker, Just and Stap, David},
      journal = {arXiv preprint arXiv:2212.06383},
      title = {Towards a general purpose machine translation system for Sranantongo},
      year = {2022},
    }

    ```




#### `SynPerChatbotRAGSumSRetrieval` { .model-copy }

Synthetic Persian Chatbot RAG Summary Dataset for summary retrieval.

**Dataset:** [`MCINext/synthetic-persian-chatbot-rag-summary-retrieval`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-summary-retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-summary-retrieval)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Spoken | LM-generated | LM-generated and verified | f1 |



??? quote "Citation"


    ```bibtex

    ```




#### `SynPerChatbotSumSRetrieval` { .model-copy }

Synthetic Persian Chatbot Summary Dataset for summary retrieval.

**Dataset:** [`MCINext/synthetic-persian-chatbot-summary-retrieval`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-summary-retrieval) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-summary-retrieval)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Spoken | LM-generated | LM-generated and verified | f1 |



??? quote "Citation"


    ```bibtex

    ```




#### `Tatoeba` { .model-copy }

1,000 English-aligned sentence pairs for each language based on the Tatoeba corpus

**Dataset:** [`mteb/tatoeba-bitext-mining`](https://huggingface.co/datasets/mteb/tatoeba-bitext-mining) • **License:** cc-by-2.0 • [Learn more →](https://github.com/facebookresearch/LASER/tree/main/data/tatoeba/v1)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | afr, amh, ang, ara, arq, ... (113) | Written | human-annotated | found | f1 |



??? quote "Citation"


    ```bibtex

    @misc{tatoeba,
      author = {Tatoeba community},
      title = {Tatoeba: Collection of sentences and translations},
      year = {2021},
    }

    ```




#### `TbilisiCityHallBitextMining` { .model-copy }

Parallel news titles from the Tbilisi City Hall website (https://tbilisi.gov.ge/).

**Dataset:** [`jupyterjazz/tbilisi-city-hall-titles`](https://huggingface.co/datasets/jupyterjazz/tbilisi-city-hall-titles) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/jupyterjazz/tbilisi-city-hall-titles)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, kat | News, Written | derived | created | f1 |



#### `VieMedEVBitextMining` { .model-copy }

A high-quality Vietnamese-English parallel data from the medical domain for machine translation

**Dataset:** [`mteb/VieMedEVBitextMining`](https://huggingface.co/datasets/mteb/VieMedEVBitextMining) • **License:** cc-by-nc-4.0 • [Learn more →](https://aclanthology.org/2015.iwslt-evaluation.11/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng, vie | Medical, Written | expert-annotated | human-translated and localized | f1 |



??? quote "Citation"


    ```bibtex

    @inproceedings{medev,
      author = {Nhu Vo and Dat Quoc Nguyen and Dung D. Le and Massimo Piccardi and Wray Buntine},
      booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING)},
      title = {{Improving Vietnamese-English Medical Machine Translation}},
      year = {2024},
    }

    ```




#### `WebFAQBitextMiningQAs` { .model-copy }

The WebFAQ Bitext Dataset consists of natural FAQ-style Question-Answer pairs that align across languages. A sentence in the "WebFAQBitextMiningQAs" task is a concatenation of a question and its corresponding answer. The dataset is sourced from FAQ pages on the web.

**Dataset:** [`PaDaS-Lab/webfaq-bitexts`](https://huggingface.co/datasets/PaDaS-Lab/webfaq-bitexts) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/PaDaS-Lab)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara, aze, ben, bul, cat, ... (49) | Web, Written | human-annotated | human-translated | f1 |



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




#### `WebFAQBitextMiningQuestions` { .model-copy }

The WebFAQ Bitext Dataset consists of natural FAQ-style Question-Answer pairs that align across languages. A sentence in the "WebFAQBitextMiningQuestions" task is the question originating from an aligned QA. The dataset is sourced from FAQ pages on the web.

**Dataset:** [`PaDaS-Lab/webfaq-bitexts`](https://huggingface.co/datasets/PaDaS-Lab/webfaq-bitexts) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/PaDaS-Lab)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara, aze, ben, bul, cat, ... (49) | Web, Written | human-annotated | human-translated | f1 |



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




## STS

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 49

#### `AFQMC` { .model-copy }

A Chinese dataset for textual relatedness

**Dataset:** [`C-MTEB/AFQMC`](https://huggingface.co/datasets/C-MTEB/AFQMC) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.emnlp-main.357)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn | Web, Written | human-annotated | found | cosine_spearman |



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




#### `ATEC` { .model-copy }

A Chinese dataset for textual relatedness

**Dataset:** [`C-MTEB/ATEC`](https://huggingface.co/datasets/C-MTEB/ATEC) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.emnlp-main.357)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn | Web, Written | human-annotated | found | cosine_spearman |



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




#### `Assin2STS` { .model-copy }

Semantic Textual Similarity part of the ASSIN 2, an evaluation shared task collocated with STIL 2019.

**Dataset:** [`nilc-nlp/assin2`](https://huggingface.co/datasets/nilc-nlp/assin2) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-030-41505-1_39)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | por | Written | human-annotated | found | cosine_spearman |



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




#### `BIOSSES` { .model-copy }

Biomedical Semantic Similarity Estimation.

**Dataset:** [`mteb/biosses-sts`](https://huggingface.co/datasets/mteb/biosses-sts) • **License:** not specified • [Learn more →](https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Medical | derived | found | cosine_spearman |



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




#### `BIOSSES-VN` { .model-copy }

A translated dataset from Biomedical Semantic Similarity Estimation. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/biosses-sts-vn`](https://huggingface.co/datasets/GreenNode/biosses-sts-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to category (t2c) | vie | Medical | derived | machine-translated and LM verified | cosine_spearman |



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




#### `BQ` { .model-copy }

A Chinese dataset for textual relatedness

**Dataset:** [`C-MTEB/BQ`](https://huggingface.co/datasets/C-MTEB/BQ) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.emnlp-main.357)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn | Web, Written | human-annotated | found | cosine_spearman |



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




#### `CDSC-R` { .model-copy }

Compositional Distributional Semantics Corpus for textual relatedness.

**Dataset:** [`PL-MTEB/cdscr-sts`](https://huggingface.co/datasets/PL-MTEB/cdscr-sts) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/P17-1073.pdf)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Web, Written | human-annotated | human-translated and localized | cosine_spearman |



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




#### `FaroeseSTS` { .model-copy }

Semantic Text Similarity (STS) corpus for Faroese.

**Dataset:** [`mteb/FaroeseSTS`](https://huggingface.co/datasets/mteb/FaroeseSTS) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.74.pdf)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fao | News, Web, Written | human-annotated | found | cosine_spearman |



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




#### `Farsick` { .model-copy }

A Persian Semantic Textual Similarity And Natural Language Inference Dataset

**Dataset:** [`MCINext/farsick-sts`](https://huggingface.co/datasets/MCINext/farsick-sts) • **License:** not specified • [Learn more →](https://github.com/ZahraGhasemi-AI/FarSick)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | not specified | derived | found | cosine_spearman |



??? quote "Citation"


    ```bibtex

    ```




#### `FinParaSTS` { .model-copy }

Finnish paraphrase-based semantic similarity corpus

**Dataset:** [`mteb/FinParaSTS`](https://huggingface.co/datasets/mteb/FinParaSTS) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/TurkuNLP/turku_paraphrase_corpus)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fin | News, Subtitles, Written | expert-annotated | found | cosine_spearman |



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




#### `GermanSTSBenchmark` { .model-copy }

Semantic Textual Similarity Benchmark (STSbenchmark) dataset translated into German. Translations were originally done by T-Systems on site services GmbH.

**Dataset:** [`mteb/GermanSTSBenchmark`](https://huggingface.co/datasets/mteb/GermanSTSBenchmark) • **License:** cc-by-sa-3.0 • [Learn more →](https://github.com/t-systems-on-site-services-gmbh/german-STSbenchmark)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | deu | News, Web, Written | human-annotated | machine-translated | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @inproceedings{huggingface:dataset:stsb_multi_mt,
      author = {Philip May},
      title = {Machine translated multilingual STS benchmark dataset.},
      url = {https://github.com/PhilipMay/stsb-multi-mt},
      year = {2021},
    }

    ```




#### `HUMESICK-R` { .model-copy }

Human evaluation subset of Semantic Textual Similarity SICK-R dataset

**Dataset:** [`mteb/mteb-human-sickr-sts`](https://huggingface.co/datasets/mteb/mteb-human-sickr-sts) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://aclanthology.org/L14-1314/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Web, Written | human-annotated | created | cosine_spearman |



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




#### `HUMESTS12` { .model-copy }

Human evaluation subset of SemEval-2012 Task 6.

**Dataset:** [`mteb/mteb-human-sts12-sts`](https://huggingface.co/datasets/mteb/mteb-human-sts12-sts) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/S12-1051.pdf)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, News, Written | human-annotated | created | cosine_spearman |



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




#### `HUMESTS22` { .model-copy }

Human evaluation subset of SemEval 2022 Task 8: Multilingual News Article Similarity

**Dataset:** [`mteb/mteb-human-sts22-sts`](https://huggingface.co/datasets/mteb/mteb-human-sts22-sts) • **License:** not specified • [Learn more →](https://competitions.codalab.org/competitions/33835)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara, eng, fra, rus | News, Written | human-annotated | found | cosine_spearman |



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




#### `HUMESTSBenchmark` { .model-copy }

Human evaluation subset of Semantic Textual Similarity Benchmark (STSbenchmark) dataset.

**Dataset:** [`mteb/mteb-human-stsbenchmark-sts`](https://huggingface.co/datasets/mteb/mteb-human-stsbenchmark-sts) • **License:** not specified • [Learn more →](https://github.com/PhilipMay/stsb-multi-mt/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Blog, News, Written | human-annotated | machine-translated and verified | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @inproceedings{huggingface:dataset:stsb_multi_mt,
      author = {Philip May},
      title = {Machine translated multilingual STS benchmark dataset.},
      url = {https://github.com/PhilipMay/stsb-multi-mt},
      year = {2021},
    }

    ```




#### `IndicCrosslingualSTS` { .model-copy }

This is a Semantic Textual Similarity testset between English and 12 high-resource Indic languages.

**Dataset:** [`mteb/IndicCrosslingualSTS`](https://huggingface.co/datasets/mteb/IndicCrosslingualSTS) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/jaygala24/indic_sts)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | asm, ben, eng, guj, hin, ... (13) | Government, News, Non-fiction, Spoken, Spoken, ... (7) | expert-annotated | created | cosine_spearman |



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




#### `JSICK` { .model-copy }

JSICK is the Japanese NLI and STS dataset by manually translating the English dataset SICK (Marelli et al., 2014) into Japanese.

**Dataset:** [`mteb/JSICK`](https://huggingface.co/datasets/mteb/JSICK) • **License:** cc-by-4.0 • [Learn more →](https://github.com/sbintuitions/JMTEB)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Web, Written | human-annotated | found | cosine_spearman |



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




#### `JSTS` { .model-copy }

Japanese Semantic Textual Similarity Benchmark dataset construct from YJ Image Captions Dataset (Miyazaki and Shimizu, 2016) and annotated by crowdsource annotators.

**Dataset:** [`mteb/JSTS`](https://huggingface.co/datasets/mteb/JSTS) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2022.lrec-1.317.pdf#page=2.00)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | jpn | Web, Written | human-annotated | found | cosine_spearman |



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




#### `KLUE-STS` { .model-copy }

Human-annotated STS dataset of Korean reviews, news, and spoken word sets. Part of the Korean Language Understanding Evaluation (KLUE).

**Dataset:** [`klue/klue`](https://huggingface.co/datasets/klue/klue) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2105.09680)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | kor | News, Reviews, Spoken, Spoken, Written | human-annotated | found | cosine_spearman |



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




#### `KorSTS` { .model-copy }

Benchmark dataset for STS in Korean. Created by machine translation and human post editing of the STS-B dataset.

**Dataset:** [`dkoterwa/kor-sts`](https://huggingface.co/datasets/dkoterwa/kor-sts) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2004.03289)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | kor | News, Web | human-annotated | machine-translated and localized | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @article{ham2020kornli,
      author = {Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
      journal = {arXiv preprint arXiv:2004.03289},
      title = {KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
      year = {2020},
    }

    ```




#### `LCQMC` { .model-copy }

A Chinese dataset for textual relatedness

**Dataset:** [`C-MTEB/LCQMC`](https://huggingface.co/datasets/C-MTEB/LCQMC) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.emnlp-main.357)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn | Web, Written | human-annotated | found | cosine_spearman |



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




#### `PAWSX` { .model-copy }

A Chinese dataset for textual relatedness

**Dataset:** [`mteb/PAWSX`](https://huggingface.co/datasets/mteb/PAWSX) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.emnlp-main.357)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn | Encyclopaedic, Web, Written | human-annotated | human-translated | cosine_spearman |



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




#### `QBQTC` { .model-copy }

A Chinese question bank question title similarity dataset

**Dataset:** [`C-MTEB/QBQTC`](https://huggingface.co/datasets/C-MTEB/QBQTC) • **License:** not specified • [Learn more →](https://github.com/CLUEbenchmark/QBQTC/tree/main/dataset)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn | Web, Written | human-annotated | found | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @misc{clue2020qbqtc,
      author = {CLUE},
      title = {QBQTC: Question Bank Question Title Corpus},
      url = {https://github.com/CLUEbenchmark/QBQTC},
      year = {2020},
    }

    ```




#### `Query2Query` { .model-copy }

Query to Query Datasets.

**Dataset:** [`MCINext/query-to-query-sts`](https://huggingface.co/datasets/MCINext/query-to-query-sts) • **License:** not specified • [Learn more →](https://mcinext.com/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | not specified | derived | found | cosine_spearman |



??? quote "Citation"


    ```bibtex

    ```




#### `RUParaPhraserSTS` { .model-copy }

ParaPhraser is a news headlines corpus with precise, near and non-paraphrases.

**Dataset:** [`merionum/ru_paraphraser`](https://huggingface.co/datasets/merionum/ru_paraphraser) • **License:** mit • [Learn more →](https://aclanthology.org/2020.ngt-1.6)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | rus | News, Written | human-annotated | found | cosine_spearman |



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




#### `RonSTS` { .model-copy }

High-quality Romanian translation of STSBenchmark.

**Dataset:** [`mteb/RonSTS`](https://huggingface.co/datasets/mteb/RonSTS) • **License:** cc-by-4.0 • [Learn more →](https://openreview.net/forum?id=JH61CD7afTv)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ron | News, Social, Web, Written | human-annotated | machine-translated and verified | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @inproceedings{dumitrescu2021liro,
      author = {Dumitrescu, Stefan Daniel and Rebeja, Petru and Lorincz, Beata and Gaman, Mihaela and Avram, Andrei and Ilie, Mihai and Pruteanu, Andrei and Stan, Adriana and Rosia, Lorena and Iacobescu, Cristina and others},
      booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
      title = {LiRo: Benchmark and leaderboard for Romanian language tasks},
      year = {2021},
    }

    ```




#### `RuSTSBenchmarkSTS` { .model-copy }

Semantic Textual Similarity Benchmark (STSbenchmark) dataset translated into Russian and verified. The dataset was checked with RuCOLA model to ensure that the translation is good and filtered.

**Dataset:** [`ai-forever/ru-stsbenchmark-sts`](https://huggingface.co/datasets/ai-forever/ru-stsbenchmark-sts) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/PhilipMay/stsb-multi-mt/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | rus | News, Social, Web, Written | human-annotated | machine-translated and verified | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @inproceedings{huggingface:dataset:stsb_multi_mt,
      author = {Philip May},
      title = {Machine translated multilingual STS benchmark dataset.},
      url = {https://github.com/PhilipMay/stsb-multi-mt},
      year = {2021},
    }

    ```




#### `SICK-BR-STS` { .model-copy }

SICK-BR is a Portuguese inference corpus, human translated from SICK

**Dataset:** [`eduagarcia/sick-br`](https://huggingface.co/datasets/eduagarcia/sick-br) • **License:** not specified • [Learn more →](https://linux.ime.usp.br/~thalen/SICK_PT.pdf)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | por | Web, Written | human-annotated | human-translated and localized | cosine_spearman |



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




#### `SICK-NL-STS` { .model-copy }

SICK-NL (read: signal), a dataset targeting Natural Language Inference in Dutch. SICK-NL is obtained by translating the SICK dataset of (Marelli et al., 2014) from English into Dutch.

**Dataset:** [`clips/mteb-nl-sick-sts-pr`](https://huggingface.co/datasets/clips/mteb-nl-sick-sts-pr) • **License:** mit • [Learn more →](https://aclanthology.org/2021.eacl-main.126/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | nld | News, Social, Spoken, Web, Written | human-annotated | machine-translated | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @inproceedings{wijnholds2021sick,
      author = {Wijnholds, Gijs and Moortgat, Michael},
      booktitle = {Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume},
      pages = {1474--1479},
      title = {SICK-NL: A Dataset for Dutch Natural Language Inference},
      year = {2021},
    }

    ```




#### `SICK-R` { .model-copy }

Semantic Textual Similarity SICK-R dataset

**Dataset:** [`mteb/sickr-sts`](https://huggingface.co/datasets/mteb/sickr-sts) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://aclanthology.org/L14-1314/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Web, Written | human-annotated | created | cosine_spearman |



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




#### `SICK-R-PL` { .model-copy }

Polish version of SICK dataset for textual relatedness.

**Dataset:** [`PL-MTEB/sickr-pl-sts`](https://huggingface.co/datasets/PL-MTEB/sickr-pl-sts) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://aclanthology.org/2020.lrec-1.207)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | pol | Web, Written | human-annotated | human-translated and localized | cosine_spearman |



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




#### `SICK-R-VN` { .model-copy }

A translated dataset from Semantic Textual Similarity SICK-R dataset as described here: The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/sickr-sts-vn`](https://huggingface.co/datasets/GreenNode/sickr-sts-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2020.lrec-1.207)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to category (t2c) | vie | Web, Written | derived | machine-translated and LM verified | cosine_spearman |



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




#### `SICKFr` { .model-copy }

SICK dataset french version

**Dataset:** [`Lajavaness/SICK-fr`](https://huggingface.co/datasets/Lajavaness/SICK-fr) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://huggingface.co/datasets/Lajavaness/SICK-fr)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fra | Web, Written | human-annotated | machine-translated | cosine_spearman |



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
      month = may,
      pages = {216--223},
      publisher = {European Language Resources Association (ELRA)},
      title = {A {SICK} cure for the evaluation of compositional distributional semantic models},
      url = {http://www.lrec-conf.org/proceedings/lrec2014/pdf/363_Paper.pdf},
      year = {2014},
    }

    ```




#### `STS12` { .model-copy }

SemEval-2012 Task 6.

**Dataset:** [`mteb/sts12-sts`](https://huggingface.co/datasets/mteb/sts12-sts) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/S12-1051.pdf)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Encyclopaedic, News, Written | human-annotated | created | cosine_spearman |



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




#### `STS13` { .model-copy }

SemEval STS 2013 dataset.

**Dataset:** [`mteb/sts13-sts`](https://huggingface.co/datasets/mteb/sts13-sts) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/S13-1004/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | News, Non-fiction, Web, Written | human-annotated | created | cosine_spearman |



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




#### `STS14` { .model-copy }

SemEval STS 2014 dataset. Currently only the English dataset

**Dataset:** [`mteb/sts14-sts`](https://huggingface.co/datasets/mteb/sts14-sts) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/S14-1002)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Blog, Spoken, Web | derived | created | cosine_spearman |



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




#### `STS15` { .model-copy }

SemEval STS 2015 dataset

**Dataset:** [`mteb/sts15-sts`](https://huggingface.co/datasets/mteb/sts15-sts) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/S15-2010)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Blog, News, Spoken, Web, Written | human-annotated | created | cosine_spearman |



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




#### `STS16` { .model-copy }

SemEval-2016 Task 4

**Dataset:** [`mteb/sts16-sts`](https://huggingface.co/datasets/mteb/sts16-sts) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/S16-1001)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Blog, Spoken, Web | human-annotated | created | cosine_spearman |



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




#### `STS17` { .model-copy }

Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation

**Dataset:** [`mteb/sts17-crosslingual-sts`](https://huggingface.co/datasets/mteb/sts17-crosslingual-sts) • **License:** not specified • [Learn more →](https://alt.qcri.org/semeval2017/task1/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara, deu, eng, fra, ita, ... (9) | News, Web, Written | human-annotated | created | cosine_spearman |



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




#### `STS22` { .model-copy }

SemEval 2022 Task 8: Multilingual News Article Similarity

**Dataset:** [`mteb/sts22-crosslingual-sts`](https://huggingface.co/datasets/mteb/sts22-crosslingual-sts) • **License:** not specified • [Learn more →](https://competitions.codalab.org/competitions/33835)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara, cmn, deu, eng, fra, ... (10) | News, Written | human-annotated | found | cosine_spearman |



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




#### `STS22.v2` { .model-copy }

SemEval 2022 Task 8: Multilingual News Article Similarity. Version 2 filters updated on STS22 by removing pairs where one of entries contain empty sentences.

**Dataset:** [`mteb/sts22-crosslingual-sts`](https://huggingface.co/datasets/mteb/sts22-crosslingual-sts) • **License:** not specified • [Learn more →](https://competitions.codalab.org/competitions/33835)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ara, cmn, deu, eng, fra, ... (10) | News, Written | human-annotated | found | cosine_spearman |



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




#### `STSB` { .model-copy }

A Chinese dataset for textual relatedness

**Dataset:** [`mteb/STSB`](https://huggingface.co/datasets/mteb/STSB) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.emnlp-main.357)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn | News, Web, Written | human-annotated | machine-translated | cosine_spearman |



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




#### `STSBenchmark` { .model-copy }

Semantic Textual Similarity Benchmark (STSbenchmark) dataset.

**Dataset:** [`mteb/stsbenchmark-sts`](https://huggingface.co/datasets/mteb/stsbenchmark-sts) • **License:** not specified • [Learn more →](https://github.com/PhilipMay/stsb-multi-mt/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | Blog, News, Written | human-annotated | machine-translated and verified | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @inproceedings{huggingface:dataset:stsb_multi_mt,
      author = {Philip May},
      title = {Machine translated multilingual STS benchmark dataset.},
      url = {https://github.com/PhilipMay/stsb-multi-mt},
      year = {2021},
    }

    ```




#### `STSBenchmark-VN` { .model-copy }

A translated dataset from Semantic Textual Similarity Benchmark (STSbenchmark) dataset. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/stsbenchmark-sts-vn`](https://huggingface.co/datasets/GreenNode/stsbenchmark-sts-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/PhilipMay/stsb-multi-mt/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to category (t2c) | vie | Blog, News, Written | derived | machine-translated and LM verified | cosine_spearman |



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




#### `STSBenchmarkMultilingualSTS` { .model-copy }

Semantic Textual Similarity Benchmark (STSbenchmark) dataset, but translated using DeepL API.

**Dataset:** [`mteb/stsb_multi_mt`](https://huggingface.co/datasets/mteb/stsb_multi_mt) • **License:** not specified • [Learn more →](https://github.com/PhilipMay/stsb-multi-mt/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | cmn, deu, eng, fra, ita, ... (10) | News, Social, Spoken, Web, Written | human-annotated | machine-translated | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @inproceedings{huggingface:dataset:stsb_multi_mt,
      author = {Philip May},
      title = {Machine translated multilingual STS benchmark dataset.},
      url = {https://github.com/PhilipMay/stsb-multi-mt},
      year = {2021},
    }

    ```




#### `STSES` { .model-copy }

Spanish test sets from SemEval-2014 (Agirre et al., 2014) and SemEval-2015 (Agirre et al., 2015)

**Dataset:** [`mteb/STSES`](https://huggingface.co/datasets/mteb/STSES) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/PlanTL-GOB-ES/sts-es)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | spa | News, Web, Written | human-annotated | found | cosine_spearman |



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




#### `SemRel24STS` { .model-copy }

SemRel2024 is a collection of Semantic Textual Relatedness (STR) datasets for 14 languages, including African and Asian languages. The datasets are composed of sentence pairs, each assigned a relatedness score between 0 (completely) unrelated and 1 (maximally related) with a large range of expected relatedness values.

**Dataset:** [`mteb/SemRel24STS`](https://huggingface.co/datasets/mteb/SemRel24STS) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/SemRel/SemRel2024)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | afr, amh, arb, arq, ary, ... (12) | Spoken, Written | human-annotated | created | cosine_spearman |



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




#### `SynPerSTS` { .model-copy }

Synthetic Persian Semantic Textual Similarity Dataset

**Dataset:** [`MCINext/synthetic-persian-sts`](https://huggingface.co/datasets/MCINext/synthetic-persian-sts) • **License:** not specified • [Learn more →](https://mcinext.com/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fas | Blog, News, Religious, Web | LM-generated | LM-generated and verified | cosine_spearman |



??? quote "Citation"


    ```bibtex

    ```




#### `UkrSedUASmallSTSv1` { .model-copy }

Small (100k+) synthetic dataset for fine-tuning text embedding models for Ukrainian language (STS task)

**Dataset:** [`mteb/UkrSedUASmallSTSv1`](https://huggingface.co/datasets/mteb/UkrSedUASmallSTSv1) • **License:** bsd-3-clause • [Learn more →](https://huggingface.co/datasets/suntez13/sed-ua-small-sts-v1)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | ukr | Constructed | derived | found | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @proceedings{SED-UA-small2025,
      author = {Oleksandr Mediakov and Dmytro Martjanov and Vasyl Lytvyn},
      booktitle = {Proceedings of the Information Systems and Networks (SISN), Volume 17},
      doi = {10.23939/sisn2025.17.403},
      pages = {403--410},
      publisher = {Lviv Polytechnic National University},
      title = {SED-UA-Small: Ukrainian Synthetic Dataset for Text Embedding Models},
      url = {https://science.lpnu.ua/sisn/all-volumes-and-issues/volume-17-2025/sed-ua-small-ukrainian-synthetic-dataset-text-embedding},
      year = {2025},
    }

    ```




## Summarization

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 4

#### `SummEval` { .model-copy }

News Article Summary Semantic Similarity Estimation.

**Dataset:** [`mteb/summeval`](https://huggingface.co/datasets/mteb/summeval) • **License:** mit • [Learn more →](https://github.com/Yale-LILY/SummEval)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | News, Written | human-annotated | created | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @article{fabbri2020summeval,
      author = {Fabbri, Alexander R and Kry{\'s}ci{\'n}ski, Wojciech and McCann, Bryan and Xiong, Caiming and Socher, Richard and Radev, Dragomir},
      journal = {arXiv preprint arXiv:2007.12626},
      title = {SummEval: Re-evaluating Summarization Evaluation},
      year = {2020},
    }

    ```




#### `SummEvalFr` { .model-copy }

News Article Summary Semantic Similarity Estimation translated from english to french with DeepL.

**Dataset:** [`lyon-nlp/summarization-summeval-fr-p2p`](https://huggingface.co/datasets/lyon-nlp/summarization-summeval-fr-p2p) • **License:** mit • [Learn more →](https://github.com/Yale-LILY/SummEval)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fra | News, Written | human-annotated | machine-translated | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @article{fabbri2020summeval,
      author = {Fabbri, Alexander R and Kry{\'s}ci{\'n}ski, Wojciech and McCann, Bryan and Xiong, Caiming and Socher, Richard and Radev, Dragomir},
      journal = {arXiv preprint arXiv:2007.12626},
      title = {SummEval: Re-evaluating Summarization Evaluation},
      year = {2020},
    }

    ```




#### `SummEvalFrSummarization.v2` { .model-copy }

News Article Summary Semantic Similarity Estimation translated from english to french with DeepL. This version fixes a bug in the evaluation script that caused the main score to be computed incorrectly.

**Dataset:** [`lyon-nlp/summarization-summeval-fr-p2p`](https://huggingface.co/datasets/lyon-nlp/summarization-summeval-fr-p2p) • **License:** mit • [Learn more →](https://github.com/Yale-LILY/SummEval)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | fra | News, Written | human-annotated | machine-translated | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @article{fabbri2020summeval,
      author = {Fabbri, Alexander R and Kry{\'s}ci{\'n}ski, Wojciech and McCann, Bryan and Xiong, Caiming and Socher, Richard and Radev, Dragomir},
      journal = {arXiv preprint arXiv:2007.12626},
      title = {SummEval: Re-evaluating Summarization Evaluation},
      year = {2020},
    }

    ```




#### `SummEvalSummarization.v2` { .model-copy }

News Article Summary Semantic Similarity Estimation. This version fixes a bug in the evaluation script that caused the main score to be computed incorrectly.

**Dataset:** [`mteb/summeval`](https://huggingface.co/datasets/mteb/summeval) • **License:** mit • [Learn more →](https://github.com/Yale-LILY/SummEval)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| text to text (t2t) | eng | News, Written | human-annotated | created | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @article{fabbri2020summeval,
      author = {Fabbri, Alexander R and Kry{\'s}ci{\'n}ski, Wojciech and McCann, Bryan and Xiong, Caiming and Socher, Richard and Radev, Dragomir},
      journal = {arXiv preprint arXiv:2007.12626},
      title = {SummEval: Re-evaluating Summarization Evaluation},
      year = {2020},
    }

    ```




## VisualSTS(eng)

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 7

#### `STS12VisualSTS` { .model-copy }

SemEval-2012 Task 6.then rendered into images.

**Dataset:** [`mteb/rendered-sts12`](https://huggingface.co/datasets/mteb/rendered-sts12) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2402.08183/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | Encyclopaedic, News, Written | human-annotated | rendered | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @article{xiao2024pixel,
      author = {Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2402.08183},
      title = {Pixel Sentence Representation Learning},
      year = {2024},
    }

    ```




#### `STS13VisualSTS` { .model-copy }

SemEval STS 2013 dataset.then rendered into images.

**Dataset:** [`mteb/rendered-sts13`](https://huggingface.co/datasets/mteb/rendered-sts13) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2402.08183/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | News, Non-fiction, Web, Written | human-annotated | rendered | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @article{xiao2024pixel,
      author = {Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2402.08183},
      title = {Pixel Sentence Representation Learning},
      year = {2024},
    }

    ```




#### `STS14VisualSTS` { .model-copy }

SemEval STS 2014 dataset. Currently only the English dataset.rendered into images.

**Dataset:** [`mteb/rendered-sts14`](https://huggingface.co/datasets/mteb/rendered-sts14) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2402.08183/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | Blog, Spoken, Web | derived | rendered | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @article{xiao2024pixel,
      author = {Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2402.08183},
      title = {Pixel Sentence Representation Learning},
      year = {2024},
    }

    ```




#### `STS15VisualSTS` { .model-copy }

SemEval STS 2015 datasetrendered into images.

**Dataset:** [`mteb/rendered-sts15`](https://huggingface.co/datasets/mteb/rendered-sts15) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2402.08183/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | Blog, News, Spoken, Web, Written | human-annotated | rendered | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @article{xiao2024pixel,
      author = {Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2402.08183},
      title = {Pixel Sentence Representation Learning},
      year = {2024},
    }

    ```




#### `STS16VisualSTS` { .model-copy }

SemEval STS 2016 datasetrendered into images.

**Dataset:** [`mteb/rendered-sts16`](https://huggingface.co/datasets/mteb/rendered-sts16) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2402.08183/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | Blog, Spoken, Web | human-annotated | rendered | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @article{xiao2024pixel,
      author = {Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2402.08183},
      title = {Pixel Sentence Representation Learning},
      year = {2024},
    }

    ```




#### `VisualSTS-b-Eng` { .model-copy }

STSBenchmarkMultilingualVisualSTS English only.

**License:** not specified • [Learn more →](https://arxiv.org/abs/2402.08183/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | News, Social, Spoken, Web, Written | human-annotated | rendered | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @article{xiao2024pixel,
      author = {Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2402.08183},
      title = {Pixel Sentence Representation Learning},
      year = {2024},
    }

    ```



??? info "Tasks"

    | name                                                                                         | type             | modalities   | languages                         |
    |:---------------------------------------------------------------------------------------------|:-----------------|:-------------|:----------------------------------|
    | [STSBenchmarkMultilingualVisualSTS](./visualsts(multi).md#stsbenchmarkmultilingualvisualsts) | VisualSTS(multi) | image        | cmn, deu, eng, fra, ita, ... (10) |


#### `VisualSTS17Eng` { .model-copy }

STS17MultilingualVisualSTS English only.

**License:** not specified • [Learn more →](https://arxiv.org/abs/2402.08183/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | eng | News, Social, Spoken, Web, Written | human-annotated | rendered | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @article{xiao2024pixel,
      author = {Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2402.08183},
      title = {Pixel Sentence Representation Learning},
      year = {2024},
    }

    ```



??? info "Tasks"

    | name                                                                           | type             | modalities   | languages                        |
    |:-------------------------------------------------------------------------------|:-----------------|:-------------|:---------------------------------|
    | [STS17MultilingualVisualSTS](./visualsts(multi).md#sts17multilingualvisualsts) | VisualSTS(multi) | image        | ara, deu, eng, fra, ita, ... (9) |




## VisualSTS(multi)

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 4

#### `STS17MultilingualVisualSTS` { .model-copy }

Semantic Textual Similarity 17 (STS-17) dataset, rendered into images.

**Dataset:** [`Pixel-Linguist/rendered-sts17`](https://huggingface.co/datasets/Pixel-Linguist/rendered-sts17) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2402.08183/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | ara, deu, eng, fra, ita, ... (9) | News, Social, Spoken, Web, Written | human-annotated | rendered | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @article{xiao2024pixel,
      author = {Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2402.08183},
      title = {Pixel Sentence Representation Learning},
      year = {2024},
    }

    ```




#### `STSBenchmarkMultilingualVisualSTS` { .model-copy }

Semantic Textual Similarity Benchmark (STSbenchmark) dataset, translated into target languages using DeepL API,then rendered into images.built upon multi-sts created by Philip May

**Dataset:** [`Pixel-Linguist/rendered-stsb`](https://huggingface.co/datasets/Pixel-Linguist/rendered-stsb) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2402.08183/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | cmn, deu, eng, fra, ita, ... (10) | News, Social, Spoken, Web, Written | human-annotated | rendered | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @article{xiao2024pixel,
      author = {Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2402.08183},
      title = {Pixel Sentence Representation Learning},
      year = {2024},
    }

    ```




#### `VisualSTS-b-Multilingual` { .model-copy }

STSBenchmarkMultilingualVisualSTS multilingual.

**License:** not specified • [Learn more →](https://arxiv.org/abs/2402.08183/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | cmn, deu, fra, ita, nld, ... (9) | News, Social, Spoken, Web, Written | human-annotated | rendered | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @article{xiao2024pixel,
      author = {Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2402.08183},
      title = {Pixel Sentence Representation Learning},
      year = {2024},
    }

    ```



??? info "Tasks"

    | name                                                                                         | type             | modalities   | languages                         |
    |:---------------------------------------------------------------------------------------------|:-----------------|:-------------|:----------------------------------|
    | [STSBenchmarkMultilingualVisualSTS](./visualsts(multi).md#stsbenchmarkmultilingualvisualsts) | VisualSTS(multi) | image        | cmn, deu, eng, fra, ita, ... (10) |


#### `VisualSTS17Multilingual` { .model-copy }

STS17MultilingualVisualSTS multilingual.

**License:** not specified • [Learn more →](https://arxiv.org/abs/2402.08183/)

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| image to image (i2i) | ara, deu, eng, fra, ita, ... (9) | News, Social, Spoken, Web, Written | human-annotated | rendered | cosine_spearman |



??? quote "Citation"


    ```bibtex

    @article{xiao2024pixel,
      author = {Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
      journal = {arXiv preprint arXiv:2402.08183},
      title = {Pixel Sentence Representation Learning},
      year = {2024},
    }

    ```



??? info "Tasks"

    | name                                                                           | type             | modalities   | languages                        |
    |:-------------------------------------------------------------------------------|:-----------------|:-------------|:---------------------------------|
    | [STS17MultilingualVisualSTS](./visualsts(multi).md#sts17multilingualvisualsts) | VisualSTS(multi) | image        | ara, deu, eng, fra, ita, ... (9) |

<!-- END-TASKS -->
