
# MultilabelClassification

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 9

#### BrazilianToxicTweetsClassification


        ToLD-Br is the biggest dataset for toxic tweets in Brazilian Portuguese, crowdsourced by 42 annotators selected from
        a pool of 129 volunteers. Annotators were selected aiming to create a plural group in terms of demographics (ethnicity,
        sexual orientation, age, gender). Each tweet was labeled by three annotators in 6 possible categories: LGBTQ+phobia,
        Xenophobia, Obscene, Insult, Misogyny and Racism.


**Dataset:** [`mteb/BrazilianToxicTweetsClassification`](https://huggingface.co/datasets/mteb/BrazilianToxicTweetsClassification) • **License:** cc-by-sa-4.0 • [Learn more →](https://paperswithcode.com/dataset/told-br)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | por | Constructed, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{DBLP:journals/corr/abs-2010-04543,
      author = {Joao Augusto Leite and
    Diego F. Silva and
    Kalina Bontcheva and
    Carolina Scarton},
      eprint = {2010.04543},
      eprinttype = {arXiv},
      journal = {CoRR},
      timestamp = {Tue, 15 Dec 2020 16:10:16 +0100},
      title = {Toxic Language Detection in Social Media for Brazilian Portuguese:
    New Dataset and Multilingual Analysis},
      url = {https://arxiv.org/abs/2010.04543},
      volume = {abs/2010.04543},
      year = {2020},
    }

    ```




#### CEDRClassification

Classification of sentences by emotions, labeled into 5 categories (joy, sadness, surprise, fear, and anger).

**Dataset:** [`ai-forever/cedr-classification`](https://huggingface.co/datasets/ai-forever/cedr-classification) • **License:** apache-2.0 • [Learn more →](https://www.sciencedirect.com/science/article/pii/S1877050921013247)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Blog, Social, Web, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{sboev2021data,
      author = {Sboev, Alexander and Naumov, Aleksandr and Rybka, Roman},
      journal = {Procedia Computer Science},
      pages = {637--642},
      publisher = {Elsevier},
      title = {Data-Driven Model for Emotion Detection in Russian Texts},
      volume = {190},
      year = {2021},
    }

    ```




#### EmitClassification

The EMit dataset is a comprehensive resource for the detection of emotions in Italian social media texts.
        The EMit dataset consists of social media messages about TV shows, TV series, music videos, and advertisements.
        Each message is annotated with one or more of the 8 primary emotions defined by Plutchik
        (anger, anticipation, disgust, fear, joy, sadness, surprise, trust), as well as an additional label “love.”


**Dataset:** [`MattiaSangermano/emit`](https://huggingface.co/datasets/MattiaSangermano/emit) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/oaraque/emit)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ita | Social, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{araque2023emit,
      author = {Araque, O and Frenda, S and Sprugnoli, R and Nozza, D and Patti, V and others},
      booktitle = {CEUR WORKSHOP PROCEEDINGS},
      organization = {CEUR-WS},
      pages = {1--8},
      title = {EMit at EVALITA 2023: Overview of the Categorical Emotion Detection in Italian Social Media Task},
      volume = {3473},
      year = {2023},
    }

    ```




#### KorHateSpeechMLClassification


        The Korean Multi-label Hate Speech Dataset, K-MHaS, consists of 109,692 utterances from Korean online news comments,
        labelled with 8 fine-grained hate speech classes (labels: Politics, Origin, Physical, Age, Gender, Religion, Race, Profanity)
        or Not Hate Speech class. Each utterance provides from a single to four labels that can handles Korean language patterns effectively.
        For more details, please refer to the paper about K-MHaS, published at COLING 2022.
        This dataset is based on the Korean online news comments available on Kaggle and Github.
        The unlabeled raw data was collected between January 2018 and June 2020.
        The language producers are users who left the comments on the Korean online news platform between 2018 and 2020.


**Dataset:** [`mteb/KorHateSpeechMLClassification`](https://huggingface.co/datasets/mteb/KorHateSpeechMLClassification) • **License:** cc-by-sa-4.0 • [Learn more →](https://paperswithcode.com/dataset/korean-multi-label-hate-speech-dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | Social, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{lee-etal-2022-k,
      address = {Gyeongju, Republic of Korea},
      author = {Lee, Jean  and
    Lim, Taejun  and
    Lee, Heejun  and
    Jo, Bogeun  and
    Kim, Yangsok  and
    Yoon, Heegeun  and
    Han, Soyeon Caren},
      booktitle = {Proceedings of the 29th International Conference on Computational Linguistics},
      month = oct,
      pages = {3530--3538},
      publisher = {International Committee on Computational Linguistics},
      title = {K-{MH}a{S}: A Multi-label Hate Speech Detection Dataset in {K}orean Online News Comment},
      url = {https://aclanthology.org/2022.coling-1.311},
      year = {2022},
    }

    ```




#### MalteseNewsClassification

A multi-label topic classification dataset for Maltese News
        Articles. The data was collected from the press_mt subset from Korpus
        Malti v4.0. Article contents were cleaned to filter out JavaScript, CSS,
        & repeated non-Maltese sub-headings. The labels are based on the category
        field from this corpus.


**Dataset:** [`MLRS/maltese_news_categories`](https://huggingface.co/datasets/MLRS/maltese_news_categories) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/MLRS/maltese_news_categories)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | mlt | Constructed, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{maltese-news-datasets,
      author = {Chaudhary, Amit Kumar  and
    Micallef, Kurt  and
    Borg, Claudia},
      booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation},
      month = may,
      publisher = {Association for Computational Linguistics},
      title = {Topic Classification and Headline Generation for {M}altese using a Public News Corpus},
      year = {2024},
    }

    ```




#### MultiEURLEXMultilabelClassification

EU laws in 23 EU languages containing annotated labels for 21 EUROVOC concepts.

**Dataset:** [`mteb/eurlex-multilingual`](https://huggingface.co/datasets/mteb/eurlex-multilingual) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/coastalcph/multi_eurlex)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | bul, ces, dan, deu, ell, ... (23) | Government, Legal, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{chalkidis-etal-2021-multieurlex,
      author = {Chalkidis, Ilias
    and Fergadiotis, Manos
    and Androutsopoulos, Ion},
      booktitle = {Proceedings of the 2021 Conference on Empirical Methods
    in Natural Language Processing},
      location = {Punta Cana, Dominican Republic},
      publisher = {Association for Computational Linguistics},
      title = {MultiEURLEX -- A multi-lingual and multi-label legal document
    classification dataset for zero-shot cross-lingual transfer},
      url = {https://arxiv.org/abs/2109.00904},
      year = {2021},
    }

    ```




#### SensitiveTopicsClassification

Multilabel classification of sentences across 18 sensitive topics.

**Dataset:** [`ai-forever/sensitive-topics-classification`](https://huggingface.co/datasets/ai-forever/sensitive-topics-classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/2021.bsnlp-1.4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Social, Web, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{babakov-etal-2021-detecting,
      abstract = {Not all topics are equally {``}flammable{''} in terms of toxicity: a calm discussion of turtles or fishing less often fuels inappropriate toxic dialogues than a discussion of politics or sexual minorities. We define a set of sensitive topics that can yield inappropriate and toxic messages and describe the methodology of collecting and labelling a dataset for appropriateness. While toxicity in user-generated data is well-studied, we aim at defining a more fine-grained notion of inappropriateness. The core of inappropriateness is that it can harm the reputation of a speaker. This is different from toxicity in two respects: (i) inappropriateness is topic-related, and (ii) inappropriate message is not toxic but still unacceptable. We collect and release two datasets for Russian: a topic-labelled dataset and an appropriateness-labelled dataset. We also release pre-trained classification models trained on this data.},
      address = {Kiyv, Ukraine},
      author = {Babakov, Nikolay  and
    Logacheva, Varvara  and
    Kozlova, Olga  and
    Semenov, Nikita  and
    Panchenko, Alexander},
      booktitle = {Proceedings of the 8th Workshop on Balto-Slavic Natural Language Processing},
      editor = {Babych, Bogdan  and
    Kanishcheva, Olga  and
    Nakov, Preslav  and
    Piskorski, Jakub  and
    Pivovarova, Lidia  and
    Starko, Vasyl  and
    Steinberger, Josef  and
    Yangarber, Roman  and
    Marci{\'n}czuk, Micha{\l}  and
    Pollak, Senja  and
    P{\v{r}}ib{\'a}{\v{n}}, Pavel  and
    Robnik-{\v{S}}ikonja, Marko},
      month = apr,
      pages = {26--36},
      publisher = {Association for Computational Linguistics},
      title = {Detecting Inappropriate Messages on Sensitive Topics that Could Harm a Company{'}s Reputation},
      url = {https://aclanthology.org/2021.bsnlp-1.4},
      year = {2021},
    }

    ```




#### SwedishPatentCPCGroupClassification

This dataset contains historical Swedish patent documents (1885-1972) classified according to the Cooperative Patent Classification (CPC) system at the group level. Each document can have multiple labels, making this a challenging multi-label classification task with significant class imbalance and data sparsity characteristics. The dataset includes patent claims text extracted from digitally recreated versions of historical Swedish patents, generated using Optical Character Recognition (OCR) from original paper documents. The text quality varies due to OCR limitations, but all CPC labels were manually assigned by patent engineers at PRV (Swedish Patent and Registration Office), ensuring high reliability for machine learning applications.

**Dataset:** [`atheer2104/swedish-patent-cpc-group-new`](https://huggingface.co/datasets/atheer2104/swedish-patent-cpc-group-new) • **License:** mit • [Learn more →](https://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-368254)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | swe | Government, Legal | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @mastersthesis{Salim1987995,
      author = {Salim, Atheer},
      institution = {KTH, School of Electrical Engineering and Computer Science (EECS)},
      keywords = {Multi-label Text Classification, Machine Learning, Patent Classification, Deep Learning, Natural Language Processing, Textklassificering med flera Klasser, Maskininlärning, Patentklassificering, Djupinlärning, Språkteknologi},
      number = {2025:571},
      pages = {70},
      school = {KTH, School of Electrical Engineering and Computer Science (EECS)},
      series = {TRITA-EECS-EX},
      title = {Machine Learning for Classifying Historical Swedish Patents : A Comparison of Textual and Combined Data Approaches},
      url = {https://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-368254},
      year = {2025},
    }

    ```




#### SwedishPatentCPCSubclassClassification

This dataset contains historical Swedish patent documents (1885-1972) classified according to the Cooperative Patent Classification (CPC) system. Each document can have multiple labels, making this a multi-label classification task with significant implications for patent retrieval and prior art search.
		The dataset includes patent claims text extracted from digitally recreated versions of historical Swedish patents, generated using Optical Character Recognition (OCR) from original paper documents. The text quality varies due to OCR limitations, but all CPC labels were manually assigned by patent engineers at PRV (Swedish Patent and Registration Office), ensuring high reliability for machine learning applications.

**Dataset:** [`atheer2104/swedish-patent-cpc-subclass-new`](https://huggingface.co/datasets/atheer2104/swedish-patent-cpc-subclass-new) • **License:** mit • [Learn more →](https://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-368254)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | swe | Government, Legal | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @mastersthesis{Salim1987995,
      author = {Salim, Atheer},
      institution = {KTH, School of Electrical Engineering and Computer Science (EECS)},
      keywords = {Multi-label Text Classification, Machine Learning, Patent Classification, Deep Learning, Natural Language Processing, Textklassificering med flera Klasser, Maskininlärning, Patentklassificering, Djupinlärning, Språkteknologi},
      number = {2025:571},
      pages = {70},
      school = {KTH, School of Electrical Engineering and Computer Science (EECS)},
      series = {TRITA-EECS-EX},
      title = {Machine Learning for Classifying Historical Swedish Patents : A Comparison of Textual and Combined Data Approaches},
      url = {https://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-368254},
      year = {2025},
    }

    ```
