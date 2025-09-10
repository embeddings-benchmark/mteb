
# MultilabelClassification

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 7 

#### BrazilianToxicTweetsClassification


        ToLD-Br is the biggest dataset for toxic tweets in Brazilian Portuguese, crowdsourced by 42 annotators selected from
        a pool of 129 volunteers. Annotators were selected aiming to create a plural group in terms of demographics (ethnicity,
        sexual orientation, age, gender). Each tweet was labeled by three annotators in 6 possible categories: LGBTQ+phobia,
        Xenophobia, Obscene, Insult, Misogyny and Racism.
        

**Dataset:** [`mteb/told-br`](https://huggingface.co/datasets/mteb/told-br) • **License:** cc-by-sa-4.0 • [Learn more →](https://paperswithcode.com/dataset/told-br)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | por | Constructed, Written | expert-annotated | found |



#### CEDRClassification

Classification of sentences by emotions, labeled into 5 categories (joy, sadness, surprise, fear, and anger).

**Dataset:** [`ai-forever/cedr-classification`](https://huggingface.co/datasets/ai-forever/cedr-classification) • **License:** apache-2.0 • [Learn more →](https://www.sciencedirect.com/science/article/pii/S1877050921013247)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Blog, Social, Web, Written | human-annotated | found |



#### EmitClassification

The EMit dataset is a comprehensive resource for the detection of emotions in Italian social media texts.
        The EMit dataset consists of social media messages about TV shows, TV series, music videos, and advertisements.
        Each message is annotated with one or more of the 8 primary emotions defined by Plutchik
        (anger, anticipation, disgust, fear, joy, sadness, surprise, trust), as well as an additional label “love.”
        

**Dataset:** [`MattiaSangermano/emit`](https://huggingface.co/datasets/MattiaSangermano/emit) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/oaraque/emit)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ita | Social, Written | expert-annotated | found |



#### KorHateSpeechMLClassification


        The Korean Multi-label Hate Speech Dataset, K-MHaS, consists of 109,692 utterances from Korean online news comments,
        labelled with 8 fine-grained hate speech classes (labels: Politics, Origin, Physical, Age, Gender, Religion, Race, Profanity)
        or Not Hate Speech class. Each utterance provides from a single to four labels that can handles Korean language patterns effectively.
        For more details, please refer to the paper about K-MHaS, published at COLING 2022.
        This dataset is based on the Korean online news comments available on Kaggle and Github.
        The unlabeled raw data was collected between January 2018 and June 2020.
        The language producers are users who left the comments on the Korean online news platform between 2018 and 2020.
        

**Dataset:** [`jeanlee/kmhas_korean_hate_speech`](https://huggingface.co/datasets/jeanlee/kmhas_korean_hate_speech) • **License:** cc-by-sa-4.0 • [Learn more →](https://paperswithcode.com/dataset/korean-multi-label-hate-speech-dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | Social, Written | expert-annotated | found |



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



#### MultiEURLEXMultilabelClassification

EU laws in 23 EU languages containing annotated labels for 21 EUROVOC concepts.

**Dataset:** [`mteb/eurlex-multilingual`](https://huggingface.co/datasets/mteb/eurlex-multilingual) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/coastalcph/multi_eurlex)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | bul, ces, dan, deu, ell, ... (23) | Government, Legal, Written | expert-annotated | found |



#### SensitiveTopicsClassification

Multilabel classification of sentences across 18 sensitive topics.

**Dataset:** [`ai-forever/sensitive-topics-classification`](https://huggingface.co/datasets/ai-forever/sensitive-topics-classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/2021.bsnlp-1.4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Social, Web, Written | human-annotated | found |
