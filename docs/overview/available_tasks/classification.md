
# Classification

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 456 

#### AJGT

Arabic Jordanian General Tweets (AJGT) Corpus consisted of 1,800 tweets annotated as positive and negative. Modern Standard Arabic (MSA) or Jordanian dialect.

**Dataset:** [`komari6/ajgt_twitter_ar`](https://huggingface.co/datasets/komari6/ajgt_twitter_ar) • **License:** afl-3.0 • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-60042-0_66/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Social, Written | human-annotated | found |



#### AJGT.v2

Arabic Jordanian General Tweets (AJGT) Corpus consisted of 1,800 tweets (900 for training and 900 for testing) annotated as positive and negative. Modern Standard Arabic (MSA) or Jordanian dialect.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/ajgt`](https://huggingface.co/datasets/mteb/ajgt) • **License:** afl-3.0 • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-60042-0_66/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Social, Written | human-annotated | found |



#### AfriSentiClassification

AfriSenti is the largest sentiment analysis dataset for under-represented African languages.

**Dataset:** [`shmuhammad/AfriSenti-twitter-sentiment`](https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2302.08956)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | amh, arq, ary, hau, ibo, ... (12) | Social, Written | derived | found |



#### AfriSentiLangClassification

AfriSentiLID is the largest LID classification dataset for African Languages.

**Dataset:** [`HausaNLP/afrisenti-lid-data`](https://huggingface.co/datasets/HausaNLP/afrisenti-lid-data) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/HausaNLP/afrisenti-lid-data/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | amh, arq, ary, hau, ibo, ... (12) | Social, Written | derived | found |



#### AllegroReviews

A Polish dataset for sentiment classification on reviews from e-commerce marketplace Allegro.

**Dataset:** [`PL-MTEB/allegro-reviews`](https://huggingface.co/datasets/PL-MTEB/allegro-reviews) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2020.acl-main.111.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Reviews | derived | found |



#### AllegroReviews.v2

A Polish dataset for sentiment classification on reviews from e-commerce marketplace Allegro.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/allegro_reviews`](https://huggingface.co/datasets/mteb/allegro_reviews) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2020.acl-main.111.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Reviews | derived | found |



#### AmazonCounterfactualClassification

A collection of Amazon customer reviews annotated for counterfactual detection pair classification.

**Dataset:** [`mteb/amazon_counterfactual`](https://huggingface.co/datasets/mteb/amazon_counterfactual) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2104.06893)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu, eng, jpn | Reviews, Written | human-annotated | found |



#### AmazonCounterfactualVNClassification

A collection of translated Amazon customer reviews annotated for counterfactual detection pair classification.
        The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
        - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
        - Applies advanced embedding models to filter the translations.
        - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.
        

**Dataset:** [`GreenNode/amazon-counterfactual-vn`](https://huggingface.co/datasets/GreenNode/amazon-counterfactual-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2104.06893)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Reviews, Written | derived | machine-translated and LM verified |



#### AmazonPolarityClassification

Amazon Polarity Classification Dataset.

**Dataset:** [`mteb/amazon_polarity`](https://huggingface.co/datasets/mteb/amazon_polarity) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/amazon_polarity)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | derived | found |



#### AmazonPolarityClassification.v2

Amazon Polarity Classification Dataset.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/amazon_polarity`](https://huggingface.co/datasets/mteb/amazon_polarity) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/amazon_polarity)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | derived | found |



#### AmazonPolarityVNClassification

A collection of translated Amazon customer reviews annotated for polarity classification.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.
            

**Dataset:** [`GreenNode/amazon-polarity-vn`](https://huggingface.co/datasets/GreenNode/amazon-polarity-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/amazon_polarity)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Reviews, Written | derived | machine-translated and LM verified |



#### AmazonReviewsClassification

A collection of Amazon reviews specifically designed to aid research in multilingual text classification.

**Dataset:** [`mteb/AmazonReviewsClassification`](https://huggingface.co/datasets/mteb/AmazonReviewsClassification) • **License:** https://docs.opendata.aws/amazon-reviews-ml/license.txt • [Learn more →](https://arxiv.org/abs/2010.02573)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn, deu, eng, fra, jpn, ... (6) | Reviews, Written | human-annotated | found |



#### AmazonReviewsVNClassification

A collection of translated Amazon reviews specifically designed to aid research in multilingual text classification.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/amazon-reviews-multi-vn`](https://huggingface.co/datasets/GreenNode/amazon-reviews-multi-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2010.02573)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Reviews, Written | derived | machine-translated and LM verified |



#### AngryTweetsClassification

A sentiment dataset with 3 classes (positiv, negativ, neutral) for Danish tweets

**Dataset:** [`DDSC/angry-tweets`](https://huggingface.co/datasets/DDSC/angry-tweets) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2021.nodalida-main.53/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Social, Written | human-annotated | found |



#### AngryTweetsClassification.v2

A sentiment dataset with 3 classes (positiv, negativ, neutral) for Danish tweets
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/angry_tweets`](https://huggingface.co/datasets/mteb/angry_tweets) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2021.nodalida-main.53/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Social, Written | human-annotated | found |



#### ArxivClassification

Classification Dataset of Arxiv Papers

**Dataset:** [`mteb/ArxivClassification`](https://huggingface.co/datasets/mteb/ArxivClassification) • **License:** not specified • [Learn more →](https://ieeexplore.ieee.org/document/8675939)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Academic, Written | derived | found |



#### ArxivClassification.v2

Classification Dataset of Arxiv Papers
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/arxiv`](https://huggingface.co/datasets/mteb/arxiv) • **License:** not specified • [Learn more →](https://ieeexplore.ieee.org/document/8675939)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Academic, Written | derived | found |



#### Banking77Classification

Dataset composed of online banking queries annotated with their corresponding intents.

**Dataset:** [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) • **License:** mit • [Learn more →](https://arxiv.org/abs/2003.04807)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Written | human-annotated | found |



#### Banking77Classification.v2

Dataset composed of online banking queries annotated with their corresponding intents.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) • **License:** mit • [Learn more →](https://arxiv.org/abs/2003.04807)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Written | human-annotated | found |



#### Banking77VNClassification

A translated dataset composed of online banking queries annotated with their corresponding intents.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/banking77-vn`](https://huggingface.co/datasets/GreenNode/banking77-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2003.04807)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Written | derived | machine-translated and LM verified |



#### BengaliDocumentClassification

Dataset for News Classification, categorized with 13 domains.

**Dataset:** [`dialect-ai/shironaam`](https://huggingface.co/datasets/dialect-ai/shironaam) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/2023.eacl-main.4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ben | News, Written | derived | found |



#### BengaliDocumentClassification.v2

Dataset for News Classification, categorized with 13 domains.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/bengali_document`](https://huggingface.co/datasets/mteb/bengali_document) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/2023.eacl-main.4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ben | News, Written | derived | found |



#### BengaliHateSpeechClassification

The Bengali Hate Speech Dataset is a Bengali-language dataset of news articles collected from various Bengali media sources and categorized based on the type of hate in the text.

**Dataset:** [`rezacsedu/bn_hate_speech`](https://huggingface.co/datasets/rezacsedu/bn_hate_speech) • **License:** mit • [Learn more →](https://huggingface.co/datasets/bn_hate_speech)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ben | News, Written | expert-annotated | found |



#### BengaliHateSpeechClassification.v2

The Bengali Hate Speech Dataset is a Bengali-language dataset of news articles collected from various Bengali media sources and categorized based on the type of hate in the text.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/bengali_hate_speech`](https://huggingface.co/datasets/mteb/bengali_hate_speech) • **License:** mit • [Learn more →](https://huggingface.co/datasets/bn_hate_speech)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ben | News, Written | expert-annotated | found |



#### BengaliSentimentAnalysis

dataset contains 3307 Negative reviews and 8500 Positive reviews collected and manually annotated from Youtube Bengali drama.

**Dataset:** [`Akash190104/bengali_sentiment_analysis`](https://huggingface.co/datasets/Akash190104/bengali_sentiment_analysis) • **License:** cc-by-4.0 • [Learn more →](https://data.mendeley.com/datasets/p6zc7krs37/4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ben | Reviews, Written | human-annotated | found |



#### BengaliSentimentAnalysis.v2

dataset contains 2854 Negative reviews and 7238 Positive reviews collected and manually annotated from Youtube Bengali drama.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/bengali_sentiment_analysis`](https://huggingface.co/datasets/mteb/bengali_sentiment_analysis) • **License:** cc-by-4.0 • [Learn more →](https://data.mendeley.com/datasets/p6zc7krs37/4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ben | Reviews, Written | human-annotated | found |



#### BulgarianStoreReviewSentimentClassfication

Bulgarian online store review dataset for sentiment classification.

**Dataset:** [`artist/Bulgarian-Online-Store-Feedback-Text-Analysis`](https://huggingface.co/datasets/artist/Bulgarian-Online-Store-Feedback-Text-Analysis) • **License:** cc-by-4.0 • [Learn more →](https://doi.org/10.7910/DVN/TXIK9P)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | bul | Reviews, Written | human-annotated | found |



#### CBD

Polish Tweets annotated for cyberbullying detection.

**Dataset:** [`PL-MTEB/cbd`](https://huggingface.co/datasets/PL-MTEB/cbd) • **License:** bsd-3-clause • [Learn more →](http://2019.poleval.pl/files/poleval2019.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Social, Written | human-annotated | found |



#### CBD.v2

Polish Tweets annotated for cyberbullying detection.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/cbd`](https://huggingface.co/datasets/mteb/cbd) • **License:** bsd-3-clause • [Learn more →](http://2019.poleval.pl/files/poleval2019.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Social, Written | human-annotated | found |



#### CSFDCZMovieReviewSentimentClassification

The dataset contains 30k user reviews from csfd.cz in Czech.

**Dataset:** [`fewshot-goes-multilingual/cs_csfd-movie-reviews`](https://huggingface.co/datasets/fewshot-goes-multilingual/cs_csfd-movie-reviews) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2304.01922)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | derived | found |



#### CSFDCZMovieReviewSentimentClassification.v2

The dataset contains 30k user reviews from csfd.cz in Czech.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/csfdcz_movie_review_sentiment`](https://huggingface.co/datasets/mteb/csfdcz_movie_review_sentiment) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2304.01922)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | derived | found |



#### CSFDSKMovieReviewSentimentClassification

The dataset contains 30k user reviews from csfd.cz in Slovak.

**Dataset:** [`fewshot-goes-multilingual/sk_csfd-movie-reviews`](https://huggingface.co/datasets/fewshot-goes-multilingual/sk_csfd-movie-reviews) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2304.01922)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | slk | Reviews, Written | derived | found |



#### CSFDSKMovieReviewSentimentClassification.v2

The dataset contains 30k user reviews from csfd.cz in Slovak.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/csfdsk_movie_review_sentiment`](https://huggingface.co/datasets/mteb/csfdsk_movie_review_sentiment) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2304.01922)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | slk | Reviews, Written | derived | found |



#### CUADAffiliateLicenseLicenseeLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if a clause describes a license grant to a licensee (incl. sublicensor) and the affiliates of such licensee/sublicensor.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADAffiliateLicenseLicensorLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause describes a license grant by affiliates of the licensor or that includes intellectual property of affiliates of the licensor.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADAntiAssignmentLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause requires consent or notice of a party if the contract is assigned to a third party.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADAuditRightsLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause gives a party the right to audit the books, records, or physical locations of the counterparty to ensure compliance with the contract.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADCapOnLiabilityLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a cap on liability upon the breach of a party's obligation. This includes time limitation for the counterparty to bring claims or maximum amount for recovery.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADChangeOfControlLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause gives one party the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADCompetitiveRestrictionExceptionLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause mentions exceptions or carveouts to Non-Compete, Exclusivity and No-Solicit of Customers.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADCovenantNotToSueLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that a party is restricted from contesting the validity of the counterparty's ownership of intellectual property or otherwise bringing a claim against the counterparty for matters unrelated to the contract.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADEffectiveDateLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies the date upon which the agreement becomes effective.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADExclusivityLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies exclusive dealing commitment with the counterparty. This includes a commitment to procure all 'requirements' from one party of certain technology, goods, or services or a prohibition on licensing or selling technology, goods or services to third parties, or a prohibition on collaborating or working with other parties), whether during the contract or after the contract ends (or both).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADExpirationDateLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies the date upon which the initial term expires.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADGoverningLawLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies which state/country’s law governs the contract.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADIPOwnershipAssignmentLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that intellectual property created by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADInsuranceLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if clause creates a requirement for insurance that must be maintained by one party for the benefit of the counterparty.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADIrrevocableOrPerpetualLicenseLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a license grant that is irrevocable or perpetual.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADJointIPOwnershipLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause provides for joint or shared ownership of intellectual property between the parties to the contract.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADLicenseGrantLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause contains a license granted by one party to its counterparty.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADLiquidatedDamagesLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause awards either party liquidated damages for breach or a fee upon the termination of a contract (termination fee).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADMinimumCommitmentLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a minimum order size or minimum amount or units per time period that one party must buy from the counterparty.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADMostFavoredNationLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADNoSolicitOfCustomersLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause restricts a party from contracting or soliciting customers or partners of the counterparty, whether during the contract or after the contract ends (or both).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADNoSolicitOfEmployeesLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause restricts a party's soliciting or hiring employees and/or contractors from the counterparty, whether during the contract or after the contract ends (or both).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADNonCompeteLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause restricts the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADNonDisparagementLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause requires a party not to disparage the counterparty.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADNonTransferableLicenseLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause limits the ability of a party to transfer the license being granted to a third party.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADNoticePeriodToTerminateRenewalLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a notice period required to terminate renewal.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADPostTerminationServicesLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause subjects a party to obligations after the termination or expiration of a contract, including any post-termination transition, payment, transfer of IP, wind-down, last-buy, or similar commitments.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADPriceRestrictionsLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause places a restriction on the ability of a party to raise or reduce prices of technology, goods, or services provided.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADRenewalTermLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a renewal term.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADRevenueProfitSharingLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause require a party to share revenue or profit with the counterparty for any technology, goods, or services.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADRofrRofoRofnLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause grant one party a right of first refusal, right of first offer or right of first negotiation to purchase, license, market, or distribute equity interest, technology, assets, products or services.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADSourceCodeEscrowLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause requires one party to deposit its source code into escrow with a third party, which can be released to the counterparty upon the occurrence of certain events (bankruptcy, insolvency, etc.).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADTerminationForConvenienceLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that one party can terminate this contract without cause (solely by giving a notice and allowing a waiting period to expire).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADThirdPartyBeneficiaryLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that that there a non-contracting party who is a beneficiary to some or all of the clauses in the contract and therefore can enforce its rights against a contracting party.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADUncappedLiabilityLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that a party's liability is uncapped upon the breach of its obligation in the contract. This also includes uncap liability for a particular type of breach such as IP infringement or breach of confidentiality obligation.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADUnlimitedAllYouCanEatLicenseLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause grants one party an “enterprise,” “all you can eat” or unlimited usage license.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADVolumeRestrictionLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a fee increase or consent requirement, etc. if one party's use of the product/services exceeds certain threshold.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CUADWarrantyDurationLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a duration of any warranty against defects or errors in technology, products, or services provided under the contract.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CanadaTaxCourtOutcomesLegalBenchClassification

The input is an excerpt of text from Tax Court of Canada decisions involving appeals of tax related matters. The task is to classify whether the excerpt includes the outcome of the appeal, and if so, to specify whether the appeal was allowed or dismissed. Partial success (e.g. appeal granted on one tax year but dismissed on another) counts as allowed (with the exception of costs orders which are disregarded). Where the excerpt does not clearly articulate an outcome, the system should indicate other as the outcome. Categorizing case outcomes is a common task that legal researchers complete in order to gather datasets involving outcomes in legal processes for the purposes of quantitative empirical legal research.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CataloniaTweetClassification

This dataset contains two corpora in Spanish and Catalan that consist of annotated Twitter
        messages for automatic stance detection. The data was collected over 12 days during February and March
        of 2019 from tweets posted in Barcelona, and during September of 2018 from tweets posted in the town of Terrassa, Catalonia.
        Each corpus is annotated with three classes: AGAINST, FAVOR and NEUTRAL, which express the stance
        towards the target - independence of Catalonia.
        

**Dataset:** [`community-datasets/catalonia_independence`](https://huggingface.co/datasets/community-datasets/catalonia_independence) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2020.lrec-1.171/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cat, spa | Government, Social, Written | expert-annotated | created |



#### ContractNLIConfidentialityOfAgreementLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA provides that the Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLIExplicitIdentificationLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that all Confidential Information shall be expressly identified by the Disclosing Party.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLIInclusionOfVerballyConveyedInformationLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that Confidential Information may include verbally conveyed information.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLILimitedUseLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLINoLicensingLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Agreement shall not grant Receiving Party any right to Confidential Information.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLINoticeOnCompelledDisclosureLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLIPermissibleAcquirementOfSimilarInformationLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may acquire information similar to Confidential Information from a third party.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLIPermissibleCopyLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may create a copy of some Confidential Information in some circumstances.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLIPermissibleDevelopmentOfSimilarInformationLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may independently develop information similar to Confidential Information.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLIPermissiblePostAgreementPossessionLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLIReturnOfConfidentialInformationLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLISharingWithEmployeesLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may share some Confidential Information with some of Receiving Party's employees.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLISharingWithThirdPartiesLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ContractNLISurvivalOfObligationsLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that some obligations of Agreement may survive termination of Agreement.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CorporateLobbyingLegalBenchClassification

The Corporate Lobbying task consists of determining whether a proposed Congressional bill may be relevant to a company based on a company's self-description in its SEC 10K filing.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### CyrillicTurkicLangClassification

Cyrillic dataset of 8 Turkic languages spoken in Russia and former USSR

**Dataset:** [`tatiana-merz/cyrillic_turkic_langs`](https://huggingface.co/datasets/tatiana-merz/cyrillic_turkic_langs) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/tatiana-merz/cyrillic_turkic_langs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | bak, chv, kaz, kir, krc, ... (9) | Web, Written | derived | found |



#### CzechProductReviewSentimentClassification

User reviews of products on Czech e-shop Mall.cz with 3 sentiment classes (positive, neutral, negative)

**Dataset:** [`fewshot-goes-multilingual/cs_mall-product-reviews`](https://huggingface.co/datasets/fewshot-goes-multilingual/cs_mall-product-reviews) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/W13-1609/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | derived | found |



#### CzechProductReviewSentimentClassification.v2

User reviews of products on Czech e-shop Mall.cz with 3 sentiment classes (positive, neutral, negative)
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/czech_product_review_sentiment`](https://huggingface.co/datasets/mteb/czech_product_review_sentiment) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/W13-1609/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | derived | found |



#### CzechSoMeSentimentClassification

User comments on Facebook

**Dataset:** [`fewshot-goes-multilingual/cs_facebook-comments`](https://huggingface.co/datasets/fewshot-goes-multilingual/cs_facebook-comments) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/W13-1609/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | derived | found |



#### CzechSoMeSentimentClassification.v2

User comments on Facebook
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/czech_so_me_sentiment`](https://huggingface.co/datasets/mteb/czech_so_me_sentiment) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/W13-1609/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | derived | found |



#### CzechSubjectivityClassification

An Czech dataset for subjectivity classification.

**Dataset:** [`pauli31/czech-subjectivity-dataset`](https://huggingface.co/datasets/pauli31/czech-subjectivity-dataset) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2009.08712)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | human-annotated | found |



#### DBpediaClassification

DBpedia14 is a dataset of English texts from Wikipedia articles, categorized into 14 non-overlapping classes based on their DBpedia ontology.

**Dataset:** [`fancyzhx/dbpedia_14`](https://huggingface.co/datasets/fancyzhx/dbpedia_14) • **License:** cc-by-sa-3.0 • [Learn more →](https://arxiv.org/abs/1509.01626)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Encyclopaedic, Written | derived | found |



#### DBpediaClassification.v2

DBpedia14 is a dataset of English texts from Wikipedia articles, categorized into 14 non-overlapping classes based on their DBpedia ontology.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/d_bpedia`](https://huggingface.co/datasets/mteb/d_bpedia) • **License:** cc-by-sa-3.0 • [Learn more →](https://arxiv.org/abs/1509.01626)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Encyclopaedic, Written | derived | found |



#### DKHateClassification

Danish Tweets annotated for Hate Speech either being Offensive or not

**Dataset:** [`DDSC/dkhate`](https://huggingface.co/datasets/DDSC/dkhate) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2020.lrec-1.430/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Social, Written | expert-annotated | found |



#### DKHateClassification.v2

Danish Tweets annotated for Hate Speech either being Offensive or not
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/dk_hate`](https://huggingface.co/datasets/mteb/dk_hate) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2020.lrec-1.430/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Social, Written | expert-annotated | found |



#### DadoEvalCoarseClassification

The DaDoEval dataset is a curated collection of 2,759 documents authored by Alcide De Gasperi, spanning the period from 1901 to 1954. Each document in the dataset is manually tagged with its date of issue.

**Dataset:** [`MattiaSangermano/DaDoEval`](https://huggingface.co/datasets/MattiaSangermano/DaDoEval) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://github.com/dhfbk/DaDoEval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ita | Written | derived | found |



#### DalajClassification

A Swedish dataset for linguistic acceptability. Available as a part of Superlim.

**Dataset:** [`AI-Sweden/SuperLim`](https://huggingface.co/datasets/AI-Sweden/SuperLim) • **License:** cc-by-4.0 • [Learn more →](https://spraakbanken.gu.se/en/resources/superlim)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swe | Non-fiction, Written | expert-annotated | created |



#### DalajClassification.v2

A Swedish dataset for linguistic acceptability. Available as a part of Superlim.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/dalaj`](https://huggingface.co/datasets/mteb/dalaj) • **License:** cc-by-4.0 • [Learn more →](https://spraakbanken.gu.se/en/resources/superlim)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swe | Non-fiction, Written | expert-annotated | created |



#### DanishPoliticalCommentsClassification

A dataset of Danish political comments rated for sentiment

**Dataset:** [`community-datasets/danish_political_comments`](https://huggingface.co/datasets/community-datasets/danish_political_comments) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/danish_political_comments)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Social, Written | derived | found |



#### DanishPoliticalCommentsClassification.v2

A dataset of Danish political comments rated for sentiment
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/danish_political_comments`](https://huggingface.co/datasets/mteb/danish_political_comments) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/danish_political_comments)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Social, Written | derived | found |



#### Ddisco

A Danish Discourse dataset with values for coherence and source (Wikipedia or Reddit)

**Dataset:** [`DDSC/ddisco`](https://huggingface.co/datasets/DDSC/ddisco) • **License:** cc-by-sa-3.0 • [Learn more →](https://aclanthology.org/2022.lrec-1.260/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Non-fiction, Social, Written | expert-annotated | found |



#### Ddisco.v2

A Danish Discourse dataset with values for coherence and source (Wikipedia or Reddit)
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/ddisco_cohesion`](https://huggingface.co/datasets/mteb/ddisco_cohesion) • **License:** cc-by-sa-3.0 • [Learn more →](https://aclanthology.org/2022.lrec-1.260/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Non-fiction, Social, Written | expert-annotated | found |



#### DeepSentiPers

Persian Sentiment Analysis Dataset

**Dataset:** [`PartAI/DeepSentiPers`](https://huggingface.co/datasets/PartAI/DeepSentiPers) • **License:** not specified • [Learn more →](https://github.com/JoyeBright/DeepSentiPers)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Reviews | derived | found |



#### DeepSentiPers.v2

Persian Sentiment Analysis Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/deep_senti_pers`](https://huggingface.co/datasets/mteb/deep_senti_pers) • **License:** not specified • [Learn more →](https://github.com/JoyeBright/DeepSentiPers)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Reviews | derived | found |



#### DefinitionClassificationLegalBenchClassification

This task consists of determining whether or not a sentence from a Supreme Court opinion offers a definition of a term.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### DigikalamagClassification

A total of 8,515 articles scraped from Digikala Online Magazine. This dataset includes seven different classes.

**Dataset:** [`PNLPhub/DigiMag`](https://huggingface.co/datasets/PNLPhub/DigiMag) • **License:** not specified • [Learn more →](https://hooshvare.github.io/docs/datasets/tc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Web | derived | found |



#### Diversity1LegalBenchClassification

Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 1).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### Diversity2LegalBenchClassification

Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 2).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### Diversity3LegalBenchClassification

Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 3).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### Diversity4LegalBenchClassification

Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 4).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### Diversity5LegalBenchClassification

Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 5).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### Diversity6LegalBenchClassification

Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 6).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### DutchBookReviewSentimentClassification

A Dutch book review for sentiment classification.

**Dataset:** [`mteb/DutchBookReviewSentimentClassification`](https://huggingface.co/datasets/mteb/DutchBookReviewSentimentClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://github.com/benjaminvdb/DBRD)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nld | Reviews, Written | derived | found |



#### DutchBookReviewSentimentClassification.v2

A Dutch book review for sentiment classification.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/dutch_book_review_sentiment`](https://huggingface.co/datasets/mteb/dutch_book_review_sentiment) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://github.com/benjaminvdb/DBRD)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nld | Reviews, Written | derived | found |



#### EmotionClassification

Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise.

**Dataset:** [`mteb/emotion`](https://huggingface.co/datasets/mteb/emotion) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/D18-1404)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | human-annotated | found |



#### EmotionClassification.v2

Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/emotion`](https://huggingface.co/datasets/mteb/emotion) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/D18-1404)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | human-annotated | found |



#### EmotionVNClassification

Emotion is a translated dataset of Vietnamese from English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/emotion-vn`](https://huggingface.co/datasets/GreenNode/emotion-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.aclweb.org/anthology/D18-1404)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Social, Written | derived | machine-translated and LM verified |



#### EstonianValenceClassification

Dataset containing annotated Estonian news data from the Postimees and Õhtuleht newspapers.

**Dataset:** [`kardosdrur/estonian-valence`](https://huggingface.co/datasets/kardosdrur/estonian-valence) • **License:** cc-by-4.0 • [Learn more →](https://figshare.com/articles/dataset/Estonian_Valence_Corpus_Eesti_valentsikorpus/24517054)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | est | News, Written | human-annotated | found |



#### EstonianValenceClassification.v2

Dataset containing annotated Estonian news data from the Postimees and Õhtuleht newspapers.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/estonian_valence`](https://huggingface.co/datasets/mteb/estonian_valence) • **License:** cc-by-4.0 • [Learn more →](https://figshare.com/articles/dataset/Estonian_Valence_Corpus_Eesti_valentsikorpus/24517054)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | est | News, Written | human-annotated | found |



#### FilipinoHateSpeechClassification

Filipino Twitter dataset for sentiment classification.

**Dataset:** [`mteb/FilipinoHateSpeechClassification`](https://huggingface.co/datasets/mteb/FilipinoHateSpeechClassification) • **License:** not specified • [Learn more →](https://pcj.csp.org.ph/index.php/pcj/issue/download/29/PCJ%20V14%20N1%20pp1-14%202019)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fil | Social, Written | human-annotated | found |



#### FilipinoHateSpeechClassification.v2

Filipino Twitter dataset for sentiment classification.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/filipino_hate_speech`](https://huggingface.co/datasets/mteb/filipino_hate_speech) • **License:** not specified • [Learn more →](https://pcj.csp.org.ph/index.php/pcj/issue/download/29/PCJ%20V14%20N1%20pp1-14%202019)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fil | Social, Written | human-annotated | found |



#### FilipinoShopeeReviewsClassification

The Shopee reviews tl 15 dataset is constructed by randomly taking 2100 training samples and 450 samples for testing and validation for each review star from 1 to 5. In total, there are 10500 training samples and 2250 each in validation and testing samples.

**Dataset:** [`scaredmeow/shopee-reviews-tl-stars`](https://huggingface.co/datasets/scaredmeow/shopee-reviews-tl-stars) • **License:** mpl-2.0 • [Learn more →](https://uijrt.com/articles/v4/i8/UIJRTV4I80009.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fil | Social, Written | human-annotated | found |



#### FinToxicityClassification


        This dataset is a DeepL -based machine translated version of the Jigsaw toxicity dataset for Finnish. The dataset is originally from a Kaggle competition https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data.
        The original dataset poses a multi-label text classification problem and includes the labels identity_attack, insult, obscene, severe_toxicity, threat and toxicity.
        Here adapted for toxicity classification, which is the most represented class.
        

**Dataset:** [`TurkuNLP/jigsaw_toxicity_pred_fi`](https://huggingface.co/datasets/TurkuNLP/jigsaw_toxicity_pred_fi) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.68)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | fin | News, Written | derived | machine-translated |



#### FinToxicityClassification.v2


        This dataset is a DeepL -based machine translated version of the Jigsaw toxicity dataset for Finnish. The dataset is originally from a Kaggle competition https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data.
        The original dataset poses a multi-label text classification problem and includes the labels identity_attack, insult, obscene, severe_toxicity, threat and toxicity.
        Here adapted for toxicity classification, which is the most represented class.

        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/fin_toxicity`](https://huggingface.co/datasets/mteb/fin_toxicity) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.68)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | fin | News, Written | derived | machine-translated |



#### FinancialPhrasebankClassification

Polar sentiment dataset of sentences from financial news, categorized by sentiment into positive, negative, or neutral.

**Dataset:** [`takala/financial_phrasebank`](https://huggingface.co/datasets/takala/financial_phrasebank) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://arxiv.org/abs/1307.5336)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Financial, News, Written | expert-annotated | found |



#### FinancialPhrasebankClassification.v2

Polar sentiment dataset of sentences from financial news, categorized by sentiment into positive, negative, or neutral.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/financial_phrasebank`](https://huggingface.co/datasets/mteb/financial_phrasebank) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://arxiv.org/abs/1307.5336)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Financial, News, Written | expert-annotated | found |



#### FrenchBookReviews

It is a French book reviews dataset containing a huge number of reader reviews on French books. Each review is pared with a rating that ranges from 0.5 to 5 (with 0.5 increment).

**Dataset:** [`Abirate/french_book_reviews`](https://huggingface.co/datasets/Abirate/french_book_reviews) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/Abirate/french_book_reviews)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fra | Reviews, Written | derived | found |



#### FrenchBookReviews.v2

It is a French book reviews dataset containing a huge number of reader reviews on French books. Each review is pared with a rating that ranges from 0.5 to 5 (with 0.5 increment).
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/french_book_reviews`](https://huggingface.co/datasets/mteb/french_book_reviews) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/Abirate/french_book_reviews)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fra | Reviews, Written | derived | found |



#### FrenkEnClassification

English subset of the FRENK dataset

**Dataset:** [`classla/FRENK-hate-en`](https://huggingface.co/datasets/classla/FRENK-hate-en) • **License:** not specified • [Learn more →](https://arxiv.org/abs/1906.02045)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | derived | found |



#### FrenkEnClassification.v2

English subset of the FRENK dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/frenk_en`](https://huggingface.co/datasets/mteb/frenk_en) • **License:** not specified • [Learn more →](https://arxiv.org/abs/1906.02045)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | derived | found |



#### FrenkHrClassification

Croatian subset of the FRENK dataset

**Dataset:** [`classla/FRENK-hate-hr`](https://huggingface.co/datasets/classla/FRENK-hate-hr) • **License:** not specified • [Learn more →](https://arxiv.org/abs/1906.02045)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | hrv | Social, Written | derived | found |



#### FrenkHrClassification.v2

Croatian subset of the FRENK dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/frenk_hr`](https://huggingface.co/datasets/mteb/frenk_hr) • **License:** not specified • [Learn more →](https://arxiv.org/abs/1906.02045)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | hrv | Social, Written | derived | found |



#### FrenkSlClassification

Slovenian subset of the FRENK dataset. Also available on HuggingFace dataset hub: English subset, Croatian subset.

**Dataset:** [`classla/FRENK-hate-sl`](https://huggingface.co/datasets/classla/FRENK-hate-sl) • **License:** not specified • [Learn more →](https://arxiv.org/pdf/1906.02045)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | slv | Social, Written | derived | found |



#### FrenkSlClassification.v2

Slovenian subset of the FRENK dataset. Also available on HuggingFace dataset hub: English subset, Croatian subset.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/frenk_sl`](https://huggingface.co/datasets/mteb/frenk_sl) • **License:** not specified • [Learn more →](https://arxiv.org/pdf/1906.02045)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | slv | Social, Written | derived | found |



#### FunctionOfDecisionSectionLegalBenchClassification

The task is to classify a paragraph extracted from a written court decision into one of seven possible categories:
            1. Facts - The paragraph describes the faction background that led up to the present lawsuit.
            2. Procedural History - The paragraph describes the course of litigation that led to the current proceeding before the court.
            3. Issue - The paragraph describes the legal or factual issue that must be resolved by the court.
            4. Rule - The paragraph describes a rule of law relevant to resolving the issue.
            5. Analysis - The paragraph analyzes the legal issue by applying the relevant legal principles to the facts of the present dispute.
            6. Conclusion - The paragraph presents a conclusion of the court.
            7. Decree - The paragraph constitutes a decree resolving the dispute.
        

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### GeoreviewClassification

Review classification (5-point scale) based on Yandex Georeview dataset

**Dataset:** [`ai-forever/georeview-classification`](https://huggingface.co/datasets/ai-forever/georeview-classification) • **License:** mit • [Learn more →](https://github.com/yandex/geo-reviews-dataset-2023)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Reviews, Written | derived | found |



#### GeoreviewClassification.v2

Review classification (5-point scale) based on Yandex Georeview dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/georeview`](https://huggingface.co/datasets/mteb/georeview) • **License:** mit • [Learn more →](https://github.com/yandex/geo-reviews-dataset-2023)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Reviews, Written | derived | found |



#### GeorgianSentimentClassification

Goergian Sentiment Dataset

**Dataset:** [`asparius/Georgian-Sentiment`](https://huggingface.co/datasets/asparius/Georgian-Sentiment) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2022.lrec-1.173)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kat | Reviews, Written | derived | found |



#### GermanPoliticiansTwitterSentimentClassification

GermanPoliticiansTwitterSentiment is a dataset of German tweets categorized with their sentiment (3 classes).

**Dataset:** [`Alienmaster/german_politicians_twitter_sentiment`](https://huggingface.co/datasets/Alienmaster/german_politicians_twitter_sentiment) • **License:** not specified • [Learn more →](https://aclanthology.org/2022.konvens-1.9)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu | Government, Social, Written | human-annotated | found |



#### GermanPoliticiansTwitterSentimentClassification.v2

GermanPoliticiansTwitterSentiment is a dataset of German tweets categorized with their sentiment (3 classes).
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/german_politicians_twitter_sentiment`](https://huggingface.co/datasets/mteb/german_politicians_twitter_sentiment) • **License:** not specified • [Learn more →](https://aclanthology.org/2022.konvens-1.9)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu | Government, Social, Written | human-annotated | found |



#### GreekLegalCodeClassification

Greek Legal Code Dataset for Classification. (subset = chapter)

**Dataset:** [`AI-team-UoA/greek_legal_code`](https://huggingface.co/datasets/AI-team-UoA/greek_legal_code) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2109.15298)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ell | Legal, Written | human-annotated | found |



#### GujaratiNewsClassification

A Gujarati dataset for 3-class classification of Gujarati news articles

**Dataset:** [`mlexplorer008/gujarati_news_classification`](https://huggingface.co/datasets/mlexplorer008/gujarati_news_classification) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-gujarati)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | guj | News, Written | derived | found |



#### GujaratiNewsClassification.v2

A Gujarati dataset for 3-class classification of Gujarati news articles
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/gujarati_news`](https://huggingface.co/datasets/mteb/gujarati_news) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-gujarati)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | guj | News, Written | derived | found |



#### HateSpeechPortugueseClassification

HateSpeechPortugueseClassification is a dataset of Portuguese tweets categorized with their sentiment (2 classes).

**Dataset:** [`hate-speech-portuguese/hate_speech_portuguese`](https://huggingface.co/datasets/hate-speech-portuguese/hate_speech_portuguese) • **License:** not specified • [Learn more →](https://aclanthology.org/W19-3510)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | por | Social, Written | expert-annotated | found |



#### HeadlineClassification

Headline rubric classification based on the paraphraser plus dataset.

**Dataset:** [`ai-forever/headline-classification`](https://huggingface.co/datasets/ai-forever/headline-classification) • **License:** mit • [Learn more →](https://aclanthology.org/2020.ngt-1.6/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | News, Written | derived | found |



#### HeadlineClassification.v2

Headline rubric classification based on the paraphraser plus dataset.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/headline`](https://huggingface.co/datasets/mteb/headline) • **License:** mit • [Learn more →](https://aclanthology.org/2020.ngt-1.6/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | News, Written | derived | found |



#### HebrewSentimentAnalysis

HebrewSentiment is a data set consists of 12,804 user comments to posts on the official Facebook page of Israel’s president, Mr. Reuven Rivlin. In October 2015, we used the open software application Netvizz (Rieder, 2013) to scrape all the comments to all of the president’s posts in the period of June – August 2014, the first three months of Rivlin’s presidency.2 While the president’s posts aimed at reconciling tensions and called for tolerance and empathy, the sentiment expressed in the comments to the president’s posts was polarized between citizens who warmly thanked the president, and citizens that fiercely critiqued his policy.

**Dataset:** [`omilab/hebrew_sentiment`](https://huggingface.co/datasets/omilab/hebrew_sentiment) • **License:** mit • [Learn more →](https://huggingface.co/datasets/hebrew_sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | heb | Reviews, Written | expert-annotated | found |



#### HebrewSentimentAnalysis.v2

HebrewSentiment is a data set consists of 12,804 user comments to posts on the official Facebook page of Israel’s president, Mr. Reuven Rivlin. In October 2015, we used the open software application Netvizz (Rieder, 2013) to scrape all the comments to all of the president’s posts in the period of June – August 2014, the first three months of Rivlin’s presidency.2 While the president’s posts aimed at reconciling tensions and called for tolerance and empathy, the sentiment expressed in the comments to the president’s posts was polarized between citizens who warmly thanked the president, and citizens that fiercely critiqued his policy.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/hebrew_sentiment_analysis`](https://huggingface.co/datasets/mteb/hebrew_sentiment_analysis) • **License:** mit • [Learn more →](https://huggingface.co/datasets/hebrew_sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | heb | Reviews, Written | expert-annotated | found |



#### HinDialectClassification

HinDialect: 26 Hindi-related languages and dialects of the Indic Continuum in North India

**Dataset:** [`mlexplorer008/hin_dialect_classification`](https://huggingface.co/datasets/mlexplorer008/hin_dialect_classification) • **License:** cc-by-sa-4.0 • [Learn more →](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4839)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | anp, awa, ben, bgc, bhb, ... (21) | Social, Spoken, Written | expert-annotated | found |



#### HindiDiscourseClassification

A Hindi Discourse dataset in Hindi with values for coherence.

**Dataset:** [`midas/hindi_discourse`](https://huggingface.co/datasets/midas/hindi_discourse) • **License:** mit • [Learn more →](https://aclanthology.org/2020.lrec-1.149/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | hin | Fiction, Social, Written | expert-annotated | found |



#### HindiDiscourseClassification.v2

A Hindi Discourse dataset in Hindi with values for coherence.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/hindi_discourse`](https://huggingface.co/datasets/mteb/hindi_discourse) • **License:** mit • [Learn more →](https://aclanthology.org/2020.lrec-1.149/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | hin | Fiction, Social, Written | expert-annotated | found |



#### HotelReviewSentimentClassification

HARD is a dataset of Arabic hotel reviews collected from the Booking.com website.

**Dataset:** [`mteb/HotelReviewSentimentClassification`](https://huggingface.co/datasets/mteb/HotelReviewSentimentClassification) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-67056-0_3)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Reviews, Written | derived | found |



#### HotelReviewSentimentClassification.v2

HARD is a dataset of Arabic hotel reviews collected from the Booking.com website.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/HotelReviewSentimentClassification`](https://huggingface.co/datasets/mteb/HotelReviewSentimentClassification) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-67056-0_3)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Reviews, Written | derived | found |



#### IFlyTek

Long Text classification for the description of Apps

**Dataset:** [`C-MTEB/IFlyTek-classification`](https://huggingface.co/datasets/C-MTEB/IFlyTek-classification) • **License:** not specified • [Learn more →](https://www.cluebenchmarks.com/introduce.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### IFlyTek.v2

Long Text classification for the description of Apps
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/i_fly_tek`](https://huggingface.co/datasets/mteb/i_fly_tek) • **License:** not specified • [Learn more →](https://www.cluebenchmarks.com/introduce.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### ImdbClassification

Large Movie Review Dataset

**Dataset:** [`mteb/imdb`](https://huggingface.co/datasets/mteb/imdb) • **License:** not specified • [Learn more →](http://www.aclweb.org/anthology/P11-1015)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | derived | found |



#### ImdbClassification.v2

Large Movie Review Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/imdb`](https://huggingface.co/datasets/mteb/imdb) • **License:** not specified • [Learn more →](http://www.aclweb.org/anthology/P11-1015)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | derived | found |



#### ImdbVNClassification

A translated dataset of large movie reviews annotated for sentiment classification.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/imdb-vn`](https://huggingface.co/datasets/GreenNode/imdb-vn) • **License:** cc-by-sa-4.0 • [Learn more →](http://www.aclweb.org/anthology/P11-1015)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Reviews, Written | derived | machine-translated and LM verified |



#### InappropriatenessClassification

Inappropriateness identification in the form of binary classification

**Dataset:** [`ai-forever/inappropriateness-classification`](https://huggingface.co/datasets/ai-forever/inappropriateness-classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/2021.bsnlp-1.4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Social, Web, Written | human-annotated | found |



#### InappropriatenessClassification.v2

Inappropriateness identification in the form of binary classification
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/inappropriateness`](https://huggingface.co/datasets/mteb/inappropriateness) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/2021.bsnlp-1.4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Social, Web, Written | human-annotated | found |



#### InappropriatenessClassificationv2

Inappropriateness identification in the form of binary classification

**Dataset:** [`mteb/InappropriatenessClassificationv2`](https://huggingface.co/datasets/mteb/InappropriatenessClassificationv2) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/2021.bsnlp-1.4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | rus | Social, Web, Written | human-annotated | found |



#### IndicLangClassification

A language identification test set for native-script as well as Romanized text which spans 22 Indic languages.

**Dataset:** [`ai4bharat/Bhasha-Abhijnaanam`](https://huggingface.co/datasets/ai4bharat/Bhasha-Abhijnaanam) • **License:** cc0-1.0 • [Learn more →](https://arxiv.org/abs/2305.15814)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | asm, ben, brx, doi, gom, ... (22) | Non-fiction, Web, Written | expert-annotated | created |



#### IndicNLPNewsClassification

A News classification dataset in multiple Indian regional languages.

**Dataset:** [`Sakshamrzt/IndicNLP-Multilingual`](https://huggingface.co/datasets/Sakshamrzt/IndicNLP-Multilingual) • **License:** cc-by-nc-4.0 • [Learn more →](https://github.com/AI4Bharat/indicnlp_corpus#indicnlp-news-article-classification-dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | guj, kan, mal, mar, ori, ... (8) | News, Written | expert-annotated | found |



#### IndicSentimentClassification

A new, multilingual, and n-way parallel dataset for sentiment analysis in 13 Indic languages.

**Dataset:** [`mteb/IndicSentiment`](https://huggingface.co/datasets/mteb/IndicSentiment) • **License:** cc0-1.0 • [Learn more →](https://arxiv.org/abs/2212.05409)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | asm, ben, brx, guj, hin, ... (13) | Reviews, Written | human-annotated | machine-translated and verified |



#### IndonesianIdClickbaitClassification

The CLICK-ID dataset is a collection of Indonesian news headlines that was collected from 12 local online news publishers.

**Dataset:** [`manandey/id_clickbait`](https://huggingface.co/datasets/manandey/id_clickbait) • **License:** cc-by-4.0 • [Learn more →](http://www.sciencedirect.com/science/article/pii/S2352340920311252)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ind | News, Written | expert-annotated | found |



#### IndonesianIdClickbaitClassification.v2

The CLICK-ID dataset is a collection of Indonesian news headlines that was collected from 12 local online news publishers.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/indonesian_id_clickbait`](https://huggingface.co/datasets/mteb/indonesian_id_clickbait) • **License:** cc-by-4.0 • [Learn more →](http://www.sciencedirect.com/science/article/pii/S2352340920311252)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ind | News, Written | expert-annotated | found |



#### IndonesianMongabayConservationClassification

Conservation dataset that was collected from mongabay.co.id contains topic-classification task (multi-label format) and sentiment classification. This task only covers sentiment analysis (positive, neutral negative)

**Dataset:** [`Datasaur/mongabay-experiment`](https://huggingface.co/datasets/Datasaur/mongabay-experiment) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.sealp-1.4/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ind | Web, Written | derived | found |



#### IndonesianMongabayConservationClassification.v2

Conservation dataset that was collected from mongabay.co.id contains topic-classification task (multi-label format) and sentiment classification. This task only covers sentiment analysis (positive, neutral negative)
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/indonesian_mongabay_conservation`](https://huggingface.co/datasets/mteb/indonesian_mongabay_conservation) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.sealp-1.4/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ind | Web, Written | derived | found |



#### InsurancePolicyInterpretationLegalBenchClassification

Given an insurance claim and policy, determine whether the claim is covered by the policy.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### InternationalCitizenshipQuestionsLegalBenchClassification

Answer questions about citizenship law from across the world. Dataset was made using the GLOBALCIT citizenship law dataset, by constructing questions about citizenship law as Yes or No questions.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### IsiZuluNewsClassification

isiZulu News Classification Dataset

**Dataset:** [`isaacchung/isizulu-news`](https://huggingface.co/datasets/isaacchung/isizulu-news) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/dsfsi/za-isizulu-siswati-news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | zul | News, Written | human-annotated | found |



#### IsiZuluNewsClassification.v2

isiZulu News Classification Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/isi_zulu_news`](https://huggingface.co/datasets/mteb/isi_zulu_news) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/dsfsi/za-isizulu-siswati-news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | zul | News, Written | human-annotated | found |



#### ItaCaseholdClassification

An Italian Dataset consisting of 1101 pairs of judgments and their official holdings between the years 2019 and 2022 from the archives of Italian Administrative Justice categorized with 64 subjects.

**Dataset:** [`itacasehold/itacasehold`](https://huggingface.co/datasets/itacasehold/itacasehold) • **License:** apache-2.0 • [Learn more →](https://doi.org/10.1145/3594536.3595177)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ita | Government, Legal, Written | expert-annotated | found |



#### Itacola

An Italian Corpus of Linguistic Acceptability taken from linguistic literature with a binary annotation made by the original authors themselves.

**Dataset:** [`gsarti/itacola`](https://huggingface.co/datasets/gsarti/itacola) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.findings-emnlp.250/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ita | Non-fiction, Spoken, Written | expert-annotated | found |



#### Itacola.v2

An Italian Corpus of Linguistic Acceptability taken from linguistic literature with a binary annotation made by the original authors themselves.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/italian_linguistic_acceptability`](https://huggingface.co/datasets/mteb/italian_linguistic_acceptability) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.findings-emnlp.250/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ita | Non-fiction, Spoken, Written | expert-annotated | found |



#### JCrewBlockerLegalBenchClassification

The J.Crew Blocker, also known as the J.Crew Protection, is a provision included in leveraged loan documents to prevent companies from removing security by transferring intellectual property (IP) into new subsidiaries and raising additional debt. The task consists of detemining whether the J.Crew Blocker is present in the document.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### JCrewBlockerLegalBenchClassification.v2

The J.Crew Blocker, also known as the J.Crew Protection, is a provision included in leveraged loan documents to prevent companies from removing security by transferring intellectual property (IP) into new subsidiaries and raising additional debt. The task consists of detemining whether the J.Crew Blocker is present in the document.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/j_crew_blocker_legal_bench`](https://huggingface.co/datasets/mteb/j_crew_blocker_legal_bench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### JDReview

review for iphone

**Dataset:** [`C-MTEB/JDReview-classification`](https://huggingface.co/datasets/C-MTEB/JDReview-classification) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### JDReview.v2

review for iphone
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/jd_review`](https://huggingface.co/datasets/mteb/jd_review) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### JapaneseSentimentClassification

Japanese sentiment classification dataset with binary
                       (positive vs negative sentiment) labels. This version reverts
                       the morphological analysis from the original multilingual dataset
                       to restore natural Japanese text without artificial spaces.
                     

**Dataset:** [`sbintuitions/JMTEB`](https://huggingface.co/datasets/sbintuitions/JMTEB) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb/multilingual-sentiment-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | jpn | Reviews, Written | derived | found |



#### JavaneseIMDBClassification

Large Movie Review Dataset translated to Javanese. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets.

**Dataset:** [`w11wo/imdb-javanese`](https://huggingface.co/datasets/w11wo/imdb-javanese) • **License:** mit • [Learn more →](https://github.com/w11wo/nlp-datasets#javanese-imdb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | jav | Reviews, Written | human-annotated | found |



#### JavaneseIMDBClassification.v2

Large Movie Review Dataset translated to Javanese. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/javanese_imdb`](https://huggingface.co/datasets/mteb/javanese_imdb) • **License:** mit • [Learn more →](https://github.com/w11wo/nlp-datasets#javanese-imdb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | jav | Reviews, Written | human-annotated | found |



#### KLUE-TC

Topic classification dataset of human-annotated news headlines. Part of the Korean Language Understanding Evaluation (KLUE).

**Dataset:** [`klue/klue`](https://huggingface.co/datasets/klue/klue) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2105.09680)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | News, Written | human-annotated | found |



#### KLUE-TC.v2

Topic classification dataset of human-annotated news headlines. Part of the Korean Language Understanding Evaluation (KLUE).
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/klue_tc`](https://huggingface.co/datasets/mteb/klue_tc) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2105.09680)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | News, Written | human-annotated | found |



#### KannadaNewsClassification

The Kannada news dataset contains only the headlines of news article in three categories: Entertainment, Tech, and Sports. The data set contains around 6300 news article headlines which are collected from Kannada news websites. The data set has been cleaned and contains train and test set using which can be used to benchmark topic classification models in Kannada.

**Dataset:** [`Akash190104/kannada_news_classification`](https://huggingface.co/datasets/Akash190104/kannada_news_classification) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/goru001/nlp-for-kannada)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kan | News, Written | derived | found |



#### KannadaNewsClassification.v2

The Kannada news dataset contains only the headlines of news article in three categories: Entertainment, Tech, and Sports. The data set contains around 6300 news article headlines which are collected from Kannada news websites. The data set has been cleaned and contains train and test set using which can be used to benchmark topic classification models in Kannada.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/kannada_news`](https://huggingface.co/datasets/mteb/kannada_news) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/goru001/nlp-for-kannada)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kan | News, Written | derived | found |



#### KinopoiskClassification

Kinopoisk review sentiment classification

**Dataset:** [`ai-forever/kinopoisk-sentiment-classification`](https://huggingface.co/datasets/ai-forever/kinopoisk-sentiment-classification) • **License:** not specified • [Learn more →](https://www.dialog-21.ru/media/1226/blinovpd.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Reviews, Written | derived | found |



#### KorFin

The KorFin-ASC is an extension of KorFin-ABSA, which is a financial sentiment analysis dataset including 8818 samples with (aspect, polarity) pairs annotated. The samples were collected from KLUE-TC and analyst reports from Naver Finance.

**Dataset:** [`amphora/korfin-asc`](https://huggingface.co/datasets/amphora/korfin-asc) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/amphora/korfin-asc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | Financial, News, Written | expert-annotated | found |



#### KorHateClassification

The dataset was created to provide the first human-labeled Korean corpus for
        toxic speech detection from a Korean online entertainment news aggregator. Recently,
        two young Korean celebrities suffered from a series of tragic incidents that led to two
        major Korean web portals to close the comments section on their platform. However, this only
        serves as a temporary solution, and the fundamental issue has not been solved yet. This dataset
        hopes to improve Korean hate speech detection. Annotation was performed by 32 annotators,
        consisting of 29 annotators from the crowdsourcing platform DeepNatural AI and three NLP researchers.
        

**Dataset:** [`inmoonlight/kor_hate`](https://huggingface.co/datasets/inmoonlight/kor_hate) • **License:** cc-by-sa-4.0 • [Learn more →](https://paperswithcode.com/dataset/korean-hatespeech-dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | Social, Written | expert-annotated | found |



#### KorHateClassification.v2

The dataset was created to provide the first human-labeled Korean corpus for
        toxic speech detection from a Korean online entertainment news aggregator. Recently,
        two young Korean celebrities suffered from a series of tragic incidents that led to two
        major Korean web portals to close the comments section on their platform. However, this only
        serves as a temporary solution, and the fundamental issue has not been solved yet. This dataset
        hopes to improve Korean hate speech detection. Annotation was performed by 32 annotators,
        consisting of 29 annotators from the crowdsourcing platform DeepNatural AI and three NLP researchers.

        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/kor_hate`](https://huggingface.co/datasets/mteb/kor_hate) • **License:** cc-by-sa-4.0 • [Learn more →](https://paperswithcode.com/dataset/korean-hatespeech-dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | Social, Written | expert-annotated | found |



#### KorSarcasmClassification


        The Korean Sarcasm Dataset was created to detect sarcasm in text, which can significantly alter the original
        meaning of a sentence. 9319 tweets were collected from Twitter and labeled for sarcasm or not_sarcasm. These
        tweets were gathered by querying for: irony sarcastic, and
        sarcasm.
        The dataset was created by gathering HTML data from Twitter. Queries for hashtags that include sarcasm
        and variants of it were used to return tweets. It was preprocessed by removing the keyword
        hashtag, urls and mentions of the user to preserve anonymity.
        

**Dataset:** [`SpellOnYou/kor_sarcasm`](https://huggingface.co/datasets/SpellOnYou/kor_sarcasm) • **License:** mit • [Learn more →](https://github.com/SpellOnYou/korean-sarcasm)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | Social, Written | expert-annotated | found |



#### KorSarcasmClassification.v2


        The Korean Sarcasm Dataset was created to detect sarcasm in text, which can significantly alter the original
        meaning of a sentence. 9319 tweets were collected from Twitter and labeled for sarcasm or not_sarcasm. These
        tweets were gathered by querying for: irony sarcastic, and
        sarcasm.
        The dataset was created by gathering HTML data from Twitter. Queries for hashtags that include sarcasm
        and variants of it were used to return tweets. It was preprocessed by removing the keyword
        hashtag, urls and mentions of the user to preserve anonymity.

        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/kor_sarcasm`](https://huggingface.co/datasets/mteb/kor_sarcasm) • **License:** mit • [Learn more →](https://github.com/SpellOnYou/korean-sarcasm)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | Social, Written | expert-annotated | found |



#### KurdishSentimentClassification

Kurdish Sentiment Dataset

**Dataset:** [`asparius/Kurdish-Sentiment`](https://huggingface.co/datasets/asparius/Kurdish-Sentiment) • **License:** cc-by-4.0 • [Learn more →](https://link.springer.com/article/10.1007/s10579-023-09716-6)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kur | Web, Written | derived | found |



#### KurdishSentimentClassification.v2

Kurdish Sentiment Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/kurdish_sentiment`](https://huggingface.co/datasets/mteb/kurdish_sentiment) • **License:** cc-by-4.0 • [Learn more →](https://link.springer.com/article/10.1007/s10579-023-09716-6)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kur | Web, Written | derived | found |



#### LanguageClassification

A language identification dataset for 20 languages.

**Dataset:** [`papluca/language-identification`](https://huggingface.co/datasets/papluca/language-identification) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/papluca/language-identification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara, bul, cmn, deu, ell, ... (20) | Fiction, Government, Non-fiction, Reviews, Web, ... (6) | derived | found |



#### LccSentimentClassification

The leipzig corpora collection, annotated for sentiment

**Dataset:** [`DDSC/lcc`](https://huggingface.co/datasets/DDSC/lcc) • **License:** cc-by-4.0 • [Learn more →](https://github.com/fnielsen/lcc-sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | News, Web, Written | expert-annotated | found |



#### LearnedHandsBenefitsLegalBenchClassification

This is a binary classification task in which the model must determine if a user's legal post discusses public benefits and social services that people can get from the government, like for food, disability, old age, housing, medical help, unemployment, child care, or other social needs.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsBusinessLegalBenchClassification

This is a binary classification task in which the model must determine if a user's legal question discusses issues faced by people who run small businesses or nonprofits, including around incorporation, licenses, taxes, regulations, and other concerns. It also includes options when there are disasters, bankruptcies, or other problems.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsConsumerLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues people face regarding money, insurance, consumer goods and contracts, taxes, and small claims about quality of service.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsCourtsLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses the logistics of how a person can interact with a lawyer or the court system. It applies to situations about procedure, rules, how to file lawsuits, how to hire lawyers, how to represent oneself, and other practical matters about dealing with these systems.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsCrimeLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues in the criminal system including when people are charged with crimes, go to a criminal trial, go to prison, or are a victim of a crime.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsDivorceLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues around filing for divorce, separation, or annulment, getting spousal support, splitting money and property, and following the court processes.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsDomesticViolenceLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses dealing with domestic violence and abuse, including getting protective orders, enforcing them, understanding abuse, reporting abuse, and getting resources and status if there is abuse.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsEducationLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues around school, including accommodations for special needs, discrimination, student debt, discipline, and other issues in education.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsEmploymentLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues related to working at a job, including discrimination and harassment, worker's compensation, workers rights, unions, getting paid, pensions, being fired, and more.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsEstatesLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses planning for end-of-life, possible incapacitation, and other special circumstances that would prevent a person from making decisions about their own well-being, finances, and property. This includes issues around wills, powers of attorney, advance directives, trusts, guardianships, conservatorships, and other estate issues that people and families deal with.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsFamilyLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues that arise within a family, like divorce, adoption, name change, guardianship, domestic violence, child custody, and other issues.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsHealthLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues with accessing health services, paying for medical care, getting public benefits for health care, protecting one's rights in medical settings, and other issues related to health.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsHousingLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues with paying your rent or mortgage, landlord-tenant issues, housing subsidies and public housing, eviction, and other problems with your apartment, mobile home, or house.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsImmigrationLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses visas, asylum, green cards, citizenship, migrant work and benefits, and other issues faced by people who are not full citizens in the US.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsTortsLegalBenchClassification

This is a binary classification task in which the model must determine if a user's legal question discusses problems that one person has with another person (or animal), like when there is a car accident, a dog bite, bullying or possible harassment, or neighbors treating each other badly.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LearnedHandsTrafficLegalBenchClassification

This is a binary classification task in which the model must determine if a user's legal post discusses problems with traffic and parking tickets, fees, driver's licenses, and other issues experienced with the traffic system. It also concerns issues with car accidents and injuries, cars' quality, repairs, purchases, and other contracts.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LegalReasoningCausalityLegalBenchClassification

Given an excerpt from a district court opinion, classify if it relies on statistical evidence in its reasoning.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### LegalReasoningCausalityLegalBenchClassification.v2

Given an excerpt from a district court opinion, classify if it relies on statistical evidence in its reasoning.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/legal_reasoning_causality_legal_bench`](https://huggingface.co/datasets/mteb/legal_reasoning_causality_legal_bench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### MAUDLegalBenchClassification

This task was constructed from the MAUD dataset, which consists of over 47,000 labels across 152 merger agreements annotated to identify 92 questions in each agreement used by the 2021 American Bar Association (ABA) Public Target Deal Points Study. Each dataset is formatted as a series of multiple-choice questions, where given a segment of the merger agreement and a Deal Point question, the model is to choose the answer that best characterizes the agreement as response.

        This is a combination of all 34 of the MAUD Legal Bench datasets:
        1. MAUD Ability To Consummate Concept Is Subject To MAE Carveouts: Given an excerpt from a merger agreement and the task is to answer: is the “ability to consummate” concept subject to Material Adverse Effect (MAE) carveouts? amongst the multiple choice options.
        2. MAUD Accuracy Of Fundamental Target RWS Bringdown Standard: Given an excerpt from a merger agreement and the task is to answer: how accurate must the fundamental representations and warranties be according to the bring down provision, amongst the multiple choice options.
        3. MAUD Accuracy Of Target Capitalization RW Outstanding Shares Bringdown Standard Answer: Given an excerpt from a merger agreement and the task is to answer: how accurate must the fundamental representations and warranties be according to the bring down provision, amongst the multiple choice options.
        4. MAUD Accuracy Of Target General RW Bringdown Timing Answer: Given an excerpt from a merger agreement and the task is to answer: how accurate must the fundamental representations and warranties be according to the bring down provision, amongst the multiple choice options.
        5. MAUD Additional Matching Rights Period For Modifications Cor: Given an excerpt from a merger agreement and the task is to answer: how long is the additional matching rights period for modifications in case the board changes its recommendation, amongst the multiple choice options.
        6. MAUD Application Of Buyer Consent Requirement Negative Interim Covenant: Given an excerpt from a merger agreement and the task is to answer: what negative covenants does the requirement of Buyer consent apply to, amongst the multiple choice options.
        7. MAUD Buyer Consent Requirement Ordinary Course: Given an excerpt from a merger agreement and the task is to answer: in case the Buyer's consent for the acquired company's ordinary business operations is required, are there any limitations on the Buyer's right to condition, withhold, or delay their consent, amongst the multiple choice options.
        8. MAUD Change In Law Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do changes in law that have disproportionate impact qualify for Material Adverse Effect (MAE), amongst the multiple choice options.
        9. MAUD Changes In GAAP Or Other Accounting Principles Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do changes in GAAP or other accounting principles that have disproportionate impact qualify for Material Adverse Effect (MAE), amongst the multiple choice options.
        10. MAUD COR Permitted In Response To Intervening Event: Given an excerpt from a merger agreement and the task is to answer: is Change of Recommendation permitted in response to an intervening event, amongst the multiple choice options.
        11. MAUD COR Permitted With Board Fiduciary Determination Only: Given an excerpt from a merger agreement and the task is to answer: is Change of Recommendation permitted as long as the board determines that such change is required to fulfill its fiduciary obligations, amongst the multiple choice options.
        12. MAUD COR Standard Intervening Event: Given an excerpt from a merger agreement and the task is to answer: what standard should the board follow when determining whether to change its recommendation in response to an intervening event, amongst the multiple choice options.
        13. MAUD COR Standard Superior Offer: Given an excerpt from a merger agreement and the task is to answer: what standard should the board follow when determining whether to change its recommendation in connection with a superior offer, amongst the multiple choice options.
        14. MAUD Definition Contains Knowledge Requirement Answer: Given an excerpt from a merger agreement and the task is to answer: what is the knowledge requirement in the definition of “Intervening Event”, amongst the multiple choice options.
        15. MAUD Definition Includes Asset Deals: Given an excerpt from a merger agreement and the task is to answer: what qualifies as a superior offer in terms of asset deals, amongst the multiple choice options.
        16. MAUD Definition Includes Stock Deals: Given an excerpt from a merger agreement and the task is to answer: what qualifies as a superior offer in terms of stock deals, amongst the multiple choice options.
        17. MAUD Fiduciary Exception Board Determination Standard: Given an excerpt from a merger agreement and the task is to answer: under what circumstances could the Board take actions on a different acquisition proposal notwithstanding the no-shop provision, amongst the multiple choice options.
        18. MAUD Fiduciary Exception Board Determination Trigger No Shop: Given an excerpt from a merger agreement and the task is to answer: what type of offer could the Board take actions on notwithstanding the no-shop provision, amongst the multiple choice options.
        19. MAUD Financial Point Of View Is The Sole Consideration: Given an excerpt from a merger agreement and the task is to answer: is “financial point of view” the sole consideration when determining whether an offer is superior, amongst the multiple choice options.
        20. MAUD FLS MAE Standard: Given an excerpt from a merger agreement and the task is to answer: what is the Forward Looking Standard (FLS) with respect to Material Adverse Effect (MAE), amongst the multiple choice options.
        21. MAUD General Economic and Financial Conditions Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do changes caused by general economic and financial conditions that have disproportionate impact qualify for Material Adverse Effect (MAE), amongst the multiple choice options.
        22. MAUD Includes Consistent With Past Practice: Given an excerpt from a merger agreement and the task is to answer: does the wording of the Efforts Covenant clause include “consistent with past practice”, amongst the multiple choice options.
        23. MAUD Initial Matching Rights Period COR: Given an excerpt from a merger agreement and the task is to answer: how long is the initial matching rights period in case the board changes its recommendation, amongst the multiple choice options.
        24. MAUD Initial Matching Rights Period FTR: Given an excerpt from a merger agreement and the task is to answer: how long is the initial matching rights period in connection with the Fiduciary Termination Right (FTR), amongst the multiple choice options.
        25. MAUDInterveningEventRequiredToOccurAfterSigningAnswer: Given an excerpt from a merger agreement and the task is to answer: is an “Intervening Event” required to occur after signing, amongst the multiple choice options.
        26. MAUD Knowledge Definition: Given an excerpt from a merger agreement and the task is to answer: what counts as Knowledge, amongst the multiple choice options.
        27. MAUDLiabilityStandardForNoShopBreachByTargetNonDORepresentatives: Given an excerpt from a merger agreement and the task is to answer:  what is the liability standard for no-shop breach by Target Non-D&O Representatives, amongst the multiple choice options.
        28. MAUD Ordinary Course Efforts Standard: Given an excerpt from a merger agreement and the task is to answer: what is the efforts standard, amongst the multiple choice options.
        29. MAUD Pandemic Or Other Public Health Event Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do pandemics or other public health events have to have disproportionate impact to qualify for Material Adverse Effect (MAE), amongst the multiple choice options.
        30. MAUD Pandemic Or Other Public Health Event Specific Reference To Pandemic Related Governmental Responses Or Measures: Given an excerpt from a merger agreement and the task is to answer: is there specific reference to pandemic-related governmental responses or measures in the clause that qualifies pandemics or other public health events for Material Adverse Effect (MAE), amongst the multiple choice options.
        31. MAUD Relational Language MAE Applies To: Given an excerpt from a merger agreement and the task is to answer: what carveouts pertaining to Material Adverse Effect (MAE) does the relational language apply to?, amongst the multiple choice options.
        32. MAUD Specific Performance: Given an excerpt from a merger agreement and the task is to answer: what is the wording of the Specific Performance clause regarding the parties' entitlement in the event of a contractual breach, amongst the multiple choice options.
        33. MAUD Tail Period Length: Given an excerpt from a merger agreement and the task is to answer: how long is the Tail Period, amongst the multiple choice options.
        34. MAUD Type Of Consideration: Given an excerpt from a merger agreement and the task is to answer: what type of consideration is specified in this agreement, amongst the multiple choice options.
        

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### MAUDLegalBenchClassification.v2

This task was constructed from the MAUD dataset, which consists of over 47,000 labels across 152 merger agreements annotated to identify 92 questions in each agreement used by the 2021 American Bar Association (ABA) Public Target Deal Points Study. Each dataset is formatted as a series of multiple-choice questions, where given a segment of the merger agreement and a Deal Point question, the model is to choose the answer that best characterizes the agreement as response.

        This is a combination of all 34 of the MAUD Legal Bench datasets:
        1. MAUD Ability To Consummate Concept Is Subject To MAE Carveouts: Given an excerpt from a merger agreement and the task is to answer: is the “ability to consummate” concept subject to Material Adverse Effect (MAE) carveouts? amongst the multiple choice options.
        2. MAUD Accuracy Of Fundamental Target RWS Bringdown Standard: Given an excerpt from a merger agreement and the task is to answer: how accurate must the fundamental representations and warranties be according to the bring down provision, amongst the multiple choice options.
        3. MAUD Accuracy Of Target Capitalization RW Outstanding Shares Bringdown Standard Answer: Given an excerpt from a merger agreement and the task is to answer: how accurate must the fundamental representations and warranties be according to the bring down provision, amongst the multiple choice options.
        4. MAUD Accuracy Of Target General RW Bringdown Timing Answer: Given an excerpt from a merger agreement and the task is to answer: how accurate must the fundamental representations and warranties be according to the bring down provision, amongst the multiple choice options.
        5. MAUD Additional Matching Rights Period For Modifications Cor: Given an excerpt from a merger agreement and the task is to answer: how long is the additional matching rights period for modifications in case the board changes its recommendation, amongst the multiple choice options.
        6. MAUD Application Of Buyer Consent Requirement Negative Interim Covenant: Given an excerpt from a merger agreement and the task is to answer: what negative covenants does the requirement of Buyer consent apply to, amongst the multiple choice options.
        7. MAUD Buyer Consent Requirement Ordinary Course: Given an excerpt from a merger agreement and the task is to answer: in case the Buyer's consent for the acquired company's ordinary business operations is required, are there any limitations on the Buyer's right to condition, withhold, or delay their consent, amongst the multiple choice options.
        8. MAUD Change In Law Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do changes in law that have disproportionate impact qualify for Material Adverse Effect (MAE), amongst the multiple choice options.
        9. MAUD Changes In GAAP Or Other Accounting Principles Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do changes in GAAP or other accounting principles that have disproportionate impact qualify for Material Adverse Effect (MAE), amongst the multiple choice options.
        10. MAUD COR Permitted In Response To Intervening Event: Given an excerpt from a merger agreement and the task is to answer: is Change of Recommendation permitted in response to an intervening event, amongst the multiple choice options.
        11. MAUD COR Permitted With Board Fiduciary Determination Only: Given an excerpt from a merger agreement and the task is to answer: is Change of Recommendation permitted as long as the board determines that such change is required to fulfill its fiduciary obligations, amongst the multiple choice options.
        12. MAUD COR Standard Intervening Event: Given an excerpt from a merger agreement and the task is to answer: what standard should the board follow when determining whether to change its recommendation in response to an intervening event, amongst the multiple choice options.
        13. MAUD COR Standard Superior Offer: Given an excerpt from a merger agreement and the task is to answer: what standard should the board follow when determining whether to change its recommendation in connection with a superior offer, amongst the multiple choice options.
        14. MAUD Definition Contains Knowledge Requirement Answer: Given an excerpt from a merger agreement and the task is to answer: what is the knowledge requirement in the definition of “Intervening Event”, amongst the multiple choice options.
        15. MAUD Definition Includes Asset Deals: Given an excerpt from a merger agreement and the task is to answer: what qualifies as a superior offer in terms of asset deals, amongst the multiple choice options.
        16. MAUD Definition Includes Stock Deals: Given an excerpt from a merger agreement and the task is to answer: what qualifies as a superior offer in terms of stock deals, amongst the multiple choice options.
        17. MAUD Fiduciary Exception Board Determination Standard: Given an excerpt from a merger agreement and the task is to answer: under what circumstances could the Board take actions on a different acquisition proposal notwithstanding the no-shop provision, amongst the multiple choice options.
        18. MAUD Fiduciary Exception Board Determination Trigger No Shop: Given an excerpt from a merger agreement and the task is to answer: what type of offer could the Board take actions on notwithstanding the no-shop provision, amongst the multiple choice options.
        19. MAUD Financial Point Of View Is The Sole Consideration: Given an excerpt from a merger agreement and the task is to answer: is “financial point of view” the sole consideration when determining whether an offer is superior, amongst the multiple choice options.
        20. MAUD FLS MAE Standard: Given an excerpt from a merger agreement and the task is to answer: what is the Forward Looking Standard (FLS) with respect to Material Adverse Effect (MAE), amongst the multiple choice options.
        21. MAUD General Economic and Financial Conditions Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do changes caused by general economic and financial conditions that have disproportionate impact qualify for Material Adverse Effect (MAE), amongst the multiple choice options.
        22. MAUD Includes Consistent With Past Practice: Given an excerpt from a merger agreement and the task is to answer: does the wording of the Efforts Covenant clause include “consistent with past practice”, amongst the multiple choice options.
        23. MAUD Initial Matching Rights Period COR: Given an excerpt from a merger agreement and the task is to answer: how long is the initial matching rights period in case the board changes its recommendation, amongst the multiple choice options.
        24. MAUD Initial Matching Rights Period FTR: Given an excerpt from a merger agreement and the task is to answer: how long is the initial matching rights period in connection with the Fiduciary Termination Right (FTR), amongst the multiple choice options.
        25. MAUDInterveningEventRequiredToOccurAfterSigningAnswer: Given an excerpt from a merger agreement and the task is to answer: is an “Intervening Event” required to occur after signing, amongst the multiple choice options.
        26. MAUD Knowledge Definition: Given an excerpt from a merger agreement and the task is to answer: what counts as Knowledge, amongst the multiple choice options.
        27. MAUDLiabilityStandardForNoShopBreachByTargetNonDORepresentatives: Given an excerpt from a merger agreement and the task is to answer:  what is the liability standard for no-shop breach by Target Non-D&O Representatives, amongst the multiple choice options.
        28. MAUD Ordinary Course Efforts Standard: Given an excerpt from a merger agreement and the task is to answer: what is the efforts standard, amongst the multiple choice options.
        29. MAUD Pandemic Or Other Public Health Event Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do pandemics or other public health events have to have disproportionate impact to qualify for Material Adverse Effect (MAE), amongst the multiple choice options.
        30. MAUD Pandemic Or Other Public Health Event Specific Reference To Pandemic Related Governmental Responses Or Measures: Given an excerpt from a merger agreement and the task is to answer: is there specific reference to pandemic-related governmental responses or measures in the clause that qualifies pandemics or other public health events for Material Adverse Effect (MAE), amongst the multiple choice options.
        31. MAUD Relational Language MAE Applies To: Given an excerpt from a merger agreement and the task is to answer: what carveouts pertaining to Material Adverse Effect (MAE) does the relational language apply to?, amongst the multiple choice options.
        32. MAUD Specific Performance: Given an excerpt from a merger agreement and the task is to answer: what is the wording of the Specific Performance clause regarding the parties' entitlement in the event of a contractual breach, amongst the multiple choice options.
        33. MAUD Tail Period Length: Given an excerpt from a merger agreement and the task is to answer: how long is the Tail Period, amongst the multiple choice options.
        34. MAUD Type Of Consideration: Given an excerpt from a merger agreement and the task is to answer: what type of consideration is specified in this agreement, amongst the multiple choice options.

        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/maud_legal_bench`](https://huggingface.co/datasets/mteb/maud_legal_bench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### MTOPDomainClassification

MTOP: Multilingual Task-Oriented Semantic Parsing

**Dataset:** [`mteb/mtop_domain`](https://huggingface.co/datasets/mteb/mtop_domain) • **License:** not specified • [Learn more →](https://arxiv.org/pdf/2008.09335.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu, eng, fra, hin, spa, ... (6) | Spoken, Spoken | human-annotated | created |



#### MTOPDomainVNClassification

A translated dataset from MTOP: Multilingual Task-Oriented Semantic Parsing
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/mtop-domain-vn`](https://huggingface.co/datasets/GreenNode/mtop-domain-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2008.09335.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Spoken, Spoken | derived | machine-translated and LM verified |



#### MTOPIntentClassification

MTOP: Multilingual Task-Oriented Semantic Parsing

**Dataset:** [`mteb/mtop_intent`](https://huggingface.co/datasets/mteb/mtop_intent) • **License:** not specified • [Learn more →](https://arxiv.org/pdf/2008.09335.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu, eng, fra, hin, spa, ... (6) | Spoken, Spoken | human-annotated | created |



#### MTOPIntentVNClassification

A translated dataset from MTOP: Multilingual Task-Oriented Semantic Parsing
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/mtop-intent-vn`](https://huggingface.co/datasets/GreenNode/mtop-intent-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2008.09335.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Spoken, Spoken | derived | machine-translated and LM verified |



#### MacedonianTweetSentimentClassification

An Macedonian dataset for tweet sentiment classification.

**Dataset:** [`isaacchung/macedonian-tweet-sentiment-classification`](https://huggingface.co/datasets/isaacchung/macedonian-tweet-sentiment-classification) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://aclanthology.org/R15-1034/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | mkd | Social, Written | human-annotated | found |



#### MacedonianTweetSentimentClassification.v2

An Macedonian dataset for tweet sentiment classification.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/macedonian_tweet_sentiment`](https://huggingface.co/datasets/mteb/macedonian_tweet_sentiment) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://aclanthology.org/R15-1034/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | mkd | Social, Written | human-annotated | found |



#### MalayalamNewsClassification

A Malayalam dataset for 3-class classification of Malayalam news articles

**Dataset:** [`mlexplorer008/malayalam_news_classification`](https://huggingface.co/datasets/mlexplorer008/malayalam_news_classification) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-malyalam)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | mal | News, Written | derived | found |



#### MalayalamNewsClassification.v2

A Malayalam dataset for 3-class classification of Malayalam news articles
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/malayalam_news`](https://huggingface.co/datasets/mteb/malayalam_news) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-malyalam)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | mal | News, Written | derived | found |



#### MarathiNewsClassification

A Marathi dataset for 3-class classification of Marathi news articles

**Dataset:** [`mlexplorer008/marathi_news_classification`](https://huggingface.co/datasets/mlexplorer008/marathi_news_classification) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-marathi)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | mar | News, Written | derived | found |



#### MarathiNewsClassification.v2

A Marathi dataset for 3-class classification of Marathi news articles
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/marathi_news`](https://huggingface.co/datasets/mteb/marathi_news) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-marathi)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | mar | News, Written | derived | found |



#### MasakhaNEWSClassification

MasakhaNEWS is the largest publicly available dataset for news topic classification in 16 languages widely spoken in Africa. The train/validation/test sets are available for all the 16 languages.

**Dataset:** [`mteb/masakhanews`](https://huggingface.co/datasets/mteb/masakhanews) • **License:** cc-by-nc-4.0 • [Learn more →](https://arxiv.org/abs/2304.09972)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | amh, eng, fra, hau, ibo, ... (16) | News, Written | expert-annotated | found |



#### MassiveIntentClassification

MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages

**Dataset:** [`mteb/amazon_massive_intent`](https://huggingface.co/datasets/mteb/amazon_massive_intent) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2204.08582)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | afr, amh, ara, aze, ben, ... (50) | Spoken | human-annotated | human-translated and localized |



#### MassiveIntentVNClassification

A translated dataset from MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/amazon-massive-intent-vn`](https://huggingface.co/datasets/GreenNode/amazon-massive-intent-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2204.08582#:~:text=MASSIVE%20contains%201M%20realistic%2C%20parallel,diverse%20languages%20from%2029%20genera.)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Spoken | derived | machine-translated and LM verified |



#### MassiveScenarioClassification

MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages

**Dataset:** [`mteb/amazon_massive_scenario`](https://huggingface.co/datasets/mteb/amazon_massive_scenario) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2204.08582)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | afr, amh, ara, aze, ben, ... (50) | Spoken | human-annotated | human-translated and localized |



#### MassiveScenarioVNClassification

A translated dataset from MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/amazon-massive-scenario-vn`](https://huggingface.co/datasets/GreenNode/amazon-massive-scenario-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2204.08582#:~:text=MASSIVE%20contains%201M%20realistic%2C%20parallel,diverse%20languages%20from%2029%20genera.)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Spoken | derived | machine-translated and LM verified |



#### Moroco

The Moldavian and Romanian Dialectal Corpus. The MOROCO data set contains Moldavian and Romanian samples of text collected from the news domain. The samples belong to one of the following six topics: (0) culture, (1) finance, (2) politics, (3) science, (4) sports, (5) tech

**Dataset:** [`universityofbucharest/moroco`](https://huggingface.co/datasets/universityofbucharest/moroco) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/moroco)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ron | News, Written | derived | found |



#### Moroco.v2

The Moldavian and Romanian Dialectal Corpus. The MOROCO data set contains Moldavian and Romanian samples of text collected from the news domain. The samples belong to one of the following six topics: (0) culture, (1) finance, (2) politics, (3) science, (4) sports, (5) tech
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/moroco`](https://huggingface.co/datasets/mteb/moroco) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/moroco)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ron | News, Written | derived | found |



#### MovieReviewSentimentClassification

The Allociné dataset is a French-language dataset for sentiment analysis that contains movie reviews produced by the online community of the Allociné.fr website.

**Dataset:** [`tblard/allocine`](https://huggingface.co/datasets/tblard/allocine) • **License:** mit • [Learn more →](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fra | Reviews, Written | derived | found |



#### MovieReviewSentimentClassification.v2

The Allociné dataset is a French-language dataset for sentiment analysis that contains movie reviews produced by the online community of the Allociné.fr website.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/movie_review_sentiment`](https://huggingface.co/datasets/mteb/movie_review_sentiment) • **License:** mit • [Learn more →](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fra | Reviews, Written | derived | found |



#### MultiHateClassification

Hate speech detection dataset with binary
                       (hateful vs non-hateful) labels. Includes 25+ distinct types of hate
                       and challenging non-hate, and 11 languages.
                     

**Dataset:** [`mteb/multi-hatecheck`](https://huggingface.co/datasets/mteb/multi-hatecheck) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2022.woah-1.15/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara, cmn, deu, eng, fra, ... (11) | Constructed, Written | expert-annotated | created |



#### MultilingualSentiment

A collection of multilingual sentiments datasets grouped into 3 classes -- positive, neutral, negative

**Dataset:** [`C-MTEB/MultilingualSentiment-classification`](https://huggingface.co/datasets/C-MTEB/MultilingualSentiment-classification) • **License:** not specified • [Learn more →](https://github.com/tyqiangz/multilingual-sentiment-datasets)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### MultilingualSentiment.v2

A collection of multilingual sentiments datasets grouped into 3 classes -- positive, neutral, negative
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/multilingual_sentiment`](https://huggingface.co/datasets/mteb/multilingual_sentiment) • **License:** not specified • [Learn more →](https://github.com/tyqiangz/multilingual-sentiment-datasets)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### MultilingualSentimentClassification

Sentiment classification dataset with binary
                       (positive vs negative sentiment) labels. Includes 30 languages and dialects.
                     

**Dataset:** [`mteb/multilingual-sentiment-classification`](https://huggingface.co/datasets/mteb/multilingual-sentiment-classification) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb/multilingual-sentiment-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara, bam, bul, cmn, cym, ... (29) | Reviews, Written | derived | found |



#### MyanmarNews

The Myanmar News dataset on Hugging Face contains news articles in Burmese. It is designed for tasks such as text classification, sentiment analysis, and language modeling. The dataset includes a variety of news topics in 4 categorie, providing a rich resource for natural language processing applications involving Burmese which is a low resource language.

**Dataset:** [`mteb/MyanmarNews`](https://huggingface.co/datasets/mteb/MyanmarNews) • **License:** gpl-3.0 • [Learn more →](https://huggingface.co/datasets/myanmar_news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | mya | News, Written | derived | found |



#### MyanmarNews.v2

The Myanmar News dataset on Hugging Face contains news articles in Burmese. It is designed for tasks such as text classification, sentiment analysis, and language modeling. The dataset includes a variety of news topics in 4 categorie, providing a rich resource for natural language processing applications involving Burmese which is a low resource language.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/myanmar_news`](https://huggingface.co/datasets/mteb/myanmar_news) • **License:** gpl-3.0 • [Learn more →](https://huggingface.co/datasets/myanmar_news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | mya | News, Written | derived | found |



#### NLPTwitterAnalysisClassification

Twitter Analysis Classification

**Dataset:** [`hamedhf/nlp_twitter_analysis`](https://huggingface.co/datasets/hamedhf/nlp_twitter_analysis) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/hamedhf/nlp_twitter_analysis/tree/main)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Social | derived | found |



#### NLPTwitterAnalysisClassification.v2

Twitter Analysis Classification
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/nlp_twitter_analysis`](https://huggingface.co/datasets/mteb/nlp_twitter_analysis) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/hamedhf/nlp_twitter_analysis/tree/main)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Social | derived | found |



#### NYSJudicialEthicsLegalBenchClassification

Answer questions on judicial ethics from the New York State Unified Court System Advisory Committee.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** mit • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### NaijaSenti

NaijaSenti is the first large-scale human-annotated Twitter sentiment dataset for the four most widely spoken languages in Nigeria — Hausa, Igbo, Nigerian-Pidgin, and Yorùbá — consisting of around 30,000 annotated tweets per language, including a significant fraction of code-mixed tweets.

**Dataset:** [`HausaNLP/NaijaSenti-Twitter`](https://huggingface.co/datasets/HausaNLP/NaijaSenti-Twitter) • **License:** cc-by-4.0 • [Learn more →](https://github.com/hausanlp/NaijaSenti)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | hau, ibo, pcm, yor | Social, Written | expert-annotated | found |



#### NepaliNewsClassification

A Nepali dataset for 7500 news articles 

**Dataset:** [`bpHigh/iNLTK_Nepali_News_Dataset`](https://huggingface.co/datasets/bpHigh/iNLTK_Nepali_News_Dataset) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/goru001/nlp-for-nepali)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nep | News, Written | derived | found |



#### NepaliNewsClassification.v2

A Nepali dataset for 7500 news articles
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/nepali_news`](https://huggingface.co/datasets/mteb/nepali_news) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/goru001/nlp-for-nepali)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nep | News, Written | derived | found |



#### NewsClassification

Large News Classification Dataset

**Dataset:** [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/1509.01626)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | News, Written | expert-annotated | found |



#### NewsClassification.v2

Large News Classification Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/news`](https://huggingface.co/datasets/mteb/news) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/1509.01626)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | News, Written | expert-annotated | found |



#### NoRecClassification

A Norwegian dataset for sentiment classification on review

**Dataset:** [`mteb/norec_classification`](https://huggingface.co/datasets/mteb/norec_classification) • **License:** cc-by-nc-4.0 • [Learn more →](https://aclanthology.org/L18-1661/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nob | Reviews, Written | derived | found |



#### NoRecClassification.v2

A Norwegian dataset for sentiment classification on review
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/no_rec`](https://huggingface.co/datasets/mteb/no_rec) • **License:** cc-by-nc-4.0 • [Learn more →](https://aclanthology.org/L18-1661/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nob | Reviews, Written | derived | found |



#### NordicLangClassification

A dataset for Nordic language identification.

**Dataset:** [`strombergnlp/nordic_langid`](https://huggingface.co/datasets/strombergnlp/nordic_langid) • **License:** cc-by-sa-3.0 • [Learn more →](https://aclanthology.org/2021.vardial-1.8/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan, fao, isl, nno, nob, ... (6) | Encyclopaedic | derived | found |



#### NorwegianParliamentClassification

Norwegian parliament speeches annotated for sentiment

**Dataset:** [`NbAiLab/norwegian_parliament`](https://huggingface.co/datasets/NbAiLab/norwegian_parliament) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/NbAiLab/norwegian_parliament)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nob | Government, Spoken | derived | found |



#### NorwegianParliamentClassification.v2

Norwegian parliament speeches annotated for sentiment
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/norwegian_parliament`](https://huggingface.co/datasets/mteb/norwegian_parliament) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/NbAiLab/norwegian_parliament)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nob | Government, Spoken | derived | found |



#### NusaParagraphEmotionClassification

NusaParagraphEmotionClassification is a multi-class emotion classification on 10 Indonesian languages from the NusaParagraph dataset.

**Dataset:** [`gentaiscool/nusaparagraph_emot`](https://huggingface.co/datasets/gentaiscool/nusaparagraph_emot) • **License:** apache-2.0 • [Learn more →](https://github.com/IndoNLP/nusa-writes)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | bbc, bew, bug, jav, mad, ... (10) | Fiction, Non-fiction, Written | human-annotated | found |



#### NusaParagraphTopicClassification

NusaParagraphTopicClassification is a multi-class topic classification on 10 Indonesian languages.

**Dataset:** [`gentaiscool/nusaparagraph_topic`](https://huggingface.co/datasets/gentaiscool/nusaparagraph_topic) • **License:** apache-2.0 • [Learn more →](https://github.com/IndoNLP/nusa-writes)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | bbc, bew, bug, jav, mad, ... (10) | Fiction, Non-fiction, Written | human-annotated | found |



#### NusaX-senti

NusaX is a high-quality multilingual parallel corpus that covers 12 languages, Indonesian, English, and 10 Indonesian local languages, namely Acehnese, Balinese, Banjarese, Buginese, Madurese, Minangkabau, Javanese, Ngaju, Sundanese, and Toba Batak. NusaX-Senti is a 3-labels (positive, neutral, negative) sentiment analysis dataset for 10 Indonesian local languages + Indonesian and English.

**Dataset:** [`indonlp/NusaX-senti`](https://huggingface.co/datasets/indonlp/NusaX-senti) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2205.15960)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ace, ban, bbc, bjn, bug, ... (12) | Constructed, Reviews, Social, Web, Written | expert-annotated | found |



#### OPP115DataRetentionLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describes how long user information is stored.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115DataSecurityLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describes how user information is protected.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115DataSecurityLegalBenchClassification.v2

Given a clause from a privacy policy, classify if the clause describes how user information is protected.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/opp115_data_security_legal_bench`](https://huggingface.co/datasets/mteb/opp115_data_security_legal_bench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115DoNotTrackLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describes if and how Do Not Track signals for online tracking and advertising are honored.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115DoNotTrackLegalBenchClassification.v2

Given a clause from a privacy policy, classify if the clause describes if and how Do Not Track signals for online tracking and advertising are honored.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/opp115_do_not_track_legal_bench`](https://huggingface.co/datasets/mteb/opp115_do_not_track_legal_bench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115FirstPartyCollectionUseLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describes how and why a service provider collects user information.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115InternationalAndSpecificAudiencesLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describe practices that pertain only to a specific group of users (e.g., children, Europeans, or California residents).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115PolicyChangeLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describes if and how users will be informed about changes to the privacy policy.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115ThirdPartySharingCollectionLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describe how user information may be shared with or collected by third parties.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115UserAccessEditAndDeletionLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describes if and how users may access, edit, or delete their information.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115UserChoiceControlLegalBenchClassification

Given a clause fro ma privacy policy, classify if the clause describes the choices and control options available to users.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OPP115UserChoiceControlLegalBenchClassification.v2

Given a clause fro ma privacy policy, classify if the clause describes the choices and control options available to users.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/opp115_user_choice_control_legal_bench`](https://huggingface.co/datasets/mteb/opp115_user_choice_control_legal_bench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OdiaNewsClassification

A Odia dataset for 3-class classification of Odia news articles

**Dataset:** [`mlexplorer008/odia_news_classification`](https://huggingface.co/datasets/mlexplorer008/odia_news_classification) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-odia)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ory | News, Written | derived | found |



#### OdiaNewsClassification.v2

A Odia dataset for 3-class classification of Odia news articles
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/odia_news`](https://huggingface.co/datasets/mteb/odia_news) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-odia)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ory | News, Written | derived | found |



#### OnlineShopping

Sentiment Analysis of User Reviews on Online Shopping Websites

**Dataset:** [`C-MTEB/OnlineShopping-classification`](https://huggingface.co/datasets/C-MTEB/OnlineShopping-classification) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### OnlineStoreReviewSentimentClassification

This dataset contains Arabic reviews of products from the SHEIN online store.

**Dataset:** [`Ruqiya/Arabic_Reviews_of_SHEIN`](https://huggingface.co/datasets/Ruqiya/Arabic_Reviews_of_SHEIN) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/Ruqiya/Arabic_Reviews_of_SHEIN)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Reviews, Written | derived | found |



#### OnlineStoreReviewSentimentClassification.v2

This dataset contains Arabic reviews of products from the SHEIN online store.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/online_store_review_sentiment`](https://huggingface.co/datasets/mteb/online_store_review_sentiment) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/Ruqiya/Arabic_Reviews_of_SHEIN)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Reviews, Written | derived | found |



#### OralArgumentQuestionPurposeLegalBenchClassification

This task classifies questions asked by Supreme Court justices at oral argument into seven categories:
        1. Background - questions seeking factual or procedural information that is missing or not clear in the briefing
        2. Clarification - questions seeking to get an advocate to clarify her position or the scope of the rule being advocated for
        3. Implications - questions about the limits of a rule or its implications for future cases
        4. Support - questions offering support for the advocate’s position
        5. Criticism - questions criticizing an advocate’s position
        6. Communicate - question designed primarily to communicate with other justices
        7. Humor - questions designed to interject humor into the argument and relieve tension
        

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OralArgumentQuestionPurposeLegalBenchClassification.v2

This task classifies questions asked by Supreme Court justices at oral argument into seven categories:
        1. Background - questions seeking factual or procedural information that is missing or not clear in the briefing
        2. Clarification - questions seeking to get an advocate to clarify her position or the scope of the rule being advocated for
        3. Implications - questions about the limits of a rule or its implications for future cases
        4. Support - questions offering support for the advocate’s position
        5. Criticism - questions criticizing an advocate’s position
        6. Communicate - question designed primarily to communicate with other justices
        7. Humor - questions designed to interject humor into the argument and relieve tension

        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/oral_argument_question_purpose_legal_bench`](https://huggingface.co/datasets/mteb/oral_argument_question_purpose_legal_bench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OverrulingLegalBenchClassification

This task consists of classifying whether or not a particular sentence of case law overturns the decision of a previous case.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### OverrulingLegalBenchClassification.v2

This task consists of classifying whether or not a particular sentence of case law overturns the decision of a previous case.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/overruling_legal_bench`](https://huggingface.co/datasets/mteb/overruling_legal_bench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### PAC

Polish Paraphrase Corpus

**Dataset:** [`laugustyniak/abusive-clauses-pl`](https://huggingface.co/datasets/laugustyniak/abusive-clauses-pl) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2211.13112.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Legal, Written | not specified | not specified |



#### PAC.v2

Polish Paraphrase Corpus
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/pac`](https://huggingface.co/datasets/mteb/pac) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2211.13112.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Legal, Written | not specified | not specified |



#### PROALegalBenchClassification

Given a statute, determine if the text contains an explicit private right of action. Given a privacy policy clause and a description of the clause, determine if the description is correct. A private right of action (PROA) exists when a statute empowers an ordinary individual (i.e., a private person) to legally enforce their rights by bringing an action in court. In short, a PROA creates the ability for an individual to sue someone in order to recover damages or halt some offending conduct. PROAs are ubiquitous in antitrust law (in which individuals harmed by anti-competitive behavior can sue offending firms for compensation) and environmental law (in which individuals can sue entities which release hazardous substances for damages).

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### PatentClassification

Classification Dataset of Patents and Abstract

**Dataset:** [`mteb/PatentClassification`](https://huggingface.co/datasets/mteb/PatentClassification) • **License:** not specified • [Learn more →](https://aclanthology.org/P19-1212.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | derived | found |



#### PatentClassification.v2

Classification Dataset of Patents and Abstract
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/patent`](https://huggingface.co/datasets/mteb/patent) • **License:** not specified • [Learn more →](https://aclanthology.org/P19-1212.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | derived | found |



#### PersianFoodSentimentClassification

Persian Food Review Dataset

**Dataset:** [`asparius/Persian-Food-Sentiment`](https://huggingface.co/datasets/asparius/Persian-Food-Sentiment) • **License:** not specified • [Learn more →](https://hooshvare.github.io/docs/datasets/sa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Reviews, Written | derived | found |



#### PersianTextEmotion

Emotion is a Persian dataset with six basic emotions: anger, fear, joy, love, sadness, and surprise.

**Dataset:** [`SeyedAli/Persian-Text-Emotion`](https://huggingface.co/datasets/SeyedAli/Persian-Text-Emotion) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/SeyedAli/Persian-Text-Emotion)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | not specified | derived | found |



#### PersianTextEmotion.v2

Emotion is a Persian dataset with six basic emotions: anger, fear, joy, love, sadness, and surprise.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/persian_text_emotion`](https://huggingface.co/datasets/mteb/persian_text_emotion) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/SeyedAli/Persian-Text-Emotion)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | not specified | derived | found |



#### PersonalJurisdictionLegalBenchClassification

Given a fact pattern describing the set of contacts between a plaintiff, defendant, and forum, determine if a court in that forum could excercise personal jurisdiction over the defendant.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### PoemSentimentClassification

Poem Sentiment is a sentiment dataset of poem verses from Project Gutenberg.

**Dataset:** [`google-research-datasets/poem_sentiment`](https://huggingface.co/datasets/google-research-datasets/poem_sentiment) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2011.02686)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | human-annotated | found |



#### PoemSentimentClassification.v2

Poem Sentiment is a sentiment dataset of poem verses from Project Gutenberg.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/poem_sentiment`](https://huggingface.co/datasets/mteb/poem_sentiment) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2011.02686)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | human-annotated | found |



#### PolEmo2.0-IN

A collection of Polish online reviews from four domains: medicine, hotels, products and school. The PolEmo2.0-IN task is to predict the sentiment of in-domain (medicine and hotels) reviews.

**Dataset:** [`PL-MTEB/polemo2_in`](https://huggingface.co/datasets/PL-MTEB/polemo2_in) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/K19-1092.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Social, Written | derived | found |



#### PolEmo2.0-IN.v2

A collection of Polish online reviews from four domains: medicine, hotels, products and school. The PolEmo2.0-IN task is to predict the sentiment of in-domain (medicine and hotels) reviews.

**Dataset:** [`mteb/pol_emo2_in`](https://huggingface.co/datasets/mteb/pol_emo2_in) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/K19-1092.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Social, Written | derived | found |



#### PolEmo2.0-OUT

A collection of Polish online reviews from four domains: medicine, hotels, products and school. The PolEmo2.0-OUT task is to predict the sentiment of out-of-domain (products and school) reviews using models train on reviews from medicine and hotels domains.

**Dataset:** [`PL-MTEB/polemo2_out`](https://huggingface.co/datasets/PL-MTEB/polemo2_out) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/K19-1092.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Social, Written | not specified | not specified |



#### PolEmo2.0-OUT.v2

A collection of Polish online reviews from four domains: medicine, hotels, products and school. The PolEmo2.0-OUT task is to predict the sentiment of out-of-domain (products and school) reviews using models train on reviews from medicine and hotels domains.

**Dataset:** [`mteb/pol_emo2_out`](https://huggingface.co/datasets/mteb/pol_emo2_out) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/K19-1092.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Social, Written | not specified | not specified |



#### PunjabiNewsClassification

A Punjabi dataset for 2-class classification of Punjabi news articles

**Dataset:** [`mlexplorer008/punjabi_news_classification`](https://huggingface.co/datasets/mlexplorer008/punjabi_news_classification) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-punjabi/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pan | News, Written | derived | found |



#### RestaurantReviewSentimentClassification

Dataset of 8364 restaurant reviews from qaym.com in Arabic for sentiment analysis

**Dataset:** [`hadyelsahar/ar_res_reviews`](https://huggingface.co/datasets/hadyelsahar/ar_res_reviews) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-18117-2_2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Reviews, Written | derived | found |



#### RestaurantReviewSentimentClassification.v2

Dataset of 8156 restaurant reviews from qaym.com in Arabic for sentiment analysis
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/restaurant_review_sentiment`](https://huggingface.co/datasets/mteb/restaurant_review_sentiment) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-18117-2_2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Reviews, Written | derived | found |



#### RomanianReviewsSentiment

LaRoSeDa (A Large Romanian Sentiment Data Set) contains 15,000 reviews written in Romanian

**Dataset:** [`universityofbucharest/laroseda`](https://huggingface.co/datasets/universityofbucharest/laroseda) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2101.04197)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ron | Reviews, Written | derived | found |



#### RomanianReviewsSentiment.v2

LaRoSeDa (A Large Romanian Sentiment Data Set) contains 15,000 reviews written in Romanian
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/romanian_reviews_sentiment`](https://huggingface.co/datasets/mteb/romanian_reviews_sentiment) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2101.04197)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ron | Reviews, Written | derived | found |



#### RomanianSentimentClassification

An Romanian dataset for sentiment classification.

**Dataset:** [`dumitrescustefan/ro_sent`](https://huggingface.co/datasets/dumitrescustefan/ro_sent) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2009.08712)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ron | Reviews, Written | human-annotated | found |



#### RomanianSentimentClassification.v2

An Romanian dataset for sentiment classification.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/romanian_sentiment`](https://huggingface.co/datasets/mteb/romanian_sentiment) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2009.08712)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ron | Reviews, Written | human-annotated | found |



#### RuNLUIntentClassification

Contains natural language data for human-robot interaction in home domain which we collected and annotated for evaluating NLU Services/platforms.

**Dataset:** [`mteb/RuNLUIntentClassification`](https://huggingface.co/datasets/mteb/RuNLUIntentClassification) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/1903.05566)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | rus | not specified | human-annotated | found |



#### RuReviewsClassification

Product review classification (3-point scale) based on RuRevies dataset

**Dataset:** [`ai-forever/ru-reviews-classification`](https://huggingface.co/datasets/ai-forever/ru-reviews-classification) • **License:** apache-2.0 • [Learn more →](https://github.com/sismetanin/rureviews)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Reviews, Written | derived | found |



#### RuReviewsClassification.v2

Product review classification (3-point scale) based on RuRevies dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/ru_reviews`](https://huggingface.co/datasets/mteb/ru_reviews) • **License:** apache-2.0 • [Learn more →](https://github.com/sismetanin/rureviews)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Reviews, Written | derived | found |



#### RuSciBenchCoreRiscClassification

This binary classification task aims to determine whether a scientific paper
        (based on its title and abstract) belongs to the Core of the Russian Science Citation Index (RISC).
        The RISC includes a wide range of publications, but the Core RISC comprises the most cited and prestigious
        journals, dissertations, theses, monographs, and studies. The task is provided for both Russian and English
        versions of the paper's title and abstract.

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_mteb`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_mteb) • **License:** mit • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng, rus | Academic, Non-fiction, Written | derived | found |



#### RuSciBenchGRNTIClassification

Classification of scientific papers (title+abstract) by rubric

**Dataset:** [`ai-forever/ru-scibench-grnti-classification`](https://huggingface.co/datasets/ai-forever/ru-scibench-grnti-classification) • **License:** not specified • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Academic, Written | derived | found |



#### RuSciBenchGRNTIClassification.v2

Classification of scientific papers based on the GRNTI (State Rubricator of Scientific and
        Technical Information) rubricator. GRNTI is a universal hierarchical classification of knowledge domains
        adopted in Russia and CIS countries to systematize the entire flow of scientific and technical information.
        This task uses the first level of the GRNTI hierarchy and top 28 classes by frequency.

        In this version, English language support has been added and data partitioning has been slightly modified.
        

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_mteb`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_mteb) • **License:** mit • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng, rus | Academic, Non-fiction, Written | derived | found |



#### RuSciBenchOECDClassification

Classification of scientific papers (title+abstract) by rubric

**Dataset:** [`ai-forever/ru-scibench-oecd-classification`](https://huggingface.co/datasets/ai-forever/ru-scibench-oecd-classification) • **License:** not specified • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Academic, Written | derived | found |



#### RuSciBenchOECDClassification.v2

Classification of scientific papers based on the OECD
        (Organization for Economic Co-operation and Development) rubricator. OECD provides
        a hierarchical 3-level system of classes for labeling scientific articles.
        This task uses the first two levels of the OECD hierarchy, top 29 classes.

        In this version, English language support has been added and data partitioning has been slightly modified.
        

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_mteb`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_mteb) • **License:** mit • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng, rus | Academic, Non-fiction, Written | derived | found |



#### RuSciBenchPubTypeClassification

This task involves classifying scientific papers (based on their title and abstract)
        into different publication types. The dataset identifies the following types:
        'Article', 'Conference proceedings', 'Survey', 'Miscellanea', 'Short message', 'Review', and 'Personalia'.
        This task is available for both Russian and English versions of the paper's title and abstract.

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_mteb`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_mteb) • **License:** mit • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng, rus | Academic, Non-fiction, Written | derived | found |



#### RuToxicOKMLCUPClassification

On the Odnoklassniki social network, users post a huge number of comments of various directions and nature every day.

**Dataset:** [`mteb/RuToxicOKMLCUPClassification`](https://huggingface.co/datasets/mteb/RuToxicOKMLCUPClassification) • **License:** not specified • [Learn more →](https://cups.online/ru/contests/okmlcup2020)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | rus | not specified | derived | found |



#### RuToxicOKMLCUPClassification.v2

On the Odnoklassniki social network, users post a huge number of comments of various directions and nature every day.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/ru_toxic_okmlcup`](https://huggingface.co/datasets/mteb/ru_toxic_okmlcup) • **License:** not specified • [Learn more →](https://cups.online/ru/contests/okmlcup2020)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | rus | not specified | derived | found |



#### RuToxicOKMLCUPMultilabelClassification

On the Odnoklassniki social network, users post a huge number of comments of various directions and nature every day.

**Dataset:** [`mteb/RuToxicOKMLCUPClassification`](https://huggingface.co/datasets/mteb/RuToxicOKMLCUPClassification) • **License:** not specified • [Learn more →](https://cups.online/ru/contests/okmlcup2020)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | rus | not specified | derived | found |



#### SCDBPAccountabilityLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer maintains internal compliance procedures on company standards regarding human trafficking and slavery? This includes any type of internal accountability mechanism. Requiring independently of the supply to comply with laws does not qualify or asking for documentary evidence of compliance does not count either.'

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### SCDBPAuditsLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer  performs any type of audit, or reserves the right to audit?'

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### SCDBPCertificationLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer  performs any type of audit, or reserves the right to audit?'

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### SCDBPTrainingLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer  provides training to employees on human trafficking and slavery? Broad policies such as ongoing dialogue on mitigating risks of human trafficking and slavery or increasing managers and purchasers knowledge about health, safety and labor practices qualify as training. Providing training to contractors who failed to comply with human trafficking laws counts as training.'

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### SCDBPVerificationLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer engages in verification and auditing as one practice, expresses that it may conduct an audit, or expressess that it is assessing supplier risks through a review of the US Dept. of Labor's List?'

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### SCDDAccountabilityLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose to what extent, if any, that the retail seller or manufacturer maintains internal accountability standards and procedures for employees or contractors failing to meet company standards regarding slavery and trafficking?'

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### SCDDAuditsLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose to what extent, if any, that the retail seller or manufacturer conducts audits of suppliers to evaluate supplier compliance with company standards for trafficking and slavery in supply chains? The disclosure shall specify if the verification was not an independent, unannounced audit.'

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### SCDDCertificationLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose to what extent, if any, that the retail seller or manufacturer requires direct suppliers to certify that materials incorporated into the product comply with the laws regarding slavery and human trafficking of the country or countries in which they are doing business?'

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### SCDDTrainingLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose to what extent, if any, that the retail seller or manufacturer provides company employees and management, who have direct responsibility for supply chain management, training on human trafficking and slavery, particularly with respect to mitigating risks within the supply chains of products?'

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### SCDDVerificationLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose to what extent, if any, that the retail seller or manufacturer engages in verification of product supply chains to evaluate and address risks of human trafficking and slavery? If the company conducts verification], the disclosure shall specify if the verification was not conducted by a third party.'

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### SDSEyeProtectionClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/SDSEyeProtectionClassification`](https://huggingface.co/datasets/BASF-AI/SDSEyeProtectionClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | LM-generated and reviewed | created |



#### SDSEyeProtectionClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sds_eye_protection`](https://huggingface.co/datasets/mteb/sds_eye_protection) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | LM-generated and reviewed | created |



#### SDSGlovesClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/SDSGlovesClassification`](https://huggingface.co/datasets/BASF-AI/SDSGlovesClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | LM-generated and reviewed | created |



#### SDSGlovesClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sds_gloves`](https://huggingface.co/datasets/mteb/sds_gloves) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | LM-generated and reviewed | created |



#### SIB200Classification

SIB-200 is the largest publicly available topic classification
        dataset based on Flores-200 covering 205 languages and dialects annotated. The dataset is
        annotated in English for the topics,  science/technology, travel, politics, sports,
        health, entertainment, and geography. The labels are then transferred to the other languages
        in Flores-200 which are human-translated.
        

**Dataset:** [`mteb/sib200`](https://huggingface.co/datasets/mteb/sib200) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2309.07445)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ace, acm, acq, aeb, afr, ... (197) | News, Written | expert-annotated | human-translated and localized |



#### SIDClassification

SID Classification

**Dataset:** [`MCINext/sid-classification`](https://huggingface.co/datasets/MCINext/sid-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Academic | derived | found |



#### SIDClassification.v2

SID Classification
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sid`](https://huggingface.co/datasets/mteb/sid) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Academic | derived | found |



#### SanskritShlokasClassification

This data set contains ~500 Shlokas  

**Dataset:** [`bpHigh/iNLTK_Sanskrit_Shlokas_Dataset`](https://huggingface.co/datasets/bpHigh/iNLTK_Sanskrit_Shlokas_Dataset) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/goru001/nlp-for-sanskrit)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | san | Religious, Written | derived | found |



#### SardiStanceClassification

SardiStance is a unique dataset designed for the task of stance detection in Italian tweets. It consists of tweets related to the Sardines movement, providing a valuable resource for researchers and practitioners in the field of NLP.

**Dataset:** [`MattiaSangermano/SardiStance`](https://huggingface.co/datasets/MattiaSangermano/SardiStance) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://github.com/mirkolai/evalita-sardistance)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ita | Social | derived | found |



#### ScalaClassification

ScaLa a linguistic acceptability dataset for the mainland Scandinavian languages automatically constructed from dependency annotations in Universal Dependencies Treebanks.
        Published as part of 'ScandEval: A Benchmark for Scandinavian Natural Language Processing'

**Dataset:** [`mteb/multilingual-scala-classification`](https://huggingface.co/datasets/mteb/multilingual-scala-classification) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan, nno, nob, swe | Blog, Fiction, News, Non-fiction, Spoken, ... (7) | human-annotated | created |



#### ScandiSentClassification

The corpus is crawled from se.trustpilot.com, no.trustpilot.com, dk.trustpilot.com, fi.trustpilot.com and trustpilot.com.

**Dataset:** [`mteb/scandisent`](https://huggingface.co/datasets/mteb/scandisent) • **License:** openrail • [Learn more →](https://github.com/timpal0l/ScandiSent)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan, eng, fin, nob, swe | Reviews, Written | expert-annotated | found |



#### SentiRuEval2016

Russian sentiment analysis evaluation SentiRuEval-2016 devoted to reputation monitoring of banks and telecom companies in Twitter. We describe the task, data, the procedure of data preparation, and participants’ results.

**Dataset:** [`mteb/SentiRuEval2016`](https://huggingface.co/datasets/mteb/SentiRuEval2016) • **License:** not specified • [Learn more →](https://github.com/mokoron/sentirueval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | rus | not specified | derived | found |



#### SentiRuEval2016.v2

Russian sentiment analysis evaluation SentiRuEval-2016 devoted to reputation monitoring of banks and telecom companies in Twitter. We describe the task, data, the procedure of data preparation, and participants’ results.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/senti_ru_eval2016`](https://huggingface.co/datasets/mteb/senti_ru_eval2016) • **License:** not specified • [Learn more →](https://github.com/mokoron/sentirueval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | rus | not specified | derived | found |



#### SentimentAnalysisHindi

Hindi Sentiment Analysis Dataset

**Dataset:** [`OdiaGenAI/sentiment_analysis_hindi`](https://huggingface.co/datasets/OdiaGenAI/sentiment_analysis_hindi) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/OdiaGenAI/sentiment_analysis_hindi)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | hin | Reviews, Written | derived | found |



#### SentimentAnalysisHindi.v2

Hindi Sentiment Analysis Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sentiment_analysis_hindi`](https://huggingface.co/datasets/mteb/sentiment_analysis_hindi) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/OdiaGenAI/sentiment_analysis_hindi)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | hin | Reviews, Written | derived | found |



#### SentimentDKSF

The Sentiment DKSF (Digikala/Snappfood comments) is a dataset for sentiment analysis.

**Dataset:** [`hezarai/sentiment-dksf`](https://huggingface.co/datasets/hezarai/sentiment-dksf) • **License:** not specified • [Learn more →](https://github.com/hezarai/hezar)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Reviews | derived | found |



#### SentimentDKSF.v2

The Sentiment DKSF (Digikala/Snappfood comments) is a dataset for sentiment analysis.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sentiment_dksf`](https://huggingface.co/datasets/mteb/sentiment_dksf) • **License:** not specified • [Learn more →](https://github.com/hezarai/hezar)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Reviews | derived | found |



#### SinhalaNewsClassification

This file contains news texts (sentences) belonging to 5 different news categories (political, business, technology, sports and Entertainment). The original dataset was released by Nisansa de Silva (Sinhala Text Classification: Observations from the Perspective of a Resource Poor Language, 2015).

**Dataset:** [`NLPC-UOM/Sinhala-News-Category-classification`](https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Category-classification) • **License:** mit • [Learn more →](https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Category-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | sin | News, Written | derived | found |



#### SinhalaNewsClassification.v2

This file contains news texts (sentences) belonging to 5 different news categories (political, business, technology, sports and Entertainment). The original dataset was released by Nisansa de Silva (Sinhala Text Classification: Observations from the Perspective of a Resource Poor Language, 2015).
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sinhala_news`](https://huggingface.co/datasets/mteb/sinhala_news) • **License:** mit • [Learn more →](https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Category-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | sin | News, Written | derived | found |



#### SinhalaNewsSourceClassification

This dataset contains Sinhala news headlines extracted from 9 news sources (websites) (Sri Lanka Army, Dinamina, GossipLanka, Hiru, ITN, Lankapuwath, NewsLK, Newsfirst, World Socialist Web Site-Sinhala).

**Dataset:** [`NLPC-UOM/Sinhala-News-Source-classification`](https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Source-classification) • **License:** mit • [Learn more →](https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Source-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | sin | News, Written | derived | found |



#### SinhalaNewsSourceClassification.v2

This dataset contains Sinhala news headlines extracted from 9 news sources (websites) (Sri Lanka Army, Dinamina, GossipLanka, Hiru, ITN, Lankapuwath, NewsLK, Newsfirst, World Socialist Web Site-Sinhala).
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sinhala_news_source`](https://huggingface.co/datasets/mteb/sinhala_news_source) • **License:** mit • [Learn more →](https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Source-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | sin | News, Written | derived | found |



#### SiswatiNewsClassification

Siswati News Classification Dataset

**Dataset:** [`isaacchung/siswati-news`](https://huggingface.co/datasets/isaacchung/siswati-news) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/dsfsi/za-isizulu-siswati-news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ssw | News, Written | human-annotated | found |



#### SiswatiNewsClassification.v2

Siswati News Classification Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/siswati_news`](https://huggingface.co/datasets/mteb/siswati_news) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/dsfsi/za-isizulu-siswati-news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ssw | News, Written | human-annotated | found |



#### SlovakHateSpeechClassification

The dataset contains posts from a social network with human annotations for hateful or offensive language in Slovak.

**Dataset:** [`TUKE-KEMT/hate_speech_slovak`](https://huggingface.co/datasets/TUKE-KEMT/hate_speech_slovak) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/TUKE-KEMT/hate_speech_slovak)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | slk | Social, Written | human-annotated | found |



#### SlovakHateSpeechClassification.v2

The dataset contains posts from a social network with human annotations for hateful or offensive language in Slovak.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/slovak_hate_speech`](https://huggingface.co/datasets/mteb/slovak_hate_speech) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/TUKE-KEMT/hate_speech_slovak)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | slk | Social, Written | human-annotated | found |



#### SlovakMovieReviewSentimentClassification

User reviews of movies on the CSFD movie database, with 2 sentiment classes (positive, negative)

**Dataset:** [`janko/sk_csfd-movie-reviews`](https://huggingface.co/datasets/janko/sk_csfd-movie-reviews) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2304.01922)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | svk | Reviews, Written | derived | found |



#### SlovakMovieReviewSentimentClassification.v2

User reviews of movies on the CSFD movie database, with 2 sentiment classes (positive, negative)
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/slovak_movie_review_sentiment`](https://huggingface.co/datasets/mteb/slovak_movie_review_sentiment) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2304.01922)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | svk | Reviews, Written | derived | found |



#### SouthAfricanLangClassification

A language identification test set for 11 South African Languages.

**Dataset:** [`mlexplorer008/south_african_language_identification`](https://huggingface.co/datasets/mlexplorer008/south_african_language_identification) • **License:** mit • [Learn more →](https://www.kaggle.com/competitions/south-african-language-identification/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | afr, eng, nbl, nso, sot, ... (11) | Non-fiction, Web, Written | expert-annotated | found |



#### SpanishNewsClassification

A Spanish dataset for news classification. The dataset includes articles from reputable Spanish news sources spanning 12 different categories.

**Dataset:** [`MarcOrfilaCarreras/spanish-news`](https://huggingface.co/datasets/MarcOrfilaCarreras/spanish-news) • **License:** mit • [Learn more →](https://huggingface.co/datasets/MarcOrfilaCarreras/spanish-news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | spa | News, Written | derived | found |



#### SpanishNewsClassification.v2

A Spanish dataset for news classification. The dataset includes articles from reputable Spanish news sources spanning 12 different categories.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/spanish_news`](https://huggingface.co/datasets/mteb/spanish_news) • **License:** mit • [Learn more →](https://huggingface.co/datasets/MarcOrfilaCarreras/spanish-news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | spa | News, Written | derived | found |



#### SpanishSentimentClassification

A Spanish dataset for sentiment classification.

**Dataset:** [`sepidmnorozy/Spanish_sentiment`](https://huggingface.co/datasets/sepidmnorozy/Spanish_sentiment) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/sepidmnorozy/Spanish_sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | spa | Reviews, Written | derived | found |



#### SpanishSentimentClassification.v2

A Spanish dataset for sentiment classification.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/spanish_sentiment`](https://huggingface.co/datasets/mteb/spanish_sentiment) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/sepidmnorozy/Spanish_sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | spa | Reviews, Written | derived | found |



#### SwahiliNewsClassification

Dataset for Swahili News Classification, categorized with 6 domains (Local News (Kitaifa), International News (Kimataifa), Finance News (Uchumi), Health News (Afya), Sports News (Michezo), and Entertainment News (Burudani)). Building and Optimizing Swahili Language Models: Techniques, Embeddings, and Datasets

**Dataset:** [`Mollel/SwahiliNewsClassification`](https://huggingface.co/datasets/Mollel/SwahiliNewsClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/Mollel/SwahiliNewsClassification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swa | News, Written | derived | found |



#### SwahiliNewsClassification.v2

Dataset for Swahili News Classification, categorized with 6 domains (Local News (Kitaifa), International News (Kimataifa), Finance News (Uchumi), Health News (Afya), Sports News (Michezo), and Entertainment News (Burudani)). Building and Optimizing Swahili Language Models: Techniques, Embeddings, and Datasets
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/swahili_news`](https://huggingface.co/datasets/mteb/swahili_news) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/Mollel/SwahiliNewsClassification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swa | News, Written | derived | found |



#### SweRecClassification

A Swedish dataset for sentiment classification on review

**Dataset:** [`mteb/swerec_classification`](https://huggingface.co/datasets/mteb/swerec_classification) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swe | Reviews, Written | derived | found |



#### SweRecClassification.v2

A Swedish dataset for sentiment classification on review
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/swe_rec`](https://huggingface.co/datasets/mteb/swe_rec) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swe | Reviews, Written | derived | found |



#### SwedishSentimentClassification

Dataset of Swedish reviews scarped from various public available websites

**Dataset:** [`mteb/SwedishSentimentClassification`](https://huggingface.co/datasets/mteb/SwedishSentimentClassification) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/swedish_reviews)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swe | Reviews, Written | derived | found |



#### SwedishSentimentClassification.v2

Dataset of Swedish reviews scarped from various public available websites
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/swedish_sentiment`](https://huggingface.co/datasets/mteb/swedish_sentiment) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/swedish_reviews)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swe | Reviews, Written | derived | found |



#### SwissJudgementClassification

Multilingual, diachronic dataset of Swiss Federal Supreme Court cases annotated with the respective binarized judgment outcome (approval/dismissal)

**Dataset:** [`rcds/swiss_judgment_prediction`](https://huggingface.co/datasets/rcds/swiss_judgment_prediction) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2021.nllp-1.3/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu, fra, ita | Legal, Written | expert-annotated | found |



#### SynPerChatbotConvSAAnger

Synthetic Persian Chatbot Conversational Sentiment Analysis Anger

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-anger`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-anger) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotConvSAFear

Synthetic Persian Chatbot Conversational Sentiment Analysis Fear

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-fear`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-fear) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotConvSAFriendship

Synthetic Persian Chatbot Conversational Sentiment Analysis Friendship

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-friendship`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-friendship) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotConvSAHappiness

Synthetic Persian Chatbot Conversational Sentiment Analysis Happiness

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-happiness`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-happiness) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotConvSAJealousy

Synthetic Persian Chatbot Conversational Sentiment Analysis Jealousy

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-jealousy`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-jealousy) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotConvSALove

Synthetic Persian Chatbot Conversational Sentiment Analysis Love

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-love`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-love) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotConvSASadness

Synthetic Persian Chatbot Conversational Sentiment Analysis Sadness

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-sadness`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-sadness) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotConvSASatisfaction

Synthetic Persian Chatbot Conversational Sentiment Analysis Satisfaction

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-satisfaction`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-satisfaction) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotConvSASurprise

Synthetic Persian Chatbot Conversational Sentiment Analysis Surprise

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-surprise`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-surprise) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotConvSAToneChatbotClassification

Synthetic Persian Chatbot Conversational Sentiment Analysis Tone Chatbot Classification

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-tone-chatbot-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-tone-chatbot-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotConvSAToneUserClassification

Synthetic Persian Chatbot Conversational Sentiment Analysis Tone User

**Dataset:** [`MCINext/chatbot-conversational-sentiment-analysis-tone-user-classification`](https://huggingface.co/datasets/MCINext/chatbot-conversational-sentiment-analysis-tone-user-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotRAGToneChatbotClassification

Synthetic Persian Chatbot RAG Tone Chatbot Classification

**Dataset:** [`MCINext/synthetic-persian-chatbot-rag-tone-chatbot-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-tone-chatbot-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotRAGToneUserClassification

Synthetic Persian Chatbot RAG Tone User Classification

**Dataset:** [`MCINext/synthetic-persian-chatbot-rag-tone-user-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-tone-user-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotSatisfactionLevelClassification

Synthetic Persian Chatbot Satisfaction Level Classification

**Dataset:** [`MCINext/synthetic-persian-chatbot-satisfaction-level-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-satisfaction-level-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotToneChatbotClassification

Synthetic Persian Chatbot Tone Chatbot Classification

**Dataset:** [`MCINext/synthetic-persian-chatbot-tone-chatbot-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-tone-chatbot-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerChatbotToneUserClassification

Synthetic Persian Chatbot Tone User Classification

**Dataset:** [`MCINext/synthetic-persian-chatbot-tone-user-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-tone-user-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



#### SynPerTextToneClassification

Persian Text Tone

**Dataset:** [`MCINext/synthetic-persian-text-tone-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-text-tone-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | not specified | LM-generated | LM-generated and verified |



#### SynPerTextToneClassification.v2

Persian Text Tone
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/syn_per_text_tone`](https://huggingface.co/datasets/mteb/syn_per_text_tone) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | not specified | LM-generated | LM-generated and verified |



#### TNews

Short Text Classification for News

**Dataset:** [`C-MTEB/TNews-classification`](https://huggingface.co/datasets/C-MTEB/TNews-classification) • **License:** not specified • [Learn more →](https://www.cluebenchmarks.com/introduce.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### TNews.v2

Short Text Classification for News
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/t_news`](https://huggingface.co/datasets/mteb/t_news) • **License:** not specified • [Learn more →](https://www.cluebenchmarks.com/introduce.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### TamilNewsClassification

A Tamil dataset for 6-class classification of Tamil news articles

**Dataset:** [`mlexplorer008/tamil_news_classification`](https://huggingface.co/datasets/mlexplorer008/tamil_news_classification) • **License:** mit • [Learn more →](https://github.com/vanangamudi/tamil-news-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | tam | News, Written | derived | found |



#### TamilNewsClassification.v2

A Tamil dataset for 6-class classification of Tamil news articles
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/tamil_news`](https://huggingface.co/datasets/mteb/tamil_news) • **License:** mit • [Learn more →](https://github.com/vanangamudi/tamil-news-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | tam | News, Written | derived | found |



#### TelemarketingSalesRuleLegalBenchClassification

Determine how 16 C.F.R. § 310.3(a)(1) and 16 C.F.R. § 310.3(a)(2) (governing deceptive practices) apply to different fact patterns. This dataset is designed to test a model’s ability to apply 16 C.F.R. § 310.3(a)(1) and 16 C.F.R. § 310.3(a)(2) of the Telemarketing Sales Rule to a simple fact pattern with a clear outcome. Each fact pattern ends with the question: “Is this a violation of the Telemarketing Sales Rule?” Each fact pattern is paired with the answer “Yes” or the answer “No.” Fact patterns are listed in the column “text,” and answers are listed in the column “label.”

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### TeluguAndhraJyotiNewsClassification

A Telugu dataset for 5-class classification of Telugu news articles

**Dataset:** [`mlexplorer008/telugu_news_classification`](https://huggingface.co/datasets/mlexplorer008/telugu_news_classification) • **License:** mit • [Learn more →](https://github.com/AnushaMotamarri/Telugu-Newspaper-Article-Dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | tel | News, Written | derived | found |



#### TeluguAndhraJyotiNewsClassification.v2

A Telugu dataset for 5-class classification of Telugu news articles
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/telugu_andhra_jyoti_news`](https://huggingface.co/datasets/mteb/telugu_andhra_jyoti_news) • **License:** mit • [Learn more →](https://github.com/AnushaMotamarri/Telugu-Newspaper-Article-Dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | tel | News, Written | derived | found |



#### TenKGnadClassification

10k German News Articles Dataset (10kGNAD) contains news articles from the online Austrian newspaper website DER Standard with their topic classification (9 classes).

**Dataset:** [`mteb/TenKGnadClassification`](https://huggingface.co/datasets/mteb/TenKGnadClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://tblock.github.io/10kGNAD/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu | News, Written | expert-annotated | found |



#### TenKGnadClassification.v2

10k German News Articles Dataset (10kGNAD) contains news articles from the online Austrian newspaper website DER Standard with their topic classification (9 classes).
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/ten_k_gnad`](https://huggingface.co/datasets/mteb/ten_k_gnad) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://tblock.github.io/10kGNAD/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu | News, Written | expert-annotated | found |



#### TextualismToolDictionariesLegalBenchClassification

Determine if a paragraph from a judicial opinion is applying a form textualism that relies on the dictionary meaning of terms.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### TextualismToolPlainLegalBenchClassification

Determine if a paragraph from a judicial opinion is applying a form textualism that relies on the ordinary (“plain”) meaning of terms.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### ToxicChatClassification

This dataset contains toxicity annotations on 10K user
            prompts collected from the Vicuna online demo. We utilize a human-AI
            collaborative annotation framework to guarantee the quality of annotation
            while maintaining a feasible annotation workload. The details of data
            collection, pre-processing, and annotation can be found in our paper.
            We believe that ToxicChat can be a valuable resource to drive further
            advancements toward building a safe and healthy environment for user-AI
            interactions.
            Only human annotated samples are selected here.

**Dataset:** [`lmsys/toxic-chat`](https://huggingface.co/datasets/lmsys/toxic-chat) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2023.findings-emnlp.311/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Constructed, Written | expert-annotated | found |



#### ToxicChatClassification.v2

This dataset contains toxicity annotations on 10K user
            prompts collected from the Vicuna online demo. We utilize a human-AI
            collaborative annotation framework to guarantee the quality of annotation
            while maintaining a feasible annotation workload. The details of data
            collection, pre-processing, and annotation can be found in our paper.
            We believe that ToxicChat can be a valuable resource to drive further
            advancements toward building a safe and healthy environment for user-AI
            interactions.
            Only human annotated samples are selected here.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/toxic_chat`](https://huggingface.co/datasets/mteb/toxic_chat) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2023.findings-emnlp.311/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Constructed, Written | expert-annotated | found |



#### ToxicConversationsClassification

Collection of comments from the Civil Comments platform together with annotations if the comment is toxic or not.

**Dataset:** [`mteb/toxic_conversations_50k`](https://huggingface.co/datasets/mteb/toxic_conversations_50k) • **License:** cc-by-4.0 • [Learn more →](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | human-annotated | found |



#### ToxicConversationsClassification.v2

Collection of comments from the Civil Comments platform together with annotations if the comment is toxic or not.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/toxic_conversations`](https://huggingface.co/datasets/mteb/toxic_conversations) • **License:** cc-by-4.0 • [Learn more →](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | human-annotated | found |



#### ToxicConversationsVNClassification

A translated dataset from Collection of comments from the Civil Comments platform together with annotations if the comment is toxic or not.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/toxic-conversations-50k-vn`](https://huggingface.co/datasets/GreenNode/toxic-conversations-50k-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Social, Written | derived | machine-translated and LM verified |



#### TswanaNewsClassification

Tswana News Classification Dataset

**Dataset:** [`dsfsi/daily-news-dikgang`](https://huggingface.co/datasets/dsfsi/daily-news-dikgang) • **License:** cc-by-sa-4.0 • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-031-49002-6_17)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tsn | News, Written | derived | found |



#### TswanaNewsClassification.v2

Tswana News Classification Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/tswana_news`](https://huggingface.co/datasets/mteb/tswana_news) • **License:** cc-by-sa-4.0 • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-031-49002-6_17)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tsn | News, Written | derived | found |



#### TurkicClassification

A dataset of news classification in three Turkic languages.

**Dataset:** [`Electrotubbie/classification_Turkic_languages`](https://huggingface.co/datasets/Electrotubbie/classification_Turkic_languages) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/Electrotubbie/classification_Turkic_languages/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | bak, kaz, kir | News, Written | derived | found |



#### TurkishMovieSentimentClassification

Turkish Movie Review Dataset

**Dataset:** [`asparius/Turkish-Movie-Review`](https://huggingface.co/datasets/asparius/Turkish-Movie-Review) • **License:** not specified • [Learn more →](https://www.win.tue.nl/~mpechen/publications/pubs/MT_WISDOM2013.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tur | Reviews, Written | derived | found |



#### TurkishMovieSentimentClassification.v2

Turkish Movie Review Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/turkish_movie_sentiment`](https://huggingface.co/datasets/mteb/turkish_movie_sentiment) • **License:** not specified • [Learn more →](https://www.win.tue.nl/~mpechen/publications/pubs/MT_WISDOM2013.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tur | Reviews, Written | derived | found |



#### TurkishProductSentimentClassification

Turkish Product Review Dataset

**Dataset:** [`asparius/Turkish-Product-Review`](https://huggingface.co/datasets/asparius/Turkish-Product-Review) • **License:** not specified • [Learn more →](https://www.win.tue.nl/~mpechen/publications/pubs/MT_WISDOM2013.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tur | Reviews, Written | derived | found |



#### TurkishProductSentimentClassification.v2

Turkish Product Review Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/turkish_product_sentiment`](https://huggingface.co/datasets/mteb/turkish_product_sentiment) • **License:** not specified • [Learn more →](https://www.win.tue.nl/~mpechen/publications/pubs/MT_WISDOM2013.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tur | Reviews, Written | derived | found |



#### TweetEmotionClassification

A dataset of 10,000 tweets that was created with the aim of covering the most frequently used emotion categories in Arabic tweets.

**Dataset:** [`mteb/TweetEmotionClassification`](https://huggingface.co/datasets/mteb/TweetEmotionClassification) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-77116-8_8)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Social, Written | human-annotated | found |



#### TweetEmotionClassification.v2

A dataset of 10,012 tweets that was created with the aim of covering the most frequently used emotion categories in Arabic tweets.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/TweetEmotionClassification`](https://huggingface.co/datasets/mteb/TweetEmotionClassification) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-77116-8_8)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Social, Written | human-annotated | found |



#### TweetSarcasmClassification

Arabic sarcasm detection dataset, which was created through the reannotation of available Arabic sentiment analysis datasets.

**Dataset:** [`iabufarha/ar_sarcasm`](https://huggingface.co/datasets/iabufarha/ar_sarcasm) • **License:** mit • [Learn more →](https://aclanthology.org/2020.osact-1.5/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Social, Written | human-annotated | found |



#### TweetSarcasmClassification.v2

Arabic sarcasm detection dataset, which was created through the reannotation of available Arabic sentiment analysis datasets.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/tweet_sarcasm`](https://huggingface.co/datasets/mteb/tweet_sarcasm) • **License:** mit • [Learn more →](https://aclanthology.org/2020.osact-1.5/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Social, Written | human-annotated | found |



#### TweetSentimentClassification

A multilingual Sentiment Analysis dataset consisting of tweets in 8 different languages.

**Dataset:** [`mteb/tweet_sentiment_multilingual`](https://huggingface.co/datasets/mteb/tweet_sentiment_multilingual) • **License:** cc-by-3.0 • [Learn more →](https://aclanthology.org/2022.lrec-1.27)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara, deu, eng, fra, hin, ... (8) | Social, Written | human-annotated | found |



#### TweetSentimentExtractionClassification



**Dataset:** [`mteb/tweet_sentiment_extraction`](https://huggingface.co/datasets/mteb/tweet_sentiment_extraction) • **License:** not specified • [Learn more →](https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | human-annotated | found |



#### TweetSentimentExtractionClassification.v2


        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/tweet_sentiment_extraction`](https://huggingface.co/datasets/mteb/tweet_sentiment_extraction) • **License:** not specified • [Learn more →](https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | human-annotated | found |



#### TweetSentimentExtractionVNClassification

A collection of translated tweets annotated for sentiment extraction.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.

**Dataset:** [`GreenNode/tweet-sentiment-extraction-vn`](https://huggingface.co/datasets/GreenNode/tweet-sentiment-extraction-vn) • **License:** cc-by-sa-4.0 • [Learn more →](https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Social, Written | derived | machine-translated and LM verified |



#### TweetTopicSingleClassification

Topic classification dataset on Twitter with 6 labels. Each instance of
        TweetTopic comes with a timestamp which distributes from September 2019 to August 2021.
        Tweets were preprocessed before the annotation to normalize some artifacts, converting
        URLs into a special token {{URL}} and non-verified usernames into {{USERNAME}}. For verified
        usernames, we replace its display name (or account name) with symbols {@}.
        

**Dataset:** [`cardiffnlp/tweet_topic_single`](https://huggingface.co/datasets/cardiffnlp/tweet_topic_single) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2209.09824)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | News, Social, Written | expert-annotated | found |



#### TweetTopicSingleClassification.v2

Topic classification dataset on Twitter with 6 labels. Each instance of
        TweetTopic comes with a timestamp which distributes from September 2019 to August 2021.
        Tweets were preprocessed before the annotation to normalize some artifacts, converting
        URLs into a special token {{URL}} and non-verified usernames into {{USERNAME}}. For verified
        usernames, we replace its display name (or account name) with symbols {@}.

        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/tweet_topic_single`](https://huggingface.co/datasets/mteb/tweet_topic_single) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2209.09824)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | News, Social, Written | expert-annotated | found |



#### UCCVCommonLawLegalBenchClassification

Determine if a contract is governed by the Uniform Commercial Code (UCC) or the common law of contracts.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### UkrFormalityClassification


        This dataset contains Ukrainian Formality Classification dataset obtained by
        trainslating English GYAFC data.
        English data source: https://aclanthology.org/N18-1012/
        Translation into Ukrainian language using model: https://huggingface.co/facebook/nllb-200-distilled-600M
        Additionally, the dataset was balanced, witha labels: 0 - informal, 1 - formal.
        

**Dataset:** [`ukr-detect/ukr-formality-dataset-translated-gyafc`](https://huggingface.co/datasets/ukr-detect/ukr-formality-dataset-translated-gyafc) • **License:** openrail++ • [Learn more →](https://huggingface.co/datasets/ukr-detect/ukr-formality-dataset-translated-gyafc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ukr | News, Written | derived | machine-translated |



#### UkrFormalityClassification.v2


        This dataset contains Ukrainian Formality Classification dataset obtained by
        trainslating English GYAFC data.
        English data source: https://aclanthology.org/N18-1012/
        Translation into Ukrainian language using model: https://huggingface.co/facebook/nllb-200-distilled-600M
        Additionally, the dataset was balanced, witha labels: 0 - informal, 1 - formal.

        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/ukr_formality`](https://huggingface.co/datasets/mteb/ukr_formality) • **License:** openrail++ • [Learn more →](https://huggingface.co/datasets/ukr-detect/ukr-formality-dataset-translated-gyafc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ukr | News, Written | derived | machine-translated |



#### UnfairTOSLegalBenchClassification

Given a clause from a terms-of-service contract, determine the category the clause belongs to. The purpose of this task is classifying clauses in Terms of Service agreements. Clauses have been annotated by into nine categories: ['Arbitration', 'Unilateral change', 'Content removal', 'Jurisdiction', 'Choice of law', 'Limitation of liability', 'Unilateral termination', 'Contract by using', 'Other']. The first eight categories correspond to clauses that would potentially be deemed potentially unfair. The last category (Other) corresponds to clauses in agreements which don’t fit into these categories.

**Dataset:** [`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



#### UrduRomanSentimentClassification

The Roman Urdu dataset is a data corpus comprising of more than 20000 records tagged for sentiment (Positive, Negative, Neutral)

**Dataset:** [`mteb/UrduRomanSentimentClassification`](https://huggingface.co/datasets/mteb/UrduRomanSentimentClassification) • **License:** mit • [Learn more →](https://archive.ics.uci.edu/dataset/458/roman+urdu+data+set)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | urd | Social, Written | derived | found |



#### UrduRomanSentimentClassification.v2

The Roman Urdu dataset is a data corpus comprising of more than 20000 records tagged for sentiment (Positive, Negative, Neutral)
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/urdu_roman_sentiment`](https://huggingface.co/datasets/mteb/urdu_roman_sentiment) • **License:** mit • [Learn more →](https://archive.ics.uci.edu/dataset/458/roman+urdu+data+set)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | urd | Social, Written | derived | found |



#### VieStudentFeedbackClassification

A Vietnamese dataset for classification of student feedback

**Dataset:** [`uitnlp/vietnamese_students_feedback`](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) • **License:** mit • [Learn more →](https://ieeexplore.ieee.org/document/8573337)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Reviews, Written | human-annotated | created |



#### VieStudentFeedbackClassification.v2

A Vietnamese dataset for classification of student feedback
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/vie_student_feedback`](https://huggingface.co/datasets/mteb/vie_student_feedback) • **License:** mit • [Learn more →](https://ieeexplore.ieee.org/document/8573337)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Reviews, Written | human-annotated | created |



#### WRIMEClassification

A dataset of Japanese social network rated for sentiment

**Dataset:** [`shunk031/wrime`](https://huggingface.co/datasets/shunk031/wrime) • **License:** https://huggingface.co/datasets/shunk031/wrime#licensing-information • [Learn more →](https://aclanthology.org/2021.naacl-main.169/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | jpn | Social, Written | human-annotated | found |



#### WRIMEClassification.v2

A dataset of Japanese social network rated for sentiment
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wrime`](https://huggingface.co/datasets/mteb/wrime) • **License:** https://huggingface.co/datasets/shunk031/wrime#licensing-information • [Learn more →](https://aclanthology.org/2021.naacl-main.169/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | jpn | Social, Written | human-annotated | found |



#### Waimai

Sentiment Analysis of user reviews on takeaway platforms

**Dataset:** [`C-MTEB/waimai-classification`](https://huggingface.co/datasets/C-MTEB/waimai-classification) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### Waimai.v2

Sentiment Analysis of user reviews on takeaway platforms
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/waimai`](https://huggingface.co/datasets/mteb/waimai) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



#### WikipediaBioMetChemClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEasy2GeneExpressionVsMetallurgyClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaEasy2GeneExpressionVsMetallurgyClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaBioMetChemClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wikipedia_bio_met_chem`](https://huggingface.co/datasets/mteb/wikipedia_bio_met_chem) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaBiolumNeurochemClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaMedium2BioluminescenceVsNeurochemistryClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaMedium2BioluminescenceVsNeurochemistryClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaChemEngSpecialtiesClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaMedium5Classification`](https://huggingface.co/datasets/BASF-AI/WikipediaMedium5Classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaChemFieldsClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEZ10Classification`](https://huggingface.co/datasets/BASF-AI/WikipediaEZ10Classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaChemFieldsClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wikipedia_chem_fields`](https://huggingface.co/datasets/mteb/wikipedia_chem_fields) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaChemistryTopicsClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEasy10Classification`](https://huggingface.co/datasets/BASF-AI/WikipediaEasy10Classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaCompChemSpectroscopyClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaMedium2ComputationalVsSpectroscopistsClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaMedium2ComputationalVsSpectroscopistsClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaCompChemSpectroscopyClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wikipedia_comp_chem_spectroscopy`](https://huggingface.co/datasets/mteb/wikipedia_comp_chem_spectroscopy) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaCryobiologySeparationClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEasy5Classification`](https://huggingface.co/datasets/BASF-AI/WikipediaEasy5Classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaCrystallographyAnalyticalClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaMedium2CrystallographyVsChromatographyTitrationpHClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaMedium2CrystallographyVsChromatographyTitrationpHClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaCrystallographyAnalyticalClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wikipedia_crystallography_analytical`](https://huggingface.co/datasets/mteb/wikipedia_crystallography_analytical) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaGreenhouseEnantiopureClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEasy2GreenhouseVsEnantiopureClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaEasy2GreenhouseVsEnantiopureClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaIsotopesFissionClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaHard2IsotopesVsFissionProductsNuclearFissionClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaHard2IsotopesVsFissionProductsNuclearFissionClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaLuminescenceClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaHard2BioluminescenceVsLuminescenceClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaHard2BioluminescenceVsLuminescenceClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaOrganicInorganicClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEasy2SpecialClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaEasy2SpecialClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaSaltsSemiconductorsClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaHard2SaltsVsSemiconductorMaterialsClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaHard2SaltsVsSemiconductorMaterialsClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaSolidStateColloidalClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEasy2SolidStateVsColloidalClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaEasy2SolidStateVsColloidalClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaTheoreticalAppliedClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEZ2Classification`](https://huggingface.co/datasets/BASF-AI/WikipediaEZ2Classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WikipediaTheoreticalAppliedClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wikipedia_theoretical_applied`](https://huggingface.co/datasets/mteb/wikipedia_theoretical_applied) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



#### WisesightSentimentClassification

Wisesight Sentiment Corpus: Social media messages in Thai language with sentiment label (positive, neutral, negative, question)

**Dataset:** [`mteb/WisesightSentimentClassification`](https://huggingface.co/datasets/mteb/WisesightSentimentClassification) • **License:** cc0-1.0 • [Learn more →](https://github.com/PyThaiNLP/wisesight-sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | tha | News, Social, Written | expert-annotated | found |



#### WisesightSentimentClassification.v2

Wisesight Sentiment Corpus: Social media messages in Thai language with sentiment label (positive, neutral, negative, question)
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wisesight_sentiment`](https://huggingface.co/datasets/mteb/wisesight_sentiment) • **License:** cc0-1.0 • [Learn more →](https://github.com/PyThaiNLP/wisesight-sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | tha | News, Social, Written | expert-annotated | found |



#### WongnaiReviewsClassification

Wongnai features over 200,000 restaurants, beauty salons, and spas across Thailand on its platform, with detailed information about each merchant and user reviews. In this dataset there are 5 classes corressponding each star rating

**Dataset:** [`Wongnai/wongnai_reviews`](https://huggingface.co/datasets/Wongnai/wongnai_reviews) • **License:** lgpl-3.0 • [Learn more →](https://github.com/wongnai/wongnai-corpus)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tha | Reviews, Written | derived | found |



#### YahooAnswersTopicsClassification

Dataset composed of questions and answers from Yahoo Answers, categorized into topics.

**Dataset:** [`community-datasets/yahoo_answers_topics`](https://huggingface.co/datasets/community-datasets/yahoo_answers_topics) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/yahoo_answers_topics)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Web, Written | human-annotated | found |



#### YahooAnswersTopicsClassification.v2

Dataset composed of questions and answers from Yahoo Answers, categorized into topics.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/yahoo_answers_topics`](https://huggingface.co/datasets/mteb/yahoo_answers_topics) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/yahoo_answers_topics)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Web, Written | human-annotated | found |



#### YelpReviewFullClassification

Yelp Review Full is a dataset for sentiment analysis, containing 5 classes corresponding to ratings 1-5.

**Dataset:** [`Yelp/yelp_review_full`](https://huggingface.co/datasets/Yelp/yelp_review_full) • **License:** https://huggingface.co/datasets/Yelp/yelp_review_full#licensing-information • [Learn more →](https://arxiv.org/abs/1509.01626)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | derived | found |



#### YelpReviewFullClassification.v2

Yelp Review Full is a dataset for sentiment analysis, containing 5 classes corresponding to ratings 1-5.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/yelp_review_full`](https://huggingface.co/datasets/mteb/yelp_review_full) • **License:** https://huggingface.co/datasets/Yelp/yelp_review_full#licensing-information • [Learn more →](https://arxiv.org/abs/1509.01626)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | derived | found |



#### YueOpenriceReviewClassification

A Cantonese dataset for review classification

**Dataset:** [`izhx/yue-openrice-review`](https://huggingface.co/datasets/izhx/yue-openrice-review) • **License:** not specified • [Learn more →](https://github.com/Christainx/Dataset_Cantonese_Openrice)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | yue | Reviews, Spoken | human-annotated | found |



#### YueOpenriceReviewClassification.v2

A Cantonese dataset for review classification
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/yue_openrice_review`](https://huggingface.co/datasets/mteb/yue_openrice_review) • **License:** not specified • [Learn more →](https://github.com/Christainx/Dataset_Cantonese_Openrice)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | yue | Reviews, Spoken | human-annotated | found |
