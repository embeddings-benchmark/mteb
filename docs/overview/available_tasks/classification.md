
# Classification

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 465

#### AJGT

Arabic Jordanian General Tweets (AJGT) Corpus consisted of 1,800 tweets annotated as positive and negative. Modern Standard Arabic (MSA) or Jordanian dialect.

**Dataset:** [`komari6/ajgt_twitter_ar`](https://huggingface.co/datasets/komari6/ajgt_twitter_ar) • **License:** afl-3.0 • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-60042-0_66/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{alomari2017arabic,
      author = {Alomari, Khaled Mohammad and ElSherif, Hatem M and Shaalan, Khaled},
      booktitle = {International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems},
      organization = {Springer},
      pages = {602--610},
      title = {Arabic tweets sentimental analysis using machine learning},
      year = {2017},
    }

    ```




#### AJGT.v2

Arabic Jordanian General Tweets (AJGT) Corpus consisted of 1,800 tweets (900 for training and 900 for testing) annotated as positive and negative. Modern Standard Arabic (MSA) or Jordanian dialect.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/ajgt`](https://huggingface.co/datasets/mteb/ajgt) • **License:** afl-3.0 • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-60042-0_66/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{alomari2017arabic,
      author = {Alomari, Khaled Mohammad and ElSherif, Hatem M and Shaalan, Khaled},
      booktitle = {International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems},
      organization = {Springer},
      pages = {602--610},
      title = {Arabic tweets sentimental analysis using machine learning},
      year = {2017},
    }

    ```




#### AfriSentiClassification

AfriSenti is the largest sentiment analysis dataset for under-represented African languages.

**Dataset:** [`mteb/AfriSentiClassification`](https://huggingface.co/datasets/mteb/AfriSentiClassification) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2302.08956)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | amh, arq, ary, hau, ibo, ... (12) | Social, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{Muhammad2023AfriSentiAT,
      author = {Shamsuddeen Hassan Muhammad and Idris Abdulmumin and Abinew Ali Ayele and Nedjma Ousidhoum and David Ifeoluwa Adelani and Seid Muhie Yimam and Ibrahim Sa'id Ahmad and Meriem Beloucif and Saif Mohammad and Sebastian Ruder and Oumaima Hourrane and Pavel Brazdil and Felermino D'ario M'ario Ant'onio Ali and Davis Davis and Salomey Osei and Bello Shehu Bello and Falalu Ibrahim and Tajuddeen Gwadabe and Samuel Rutunda and Tadesse Belay and Wendimu Baye Messelle and Hailu Beshada Balcha and Sisay Adugna Chala and Hagos Tesfahun Gebremichael and Bernard Opoku and Steven Arthur},
      title = {AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages},
      year = {2023},
    }

    ```




#### AfriSentiLangClassification

AfriSentiLID is the largest LID classification dataset for African Languages.

**Dataset:** [`HausaNLP/afrisenti-lid-data`](https://huggingface.co/datasets/HausaNLP/afrisenti-lid-data) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/HausaNLP/afrisenti-lid-data/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | amh, arq, ary, hau, ibo, ... (12) | Social, Written | derived | found |



??? quote "Citation"


    ```bibtex


    ```




#### AllegroReviews

A Polish dataset for sentiment classification on reviews from e-commerce marketplace Allegro.

**Dataset:** [`PL-MTEB/allegro-reviews`](https://huggingface.co/datasets/PL-MTEB/allegro-reviews) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2020.acl-main.111.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Reviews | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{rybak-etal-2020-klej,
      abstract = {In recent years, a series of Transformer-based models unlocked major improvements in general natural language understanding (NLU) tasks. Such a fast pace of research would not be possible without general NLU benchmarks, which allow for a fair comparison of the proposed methods. However, such benchmarks are available only for a handful of languages. To alleviate this issue, we introduce a comprehensive multi-task benchmark for the Polish language understanding, accompanied by an online leaderboard. It consists of a diverse set of tasks, adopted from existing datasets for named entity recognition, question-answering, textual entailment, and others. We also introduce a new sentiment analysis task for the e-commerce domain, named Allegro Reviews (AR). To ensure a common evaluation scheme and promote models that generalize to different NLU tasks, the benchmark includes datasets from varying domains and applications. Additionally, we release HerBERT, a Transformer-based model trained specifically for the Polish language, which has the best average performance and obtains the best results for three out of nine tasks. Finally, we provide an extensive evaluation, including several standard baselines and recently proposed, multilingual Transformer-based models.},
      address = {Online},
      author = {Rybak, Piotr  and
    Mroczkowski, Robert  and
    Tracz, Janusz  and
    Gawlik, Ireneusz},
      booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
      doi = {10.18653/v1/2020.acl-main.111},
      editor = {Jurafsky, Dan  and
    Chai, Joyce  and
    Schluter, Natalie  and
    Tetreault, Joel},
      month = jul,
      pages = {1191--1201},
      publisher = {Association for Computational Linguistics},
      title = {{KLEJ}: Comprehensive Benchmark for {P}olish Language Understanding},
      url = {https://aclanthology.org/2020.acl-main.111/},
      year = {2020},
    }

    ```




#### AllegroReviews.v2

A Polish dataset for sentiment classification on reviews from e-commerce marketplace Allegro.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/allegro_reviews`](https://huggingface.co/datasets/mteb/allegro_reviews) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2020.acl-main.111.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Reviews | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{rybak-etal-2020-klej,
      abstract = {In recent years, a series of Transformer-based models unlocked major improvements in general natural language understanding (NLU) tasks. Such a fast pace of research would not be possible without general NLU benchmarks, which allow for a fair comparison of the proposed methods. However, such benchmarks are available only for a handful of languages. To alleviate this issue, we introduce a comprehensive multi-task benchmark for the Polish language understanding, accompanied by an online leaderboard. It consists of a diverse set of tasks, adopted from existing datasets for named entity recognition, question-answering, textual entailment, and others. We also introduce a new sentiment analysis task for the e-commerce domain, named Allegro Reviews (AR). To ensure a common evaluation scheme and promote models that generalize to different NLU tasks, the benchmark includes datasets from varying domains and applications. Additionally, we release HerBERT, a Transformer-based model trained specifically for the Polish language, which has the best average performance and obtains the best results for three out of nine tasks. Finally, we provide an extensive evaluation, including several standard baselines and recently proposed, multilingual Transformer-based models.},
      address = {Online},
      author = {Rybak, Piotr  and
    Mroczkowski, Robert  and
    Tracz, Janusz  and
    Gawlik, Ireneusz},
      booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
      doi = {10.18653/v1/2020.acl-main.111},
      editor = {Jurafsky, Dan  and
    Chai, Joyce  and
    Schluter, Natalie  and
    Tetreault, Joel},
      month = jul,
      pages = {1191--1201},
      publisher = {Association for Computational Linguistics},
      title = {{KLEJ}: Comprehensive Benchmark for {P}olish Language Understanding},
      url = {https://aclanthology.org/2020.acl-main.111/},
      year = {2020},
    }

    ```




#### AmazonCounterfactualClassification

A collection of Amazon customer reviews annotated for counterfactual detection pair classification.

**Dataset:** [`mteb/amazon_counterfactual`](https://huggingface.co/datasets/mteb/amazon_counterfactual) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2104.06893)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu, eng, jpn | Reviews, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{oneill-etal-2021-wish,
      abstract = {Counterfactual statements describe events that did not or cannot take place. We consider the problem of counterfactual detection (CFD) in product reviews. For this purpose, we annotate a multilingual CFD dataset from Amazon product reviews covering counterfactual statements written in English, German, and Japanese languages. The dataset is unique as it contains counterfactuals in multiple languages, covers a new application area of e-commerce reviews, and provides high quality professional annotations. We train CFD models using different text representation methods and classifiers. We find that these models are robust against the selectional biases introduced due to cue phrase-based sentence selection. Moreover, our CFD dataset is compatible with prior datasets and can be merged to learn accurate CFD models. Applying machine translation on English counterfactual examples to create multilingual data performs poorly, demonstrating the language-specificity of this problem, which has been ignored so far.},
      address = {Online and Punta Cana, Dominican Republic},
      author = {O{'}Neill, James  and
    Rozenshtein, Polina  and
    Kiryo, Ryuichi  and
    Kubota, Motoko  and
    Bollegala, Danushka},
      booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
      doi = {10.18653/v1/2021.emnlp-main.568},
      editor = {Moens, Marie-Francine  and
    Huang, Xuanjing  and
    Specia, Lucia  and
    Yih, Scott Wen-tau},
      month = nov,
      pages = {7092--7108},
      publisher = {Association for Computational Linguistics},
      title = {{I} Wish {I} Would Have Loved This One, But {I} Didn{'}t {--} A Multilingual Dataset for Counterfactual Detection in Product Review},
      url = {https://aclanthology.org/2021.emnlp-main.568},
      year = {2021},
    }

    ```




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




#### AmazonPolarityClassification

Amazon Polarity Classification Dataset.

**Dataset:** [`mteb/amazon_polarity`](https://huggingface.co/datasets/mteb/amazon_polarity) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/amazon_polarity)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{McAuley2013HiddenFA,
      author = {Julian McAuley and Jure Leskovec},
      journal = {Proceedings of the 7th ACM conference on Recommender systems},
      title = {Hidden factors and hidden topics: understanding rating dimensions with review text},
      url = {https://api.semanticscholar.org/CorpusID:6440341},
      year = {2013},
    }

    ```




#### AmazonPolarityClassification.v2

Amazon Polarity Classification Dataset.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/amazon_polarity`](https://huggingface.co/datasets/mteb/amazon_polarity) • **License:** apache-2.0 • [Learn more →](https://huggingface.co/datasets/amazon_polarity)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{McAuley2013HiddenFA,
      author = {Julian McAuley and Jure Leskovec},
      journal = {Proceedings of the 7th ACM conference on Recommender systems},
      title = {Hidden factors and hidden topics: understanding rating dimensions with review text},
      url = {https://api.semanticscholar.org/CorpusID:6440341},
      year = {2013},
    }

    ```




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




#### AmazonReviewsClassification

A collection of Amazon reviews specifically designed to aid research in multilingual text classification.

**Dataset:** [`mteb/AmazonReviewsClassification`](https://huggingface.co/datasets/mteb/AmazonReviewsClassification) • **License:** https://docs.opendata.aws/amazon-reviews-ml/license.txt • [Learn more →](https://arxiv.org/abs/2010.02573)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn, deu, eng, fra, jpn, ... (6) | Reviews, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{keung2020multilingual,
      archiveprefix = {arXiv},
      author = {Phillip Keung and Yichao Lu and György Szarvas and Noah A. Smith},
      eprint = {2010.02573},
      primaryclass = {cs.CL},
      title = {The Multilingual Amazon Reviews Corpus},
      year = {2020},
    }

    ```




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




#### AngryTweetsClassification

A sentiment dataset with 3 classes (positiv, negativ, neutral) for Danish tweets

**Dataset:** [`DDSC/angry-tweets`](https://huggingface.co/datasets/DDSC/angry-tweets) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2021.nodalida-main.53/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{pauli2021danlp,
      author = {Pauli, Amalie Brogaard and Barrett, Maria and Lacroix, Oph{\'e}lie and Hvingelby, Rasmus},
      booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
      pages = {460--466},
      title = {DaNLP: An open-source toolkit for Danish Natural Language Processing},
      year = {2021},
    }

    ```




#### AngryTweetsClassification.v2

A sentiment dataset with 3 classes (positiv, negativ, neutral) for Danish tweets
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/angry_tweets`](https://huggingface.co/datasets/mteb/angry_tweets) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2021.nodalida-main.53/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{pauli2021danlp,
      author = {Pauli, Amalie Brogaard and Barrett, Maria and Lacroix, Oph{\'e}lie and Hvingelby, Rasmus},
      booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
      pages = {460--466},
      title = {DaNLP: An open-source toolkit for Danish Natural Language Processing},
      year = {2021},
    }

    ```




#### ArxivClassification

Classification Dataset of Arxiv Papers

**Dataset:** [`mteb/ArxivClassification`](https://huggingface.co/datasets/mteb/ArxivClassification) • **License:** not specified • [Learn more →](https://ieeexplore.ieee.org/document/8675939)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Academic, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{8675939,
      author = {He, Jun and Wang, Liqun and Liu, Liu and Feng, Jiao and Wu, Hao},
      doi = {10.1109/ACCESS.2019.2907992},
      journal = {IEEE Access},
      number = {},
      pages = {40707-40718},
      title = {Long Document Classification From Local Word Glimpses via Recurrent Attention Learning},
      volume = {7},
      year = {2019},
    }

    ```




#### ArxivClassification.v2

Classification Dataset of Arxiv Papers
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/arxiv`](https://huggingface.co/datasets/mteb/arxiv) • **License:** not specified • [Learn more →](https://ieeexplore.ieee.org/document/8675939)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Academic, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{8675939,
      author = {He, Jun and Wang, Liqun and Liu, Liu and Feng, Jiao and Wu, Hao},
      doi = {10.1109/ACCESS.2019.2907992},
      journal = {IEEE Access},
      number = {},
      pages = {40707-40718},
      title = {Long Document Classification From Local Word Glimpses via Recurrent Attention Learning},
      volume = {7},
      year = {2019},
    }

    ```




#### Banking77Classification

Dataset composed of online banking queries annotated with their corresponding intents.

**Dataset:** [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) • **License:** mit • [Learn more →](https://arxiv.org/abs/2003.04807)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{casanueva-etal-2020-efficient,
      address = {Online},
      author = {Casanueva, I{\~n}igo  and
    Tem{\v{c}}inas, Tadas  and
    Gerz, Daniela  and
    Henderson, Matthew  and
    Vuli{\'c}, Ivan},
      booktitle = {Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI},
      doi = {10.18653/v1/2020.nlp4convai-1.5},
      editor = {Wen, Tsung-Hsien  and
    Celikyilmaz, Asli  and
    Yu, Zhou  and
    Papangelis, Alexandros  and
    Eric, Mihail  and
    Kumar, Anuj  and
    Casanueva, I{\~n}igo  and
    Shah, Rushin},
      month = jul,
      pages = {38--45},
      publisher = {Association for Computational Linguistics},
      title = {Efficient Intent Detection with Dual Sentence Encoders},
      url = {https://aclanthology.org/2020.nlp4convai-1.5},
      year = {2020},
    }

    ```




#### Banking77Classification.v2

Dataset composed of online banking queries annotated with their corresponding intents.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) • **License:** mit • [Learn more →](https://arxiv.org/abs/2003.04807)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{casanueva-etal-2020-efficient,
      address = {Online},
      author = {Casanueva, I{\~n}igo  and
    Tem{\v{c}}inas, Tadas  and
    Gerz, Daniela  and
    Henderson, Matthew  and
    Vuli{\'c}, Ivan},
      booktitle = {Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI},
      doi = {10.18653/v1/2020.nlp4convai-1.5},
      editor = {Wen, Tsung-Hsien  and
    Celikyilmaz, Asli  and
    Yu, Zhou  and
    Papangelis, Alexandros  and
    Eric, Mihail  and
    Kumar, Anuj  and
    Casanueva, I{\~n}igo  and
    Shah, Rushin},
      month = jul,
      pages = {38--45},
      publisher = {Association for Computational Linguistics},
      title = {Efficient Intent Detection with Dual Sentence Encoders},
      url = {https://aclanthology.org/2020.nlp4convai-1.5},
      year = {2020},
    }

    ```




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




#### BengaliDocumentClassification

Dataset for News Classification, categorized with 13 domains.

**Dataset:** [`dialect-ai/shironaam`](https://huggingface.co/datasets/dialect-ai/shironaam) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/2023.eacl-main.4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ben | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{akash-etal-2023-shironaam,
      address = {Dubrovnik, Croatia},
      author = {Akash, Abu Ubaida  and
    Nayeem, Mir Tafseer  and
    Shohan, Faisal Tareque  and
    Islam, Tanvir},
      booktitle = {Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics},
      month = may,
      pages = {52--67},
      publisher = {Association for Computational Linguistics},
      title = {Shironaam: {B}engali News Headline Generation using Auxiliary Information},
      url = {https://aclanthology.org/2023.eacl-main.4},
      year = {2023},
    }

    ```




#### BengaliDocumentClassification.v2

Dataset for News Classification, categorized with 13 domains.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/bengali_document`](https://huggingface.co/datasets/mteb/bengali_document) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/2023.eacl-main.4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ben | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{akash-etal-2023-shironaam,
      address = {Dubrovnik, Croatia},
      author = {Akash, Abu Ubaida  and
    Nayeem, Mir Tafseer  and
    Shohan, Faisal Tareque  and
    Islam, Tanvir},
      booktitle = {Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics},
      month = may,
      pages = {52--67},
      publisher = {Association for Computational Linguistics},
      title = {Shironaam: {B}engali News Headline Generation using Auxiliary Information},
      url = {https://aclanthology.org/2023.eacl-main.4},
      year = {2023},
    }

    ```




#### BengaliHateSpeechClassification

The Bengali Hate Speech Dataset is a Bengali-language dataset of news articles collected from various Bengali media sources and categorized based on the type of hate in the text.

**Dataset:** [`rezacsedu/bn_hate_speech`](https://huggingface.co/datasets/rezacsedu/bn_hate_speech) • **License:** mit • [Learn more →](https://huggingface.co/datasets/bn_hate_speech)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ben | News, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{karim2020BengaliNLP,
      author = {Karim, Md. Rezaul and Chakravarti, Bharathi Raja and P. McCrae, John and Cochez, Michael},
      booktitle = {7th IEEE International Conference on Data Science and Advanced Analytics (IEEE DSAA,2020)},
      publisher = {IEEE},
      title = {Classification Benchmarks for Under-resourced Bengali Language based on Multichannel Convolutional-LSTM Network},
      year = {2020},
    }

    ```




#### BengaliHateSpeechClassification.v2

The Bengali Hate Speech Dataset is a Bengali-language dataset of news articles collected from various Bengali media sources and categorized based on the type of hate in the text.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/bengali_hate_speech`](https://huggingface.co/datasets/mteb/bengali_hate_speech) • **License:** mit • [Learn more →](https://huggingface.co/datasets/bn_hate_speech)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ben | News, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{karim2020BengaliNLP,
      author = {Karim, Md. Rezaul and Chakravarti, Bharathi Raja and P. McCrae, John and Cochez, Michael},
      booktitle = {7th IEEE International Conference on Data Science and Advanced Analytics (IEEE DSAA,2020)},
      publisher = {IEEE},
      title = {Classification Benchmarks for Under-resourced Bengali Language based on Multichannel Convolutional-LSTM Network},
      year = {2020},
    }

    ```




#### BengaliSentimentAnalysis

dataset contains 3307 Negative reviews and 8500 Positive reviews collected and manually annotated from Youtube Bengali drama.

**Dataset:** [`Akash190104/bengali_sentiment_analysis`](https://huggingface.co/datasets/Akash190104/bengali_sentiment_analysis) • **License:** cc-by-4.0 • [Learn more →](https://data.mendeley.com/datasets/p6zc7krs37/4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ben | Reviews, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{sazzed2020cross,
      author = {Sazzed, Salim},
      booktitle = {Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020)},
      pages = {50--60},
      title = {Cross-lingual sentiment classification in low-resource Bengali language},
      year = {2020},
    }

    ```




#### BengaliSentimentAnalysis.v2

dataset contains 2854 Negative reviews and 7238 Positive reviews collected and manually annotated from Youtube Bengali drama.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/bengali_sentiment_analysis`](https://huggingface.co/datasets/mteb/bengali_sentiment_analysis) • **License:** cc-by-4.0 • [Learn more →](https://data.mendeley.com/datasets/p6zc7krs37/4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ben | Reviews, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{sazzed2020cross,
      author = {Sazzed, Salim},
      booktitle = {Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020)},
      pages = {50--60},
      title = {Cross-lingual sentiment classification in low-resource Bengali language},
      year = {2020},
    }

    ```




#### BulgarianStoreReviewSentimentClassfication

Bulgarian online store review dataset for sentiment classification.

**Dataset:** [`artist/Bulgarian-Online-Store-Feedback-Text-Analysis`](https://huggingface.co/datasets/artist/Bulgarian-Online-Store-Feedback-Text-Analysis) • **License:** cc-by-4.0 • [Learn more →](https://doi.org/10.7910/DVN/TXIK9P)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | bul | Reviews, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @data{DVN/TXIK9P_2018,
      author = {Georgieva-Trifonova, Tsvetanka and Stefanova, Milena and Kalchev, Stefan},
      doi = {10.7910/DVN/TXIK9P},
      publisher = {Harvard Dataverse},
      title = {{Dataset for ``Customer Feedback Text Analysis for Online Stores Reviews in Bulgarian''}},
      url = {https://doi.org/10.7910/DVN/TXIK9P},
      version = {V1},
      year = {2018},
    }

    ```




#### CBD

Polish Tweets annotated for cyberbullying detection.

**Dataset:** [`PL-MTEB/cbd`](https://huggingface.co/datasets/PL-MTEB/cbd) • **License:** bsd-3-clause • [Learn more →](http://2019.poleval.pl/files/poleval2019.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @proceedings{ogr:kob:19:poleval,
      address = {Warsaw, Poland},
      editor = {Maciej Ogrodniczuk and Łukasz Kobyliński},
      isbn = {978-83-63159-28-3},
      publisher = {Institute of Computer Science, Polish Academy of Sciences},
      title = {{Proceedings of the PolEval 2019 Workshop}},
      url = {http://2019.poleval.pl/files/poleval2019.pdf},
      year = {2019},
    }

    ```




#### CBD.v2

Polish Tweets annotated for cyberbullying detection.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/cbd`](https://huggingface.co/datasets/mteb/cbd) • **License:** bsd-3-clause • [Learn more →](http://2019.poleval.pl/files/poleval2019.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @proceedings{ogr:kob:19:poleval,
      address = {Warsaw, Poland},
      editor = {Maciej Ogrodniczuk and Łukasz Kobyliński},
      isbn = {978-83-63159-28-3},
      publisher = {Institute of Computer Science, Polish Academy of Sciences},
      title = {{Proceedings of the PolEval 2019 Workshop}},
      url = {http://2019.poleval.pl/files/poleval2019.pdf},
      year = {2019},
    }

    ```




#### CSFDCZMovieReviewSentimentClassification

The dataset contains 30k user reviews from csfd.cz in Czech.

**Dataset:** [`fewshot-goes-multilingual/cs_csfd-movie-reviews`](https://huggingface.co/datasets/fewshot-goes-multilingual/cs_csfd-movie-reviews) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2304.01922)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{štefánik2023resources,
      archiveprefix = {arXiv},
      author = {Michal Štefánik and Marek Kadlčík and Piotr Gramacki and Petr Sojka},
      eprint = {2304.01922},
      primaryclass = {cs.CL},
      title = {Resources and Few-shot Learners for In-context Learning in Slavic Languages},
      year = {2023},
    }

    ```




#### CSFDCZMovieReviewSentimentClassification.v2

The dataset contains 30k user reviews from csfd.cz in Czech.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/csfdcz_movie_review_sentiment`](https://huggingface.co/datasets/mteb/csfdcz_movie_review_sentiment) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2304.01922)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{štefánik2023resources,
      archiveprefix = {arXiv},
      author = {Michal Štefánik and Marek Kadlčík and Piotr Gramacki and Petr Sojka},
      eprint = {2304.01922},
      primaryclass = {cs.CL},
      title = {Resources and Few-shot Learners for In-context Learning in Slavic Languages},
      year = {2023},
    }

    ```




#### CSFDSKMovieReviewSentimentClassification

The dataset contains 30k user reviews from csfd.cz in Slovak.

**Dataset:** [`fewshot-goes-multilingual/sk_csfd-movie-reviews`](https://huggingface.co/datasets/fewshot-goes-multilingual/sk_csfd-movie-reviews) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2304.01922)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | slk | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{štefánik2023resources,
      archiveprefix = {arXiv},
      author = {Michal Štefánik and Marek Kadlčík and Piotr Gramacki and Petr Sojka},
      eprint = {2304.01922},
      primaryclass = {cs.CL},
      title = {Resources and Few-shot Learners for In-context Learning in Slavic Languages},
      year = {2023},
    }

    ```




#### CSFDSKMovieReviewSentimentClassification.v2

The dataset contains 30k user reviews from csfd.cz in Slovak.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/csfdsk_movie_review_sentiment`](https://huggingface.co/datasets/mteb/csfdsk_movie_review_sentiment) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2304.01922)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | slk | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{štefánik2023resources,
      archiveprefix = {arXiv},
      author = {Michal Štefánik and Marek Kadlčík and Piotr Gramacki and Petr Sojka},
      eprint = {2304.01922},
      primaryclass = {cs.CL},
      title = {Resources and Few-shot Learners for In-context Learning in Slavic Languages},
      year = {2023},
    }

    ```




#### CUADAffiliateLicenseLicenseeLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if a clause describes a license grant to a licensee (incl. sublicensor) and the affiliates of such licensee/sublicensor.

**Dataset:** [`mteb/CUADAffiliateLicenseLicenseeLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADAffiliateLicenseLicenseeLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADAffiliateLicenseLicensorLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause describes a license grant by affiliates of the licensor or that includes intellectual property of affiliates of the licensor.

**Dataset:** [`mteb/CUADAffiliateLicenseLicensorLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADAffiliateLicenseLicensorLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADAntiAssignmentLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause requires consent or notice of a party if the contract is assigned to a third party.

**Dataset:** [`mteb/CUADAntiAssignmentLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADAntiAssignmentLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADAuditRightsLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause gives a party the right to audit the books, records, or physical locations of the counterparty to ensure compliance with the contract.

**Dataset:** [`mteb/CUADAuditRightsLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADAuditRightsLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADCapOnLiabilityLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a cap on liability upon the breach of a party's obligation. This includes time limitation for the counterparty to bring claims or maximum amount for recovery.

**Dataset:** [`mteb/CUADCapOnLiabilityLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADCapOnLiabilityLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADChangeOfControlLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause gives one party the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law.

**Dataset:** [`mteb/CUADChangeOfControlLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADChangeOfControlLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADCompetitiveRestrictionExceptionLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause mentions exceptions or carveouts to Non-Compete, Exclusivity and No-Solicit of Customers.

**Dataset:** [`mteb/CUADCompetitiveRestrictionExceptionLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADCompetitiveRestrictionExceptionLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADCovenantNotToSueLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that a party is restricted from contesting the validity of the counterparty's ownership of intellectual property or otherwise bringing a claim against the counterparty for matters unrelated to the contract.

**Dataset:** [`mteb/CUADCovenantNotToSueLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADCovenantNotToSueLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADEffectiveDateLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies the date upon which the agreement becomes effective.

**Dataset:** [`mteb/CUADEffectiveDateLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADEffectiveDateLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADExclusivityLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies exclusive dealing commitment with the counterparty. This includes a commitment to procure all 'requirements' from one party of certain technology, goods, or services or a prohibition on licensing or selling technology, goods or services to third parties, or a prohibition on collaborating or working with other parties), whether during the contract or after the contract ends (or both).

**Dataset:** [`mteb/CUADExclusivityLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADExclusivityLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADExpirationDateLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies the date upon which the initial term expires.

**Dataset:** [`mteb/CUADExpirationDateLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADExpirationDateLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADGoverningLawLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies which state/country’s law governs the contract.

**Dataset:** [`mteb/CUADGoverningLawLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADGoverningLawLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADIPOwnershipAssignmentLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that intellectual property created by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events.

**Dataset:** [`mteb/CUADIPOwnershipAssignmentLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADIPOwnershipAssignmentLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADInsuranceLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if clause creates a requirement for insurance that must be maintained by one party for the benefit of the counterparty.

**Dataset:** [`mteb/CUADInsuranceLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADInsuranceLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADIrrevocableOrPerpetualLicenseLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a license grant that is irrevocable or perpetual.

**Dataset:** [`mteb/CUADIrrevocableOrPerpetualLicenseLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADIrrevocableOrPerpetualLicenseLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADJointIPOwnershipLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause provides for joint or shared ownership of intellectual property between the parties to the contract.

**Dataset:** [`mteb/CUADJointIPOwnershipLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADJointIPOwnershipLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADLicenseGrantLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause contains a license granted by one party to its counterparty.

**Dataset:** [`mteb/CUADLicenseGrantLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADLicenseGrantLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADLiquidatedDamagesLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause awards either party liquidated damages for breach or a fee upon the termination of a contract (termination fee).

**Dataset:** [`mteb/CUADLiquidatedDamagesLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADLiquidatedDamagesLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADMinimumCommitmentLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a minimum order size or minimum amount or units per time period that one party must buy from the counterparty.

**Dataset:** [`mteb/CUADMinimumCommitmentLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADMinimumCommitmentLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADMostFavoredNationLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms.

**Dataset:** [`mteb/CUADMostFavoredNationLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADMostFavoredNationLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADNoSolicitOfCustomersLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause restricts a party from contracting or soliciting customers or partners of the counterparty, whether during the contract or after the contract ends (or both).

**Dataset:** [`mteb/CUADNoSolicitOfCustomersLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADNoSolicitOfCustomersLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADNoSolicitOfEmployeesLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause restricts a party's soliciting or hiring employees and/or contractors from the counterparty, whether during the contract or after the contract ends (or both).

**Dataset:** [`mteb/CUADNoSolicitOfEmployeesLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADNoSolicitOfEmployeesLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADNonCompeteLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause restricts the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector.

**Dataset:** [`mteb/CUADNonCompeteLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADNonCompeteLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADNonDisparagementLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause requires a party not to disparage the counterparty.

**Dataset:** [`mteb/CUADNonDisparagementLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADNonDisparagementLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADNonTransferableLicenseLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause limits the ability of a party to transfer the license being granted to a third party.

**Dataset:** [`mteb/CUADNonTransferableLicenseLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADNonTransferableLicenseLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADNoticePeriodToTerminateRenewalLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a notice period required to terminate renewal.

**Dataset:** [`mteb/CUADNoticePeriodToTerminateRenewalLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADNoticePeriodToTerminateRenewalLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADPostTerminationServicesLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause subjects a party to obligations after the termination or expiration of a contract, including any post-termination transition, payment, transfer of IP, wind-down, last-buy, or similar commitments.

**Dataset:** [`mteb/CUADPostTerminationServicesLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADPostTerminationServicesLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADPriceRestrictionsLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause places a restriction on the ability of a party to raise or reduce prices of technology, goods, or services provided.

**Dataset:** [`mteb/CUADPriceRestrictionsLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADPriceRestrictionsLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADRenewalTermLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a renewal term.

**Dataset:** [`mteb/CUADRenewalTermLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADRenewalTermLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADRevenueProfitSharingLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause require a party to share revenue or profit with the counterparty for any technology, goods, or services.

**Dataset:** [`mteb/CUADRevenueProfitSharingLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADRevenueProfitSharingLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADRofrRofoRofnLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause grant one party a right of first refusal, right of first offer or right of first negotiation to purchase, license, market, or distribute equity interest, technology, assets, products or services.

**Dataset:** [`mteb/CUADRofrRofoRofnLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADRofrRofoRofnLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADSourceCodeEscrowLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause requires one party to deposit its source code into escrow with a third party, which can be released to the counterparty upon the occurrence of certain events (bankruptcy, insolvency, etc.).

**Dataset:** [`mteb/CUADSourceCodeEscrowLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADSourceCodeEscrowLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADTerminationForConvenienceLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that one party can terminate this contract without cause (solely by giving a notice and allowing a waiting period to expire).

**Dataset:** [`mteb/CUADTerminationForConvenienceLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADTerminationForConvenienceLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADThirdPartyBeneficiaryLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that that there a non-contracting party who is a beneficiary to some or all of the clauses in the contract and therefore can enforce its rights against a contracting party.

**Dataset:** [`mteb/CUADThirdPartyBeneficiaryLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADThirdPartyBeneficiaryLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADUncappedLiabilityLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that a party's liability is uncapped upon the breach of its obligation in the contract. This also includes uncap liability for a particular type of breach such as IP infringement or breach of confidentiality obligation.

**Dataset:** [`mteb/CUADUncappedLiabilityLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADUncappedLiabilityLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADUnlimitedAllYouCanEatLicenseLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause grants one party an “enterprise,” “all you can eat” or unlimited usage license.

**Dataset:** [`mteb/CUADUnlimitedAllYouCanEatLicenseLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADUnlimitedAllYouCanEatLicenseLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADVolumeRestrictionLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a fee increase or consent requirement, etc. if one party's use of the product/services exceeds certain threshold.

**Dataset:** [`mteb/CUADVolumeRestrictionLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADVolumeRestrictionLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CUADWarrantyDurationLegalBenchClassification

This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a duration of any warranty against defects or errors in technology, products, or services provided under the contract.

**Dataset:** [`mteb/CUADWarrantyDurationLegalBenchClassification`](https://huggingface.co/datasets/mteb/CUADWarrantyDurationLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CanadaTaxCourtOutcomesLegalBenchClassification

The input is an excerpt of text from Tax Court of Canada decisions involving appeals of tax related matters. The task is to classify whether the excerpt includes the outcome of the appeal, and if so, to specify whether the appeal was allowed or dismissed. Partial success (e.g. appeal granted on one tax year but dismissed on another) counts as allowed (with the exception of costs orders which are disregarded). Where the excerpt does not clearly articulate an outcome, the system should indicate other as the outcome. Categorizing case outcomes is a common task that legal researchers complete in order to gather datasets involving outcomes in legal processes for the purposes of quantitative empirical legal research.

**Dataset:** [`mteb/CanadaTaxCourtOutcomesLegalBenchClassification`](https://huggingface.co/datasets/mteb/CanadaTaxCourtOutcomesLegalBenchClassification) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




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



??? quote "Citation"


    ```bibtex

    @inproceedings{zotova-etal-2020-multilingual,
      author = {Zotova, Elena  and
    Agerri, Rodrigo  and
    Nu{\~n}ez, Manuel  and
    Rigau, German},
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
    Mazo, H{\'e}l{\`e}ne  and
    Moreno, Asuncion  and
    Odijk, Jan  and
    Piperidis, Stelios},
      isbn = {979-10-95546-34-4},
      month = may,
      pages = {1368--1375},
      publisher = {European Language Resources Association},
      title = {Multilingual Stance Detection in Tweets: The {C}atalonia Independence Corpus},
      year = {2020},
    }

    ```




#### ContractNLIConfidentialityOfAgreementLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA provides that the Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.

**Dataset:** [`mteb/ContractNLIConfidentialityOfAgreementLegalBenchClassification`](https://huggingface.co/datasets/mteb/ContractNLIConfidentialityOfAgreementLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @article{koreeda2021contractnli,
      author = {Koreeda, Yuta and Manning, Christopher D},
      journal = {arXiv preprint arXiv:2110.01799},
      title = {ContractNLI: A dataset for document-level natural language inference for contracts},
      year = {2021},
    }

    ```




#### ContractNLIExplicitIdentificationLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that all Confidential Information shall be expressly identified by the Disclosing Party.

**Dataset:** [`mteb/ContractNLIExplicitIdentificationLegalBenchClassification`](https://huggingface.co/datasets/mteb/ContractNLIExplicitIdentificationLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @article{koreeda2021contractnli,
      author = {Koreeda, Yuta and Manning, Christopher D},
      journal = {arXiv preprint arXiv:2110.01799},
      title = {ContractNLI: A dataset for document-level natural language inference for contracts},
      year = {2021},
    }

    ```




#### ContractNLIInclusionOfVerballyConveyedInformationLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that Confidential Information may include verbally conveyed information.

**Dataset:** [`mteb/ContractNLIInclusionOfVerballyConveyedInformationLegalBenchClassification`](https://huggingface.co/datasets/mteb/ContractNLIInclusionOfVerballyConveyedInformationLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @article{koreeda2021contractnli,
      author = {Koreeda, Yuta and Manning, Christopher D},
      journal = {arXiv preprint arXiv:2110.01799},
      title = {ContractNLI: A dataset for document-level natural language inference for contracts},
      year = {2021},
    }

    ```




#### ContractNLILimitedUseLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.

**Dataset:** [`mteb/ContractNLILimitedUseLegalBenchClassification`](https://huggingface.co/datasets/mteb/ContractNLILimitedUseLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @article{koreeda2021contractnli,
      author = {Koreeda, Yuta and Manning, Christopher D},
      journal = {arXiv preprint arXiv:2110.01799},
      title = {ContractNLI: A dataset for document-level natural language inference for contracts},
      year = {2021},
    }

    ```




#### ContractNLINoLicensingLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Agreement shall not grant Receiving Party any right to Confidential Information.

**Dataset:** [`mteb/ContractNLINoLicensingLegalBenchClassification`](https://huggingface.co/datasets/mteb/ContractNLINoLicensingLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @article{koreeda2021contractnli,
      author = {Koreeda, Yuta and Manning, Christopher D},
      journal = {arXiv preprint arXiv:2110.01799},
      title = {ContractNLI: A dataset for document-level natural language inference for contracts},
      year = {2021},
    }

    ```




#### ContractNLINoticeOnCompelledDisclosureLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.

**Dataset:** [`mteb/ContractNLINoticeOnCompelledDisclosureLegalBenchClassification`](https://huggingface.co/datasets/mteb/ContractNLINoticeOnCompelledDisclosureLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @article{koreeda2021contractnli,
      author = {Koreeda, Yuta and Manning, Christopher D},
      journal = {arXiv preprint arXiv:2110.01799},
      title = {ContractNLI: A dataset for document-level natural language inference for contracts},
      year = {2021},
    }

    ```




#### ContractNLIPermissibleAcquirementOfSimilarInformationLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may acquire information similar to Confidential Information from a third party.

**Dataset:** [`mteb/ContractNLIPermissibleAcquirementOfSimilarInformationLegalBenchClassification`](https://huggingface.co/datasets/mteb/ContractNLIPermissibleAcquirementOfSimilarInformationLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @article{koreeda2021contractnli,
      author = {Koreeda, Yuta and Manning, Christopher D},
      journal = {arXiv preprint arXiv:2110.01799},
      title = {ContractNLI: A dataset for document-level natural language inference for contracts},
      year = {2021},
    }

    ```




#### ContractNLIPermissibleCopyLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may create a copy of some Confidential Information in some circumstances.

**Dataset:** [`mteb/ContractNLIPermissibleCopyLegalBenchClassification`](https://huggingface.co/datasets/mteb/ContractNLIPermissibleCopyLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @article{koreeda2021contractnli,
      author = {Koreeda, Yuta and Manning, Christopher D},
      journal = {arXiv preprint arXiv:2110.01799},
      title = {ContractNLI: A dataset for document-level natural language inference for contracts},
      year = {2021},
    }

    ```




#### ContractNLIPermissibleDevelopmentOfSimilarInformationLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may independently develop information similar to Confidential Information.

**Dataset:** [`mteb/ContractNLIPermissibleDevelopmentOfSimilarInformationLegalBenchClassification`](https://huggingface.co/datasets/mteb/ContractNLIPermissibleDevelopmentOfSimilarInformationLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @article{koreeda2021contractnli,
      author = {Koreeda, Yuta and Manning, Christopher D},
      journal = {arXiv preprint arXiv:2110.01799},
      title = {ContractNLI: A dataset for document-level natural language inference for contracts},
      year = {2021},
    }

    ```




#### ContractNLIPermissiblePostAgreementPossessionLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.

**Dataset:** [`mteb/ContractNLIPermissiblePostAgreementPossessionLegalBenchClassification`](https://huggingface.co/datasets/mteb/ContractNLIPermissiblePostAgreementPossessionLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @article{koreeda2021contractnli,
      author = {Koreeda, Yuta and Manning, Christopher D},
      journal = {arXiv preprint arXiv:2110.01799},
      title = {ContractNLI: A dataset for document-level natural language inference for contracts},
      year = {2021},
    }

    ```




#### ContractNLIReturnOfConfidentialInformationLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.

**Dataset:** [`mteb/ContractNLIReturnOfConfidentialInformationLegalBenchClassification`](https://huggingface.co/datasets/mteb/ContractNLIReturnOfConfidentialInformationLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @article{koreeda2021contractnli,
      author = {Koreeda, Yuta and Manning, Christopher D},
      journal = {arXiv preprint arXiv:2110.01799},
      title = {ContractNLI: A dataset for document-level natural language inference for contracts},
      year = {2021},
    }

    ```




#### ContractNLISharingWithEmployeesLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may share some Confidential Information with some of Receiving Party's employees.

**Dataset:** [`mteb/ContractNLISharingWithEmployeesLegalBenchClassification`](https://huggingface.co/datasets/mteb/ContractNLISharingWithEmployeesLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @article{koreeda2021contractnli,
      author = {Koreeda, Yuta and Manning, Christopher D},
      journal = {arXiv preprint arXiv:2110.01799},
      title = {ContractNLI: A dataset for document-level natural language inference for contracts},
      year = {2021},
    }

    ```




#### ContractNLISharingWithThirdPartiesLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).

**Dataset:** [`mteb/ContractNLISharingWithThirdPartiesLegalBenchClassification`](https://huggingface.co/datasets/mteb/ContractNLISharingWithThirdPartiesLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @article{koreeda2021contractnli,
      author = {Koreeda, Yuta and Manning, Christopher D},
      journal = {arXiv preprint arXiv:2110.01799},
      title = {ContractNLI: A dataset for document-level natural language inference for contracts},
      year = {2021},
    }

    ```




#### ContractNLISurvivalOfObligationsLegalBenchClassification

This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that some obligations of Agreement may survive termination of Agreement.

**Dataset:** [`mteb/ContractNLISurvivalOfObligationsLegalBenchClassification`](https://huggingface.co/datasets/mteb/ContractNLISurvivalOfObligationsLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @article{koreeda2021contractnli,
      author = {Koreeda, Yuta and Manning, Christopher D},
      journal = {arXiv preprint arXiv:2110.01799},
      title = {ContractNLI: A dataset for document-level natural language inference for contracts},
      year = {2021},
    }

    ```




#### CorporateLobbyingLegalBenchClassification

The Corporate Lobbying task consists of determining whether a proposed Congressional bill may be relevant to a company based on a company's self-description in its SEC 10K filing.

**Dataset:** [`mteb/CorporateLobbyingLegalBenchClassification`](https://huggingface.co/datasets/mteb/CorporateLobbyingLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### CyrillicTurkicLangClassification

Cyrillic dataset of 8 Turkic languages spoken in Russia and former USSR

**Dataset:** [`tatiana-merz/cyrillic_turkic_langs`](https://huggingface.co/datasets/tatiana-merz/cyrillic_turkic_langs) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/tatiana-merz/cyrillic_turkic_langs)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | bak, chv, kaz, kir, krc, ... (9) | Web, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{goldhahn2012building,
      author = {Goldhahn, Dirk and Eckart, Thomas and Quasthoff, Uwe},
      booktitle = {Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC'12)},
      title = {Building Large Monolingual Dictionaries at the Leipzig Corpora Collection: From 100 to 200 Languages},
      year = {2012},
    }

    ```




#### CzechProductReviewSentimentClassification

User reviews of products on Czech e-shop Mall.cz with 3 sentiment classes (positive, neutral, negative)

**Dataset:** [`fewshot-goes-multilingual/cs_mall-product-reviews`](https://huggingface.co/datasets/fewshot-goes-multilingual/cs_mall-product-reviews) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/W13-1609/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{habernal-etal-2013-sentiment,
      address = {Atlanta, Georgia},
      author = {Habernal, Ivan  and
    Pt{\'a}{\v{c}}ek, Tom{\'a}{\v{s}}  and
    Steinberger, Josef},
      booktitle = {Proceedings of the 4th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis},
      editor = {Balahur, Alexandra  and
    van der Goot, Erik  and
    Montoyo, Andres},
      month = jun,
      pages = {65--74},
      publisher = {Association for Computational Linguistics},
      title = {Sentiment Analysis in {C}zech Social Media Using Supervised Machine Learning},
      url = {https://aclanthology.org/W13-1609},
      year = {2013},
    }

    ```




#### CzechProductReviewSentimentClassification.v2

User reviews of products on Czech e-shop Mall.cz with 3 sentiment classes (positive, neutral, negative)
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/czech_product_review_sentiment`](https://huggingface.co/datasets/mteb/czech_product_review_sentiment) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/W13-1609/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{habernal-etal-2013-sentiment,
      address = {Atlanta, Georgia},
      author = {Habernal, Ivan  and
    Pt{\'a}{\v{c}}ek, Tom{\'a}{\v{s}}  and
    Steinberger, Josef},
      booktitle = {Proceedings of the 4th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis},
      editor = {Balahur, Alexandra  and
    van der Goot, Erik  and
    Montoyo, Andres},
      month = jun,
      pages = {65--74},
      publisher = {Association for Computational Linguistics},
      title = {Sentiment Analysis in {C}zech Social Media Using Supervised Machine Learning},
      url = {https://aclanthology.org/W13-1609},
      year = {2013},
    }

    ```




#### CzechSoMeSentimentClassification

User comments on Facebook

**Dataset:** [`fewshot-goes-multilingual/cs_facebook-comments`](https://huggingface.co/datasets/fewshot-goes-multilingual/cs_facebook-comments) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/W13-1609/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{habernal-etal-2013-sentiment,
      address = {Atlanta, Georgia},
      author = {Habernal, Ivan  and
    Pt{\'a}{\v{c}}ek, Tom{\'a}{\v{s}}  and
    Steinberger, Josef},
      booktitle = {Proceedings of the 4th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis},
      editor = {Balahur, Alexandra  and
    van der Goot, Erik  and
    Montoyo, Andres},
      month = jun,
      pages = {65--74},
      publisher = {Association for Computational Linguistics},
      title = {Sentiment Analysis in {C}zech Social Media Using Supervised Machine Learning},
      url = {https://aclanthology.org/W13-1609},
      year = {2013},
    }

    ```




#### CzechSoMeSentimentClassification.v2

User comments on Facebook
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/czech_so_me_sentiment`](https://huggingface.co/datasets/mteb/czech_so_me_sentiment) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/W13-1609/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{habernal-etal-2013-sentiment,
      address = {Atlanta, Georgia},
      author = {Habernal, Ivan  and
    Pt{\'a}{\v{c}}ek, Tom{\'a}{\v{s}}  and
    Steinberger, Josef},
      booktitle = {Proceedings of the 4th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis},
      editor = {Balahur, Alexandra  and
    van der Goot, Erik  and
    Montoyo, Andres},
      month = jun,
      pages = {65--74},
      publisher = {Association for Computational Linguistics},
      title = {Sentiment Analysis in {C}zech Social Media Using Supervised Machine Learning},
      url = {https://aclanthology.org/W13-1609},
      year = {2013},
    }

    ```




#### CzechSubjectivityClassification

An Czech dataset for subjectivity classification.

**Dataset:** [`pauli31/czech-subjectivity-dataset`](https://huggingface.co/datasets/pauli31/czech-subjectivity-dataset) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2009.08712)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ces | Reviews, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{priban-steinberger-2022-czech,
      address = {Marseille, France},
      author = {P{\v{r}}ib{\'a}{\v{n}}, Pavel  and
    Steinberger, Josef},
      booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
      month = jun,
      pages = {1381--1391},
      publisher = {European Language Resources Association},
      title = {\{C\}zech Dataset for Cross-lingual Subjectivity Classification},
      url = {https://aclanthology.org/2022.lrec-1.148},
      year = {2022},
    }

    ```




#### DBpediaClassification

DBpedia14 is a dataset of English texts from Wikipedia articles, categorized into 14 non-overlapping classes based on their DBpedia ontology.

**Dataset:** [`fancyzhx/dbpedia_14`](https://huggingface.co/datasets/fancyzhx/dbpedia_14) • **License:** cc-by-sa-3.0 • [Learn more →](https://arxiv.org/abs/1509.01626)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{NIPS2015_250cf8b5,
      author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
      booktitle = {Advances in Neural Information Processing Systems},
      editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
      pages = {},
      publisher = {Curran Associates, Inc.},
      title = {Character-level Convolutional Networks for Text Classification},
      url = {https://proceedings.neurips.cc/paper_files/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf},
      volume = {28},
      year = {2015},
    }

    ```




#### DBpediaClassification.v2

DBpedia14 is a dataset of English texts from Wikipedia articles, categorized into 14 non-overlapping classes based on their DBpedia ontology.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/d_bpedia`](https://huggingface.co/datasets/mteb/d_bpedia) • **License:** cc-by-sa-3.0 • [Learn more →](https://arxiv.org/abs/1509.01626)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Encyclopaedic, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{NIPS2015_250cf8b5,
      author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
      booktitle = {Advances in Neural Information Processing Systems},
      editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
      pages = {},
      publisher = {Curran Associates, Inc.},
      title = {Character-level Convolutional Networks for Text Classification},
      url = {https://proceedings.neurips.cc/paper_files/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf},
      volume = {28},
      year = {2015},
    }

    ```




#### DKHateClassification

Danish Tweets annotated for Hate Speech either being Offensive or not

**Dataset:** [`DDSC/dkhate`](https://huggingface.co/datasets/DDSC/dkhate) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2020.lrec-1.430/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Social, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{sigurbergsson-derczynski-2020-offensive,
      abstract = {The presence of offensive language on social media platforms and the implications this poses is becoming a major concern in modern society. Given the enormous amount of content created every day, automatic methods are required to detect and deal with this type of content. Until now, most of the research has focused on solving the problem for the English language, while the problem is multilingual. We construct a Danish dataset DKhate containing user-generated comments from various social media platforms, and to our knowledge, the first of its kind, annotated for various types and target of offensive language. We develop four automatic classification systems, each designed to work for both the English and the Danish language. In the detection of offensive language in English, the best performing system achieves a macro averaged F1-score of 0.74, and the best performing system for Danish achieves a macro averaged F1-score of 0.70. In the detection of whether or not an offensive post is targeted, the best performing system for English achieves a macro averaged F1-score of 0.62, while the best performing system for Danish achieves a macro averaged F1-score of 0.73. Finally, in the detection of the target type in a targeted offensive post, the best performing system for English achieves a macro averaged F1-score of 0.56, and the best performing system for Danish achieves a macro averaged F1-score of 0.63. Our work for both the English and the Danish language captures the type and targets of offensive language, and present automatic methods for detecting different kinds of offensive language such as hate speech and cyberbullying.},
      address = {Marseille, France},
      author = {Sigurbergsson, Gudbjartur Ingi  and
    Derczynski, Leon},
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
    Mazo, H{\'e}l{\`e}ne  and
    Moreno, Asuncion  and
    Odijk, Jan  and
    Piperidis, Stelios},
      isbn = {979-10-95546-34-4},
      language = {English},
      month = may,
      pages = {3498--3508},
      publisher = {European Language Resources Association},
      title = {Offensive Language and Hate Speech Detection for {D}anish},
      url = {https://aclanthology.org/2020.lrec-1.430},
      year = {2020},
    }

    ```




#### DKHateClassification.v2

Danish Tweets annotated for Hate Speech either being Offensive or not
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/dk_hate`](https://huggingface.co/datasets/mteb/dk_hate) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2020.lrec-1.430/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Social, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{sigurbergsson-derczynski-2020-offensive,
      abstract = {The presence of offensive language on social media platforms and the implications this poses is becoming a major concern in modern society. Given the enormous amount of content created every day, automatic methods are required to detect and deal with this type of content. Until now, most of the research has focused on solving the problem for the English language, while the problem is multilingual. We construct a Danish dataset DKhate containing user-generated comments from various social media platforms, and to our knowledge, the first of its kind, annotated for various types and target of offensive language. We develop four automatic classification systems, each designed to work for both the English and the Danish language. In the detection of offensive language in English, the best performing system achieves a macro averaged F1-score of 0.74, and the best performing system for Danish achieves a macro averaged F1-score of 0.70. In the detection of whether or not an offensive post is targeted, the best performing system for English achieves a macro averaged F1-score of 0.62, while the best performing system for Danish achieves a macro averaged F1-score of 0.73. Finally, in the detection of the target type in a targeted offensive post, the best performing system for English achieves a macro averaged F1-score of 0.56, and the best performing system for Danish achieves a macro averaged F1-score of 0.63. Our work for both the English and the Danish language captures the type and targets of offensive language, and present automatic methods for detecting different kinds of offensive language such as hate speech and cyberbullying.},
      address = {Marseille, France},
      author = {Sigurbergsson, Gudbjartur Ingi  and
    Derczynski, Leon},
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
    Mazo, H{\'e}l{\`e}ne  and
    Moreno, Asuncion  and
    Odijk, Jan  and
    Piperidis, Stelios},
      isbn = {979-10-95546-34-4},
      language = {English},
      month = may,
      pages = {3498--3508},
      publisher = {European Language Resources Association},
      title = {Offensive Language and Hate Speech Detection for {D}anish},
      url = {https://aclanthology.org/2020.lrec-1.430},
      year = {2020},
    }

    ```




#### DadoEvalCoarseClassification

The DaDoEval dataset is a curated collection of 2,759 documents authored by Alcide De Gasperi, spanning the period from 1901 to 1954. Each document in the dataset is manually tagged with its date of issue.

**Dataset:** [`MattiaSangermano/DaDoEval`](https://huggingface.co/datasets/MattiaSangermano/DaDoEval) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://github.com/dhfbk/DaDoEval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ita | Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{menini2020dadoeval,
      author = {Menini, Stefano and Moretti, Giovanni and Sprugnoli, Rachele and Tonelli, Sara and others},
      booktitle = {Proceedings of the Seventh Evaluation Campaign of Natural Language Processing and Speech Tools for Italian. Final Workshop (EVALITA 2020)},
      organization = {Accademia University Press},
      pages = {391--397},
      title = {DaDoEval@ EVALITA 2020: Same-genre and cross-genre dating of historical documents},
      year = {2020},
    }

    ```




#### DalajClassification

A Swedish dataset for linguistic acceptability. Available as a part of Superlim.

**Dataset:** [`mteb/DalajClassification`](https://huggingface.co/datasets/mteb/DalajClassification) • **License:** cc-by-4.0 • [Learn more →](https://spraakbanken.gu.se/en/resources/superlim)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swe | Non-fiction, Written | expert-annotated | created |



??? quote "Citation"


    ```bibtex

    @misc{2105.06681,
      author = {Elena Volodina and Yousuf Ali Mohammed and Julia Klezl},
      eprint = {arXiv:2105.06681},
      title = {DaLAJ - a dataset for linguistic acceptability judgments for Swedish: Format, baseline, sharing},
      year = {2021},
    }

    ```




#### DalajClassification.v2

A Swedish dataset for linguistic acceptability. Available as a part of Superlim.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/dalaj`](https://huggingface.co/datasets/mteb/dalaj) • **License:** cc-by-4.0 • [Learn more →](https://spraakbanken.gu.se/en/resources/superlim)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swe | Non-fiction, Written | expert-annotated | created |



??? quote "Citation"


    ```bibtex

    @misc{2105.06681,
      author = {Elena Volodina and Yousuf Ali Mohammed and Julia Klezl},
      eprint = {arXiv:2105.06681},
      title = {DaLAJ - a dataset for linguistic acceptability judgments for Swedish: Format, baseline, sharing},
      year = {2021},
    }

    ```




#### DanishPoliticalCommentsClassification

A dataset of Danish political comments rated for sentiment

**Dataset:** [`mteb/DanishPoliticalCommentsClassification`](https://huggingface.co/datasets/mteb/DanishPoliticalCommentsClassification) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/danish_political_comments)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Social, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @techreport{SAMsentiment,
      author = {Mads Guldborg Kjeldgaard Kongsbak and Steffan Eybye Christensen and Lucas Høyberg Puvis~de~Chavannes and Peter Due Jensen},
      institution = {IT University of Copenhagen},
      title = {Sentiment Analysis Multitool, SAM},
      year = {2019},
    }

    ```




#### DanishPoliticalCommentsClassification.v2

A dataset of Danish political comments rated for sentiment
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/danish_political_comments`](https://huggingface.co/datasets/mteb/danish_political_comments) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/danish_political_comments)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Social, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @techreport{SAMsentiment,
      author = {Mads Guldborg Kjeldgaard Kongsbak and Steffan Eybye Christensen and Lucas Høyberg Puvis~de~Chavannes and Peter Due Jensen},
      institution = {IT University of Copenhagen},
      title = {Sentiment Analysis Multitool, SAM},
      year = {2019},
    }

    ```




#### Ddisco

A Danish Discourse dataset with values for coherence and source (Wikipedia or Reddit)

**Dataset:** [`DDSC/ddisco`](https://huggingface.co/datasets/DDSC/ddisco) • **License:** cc-by-sa-3.0 • [Learn more →](https://aclanthology.org/2022.lrec-1.260/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Non-fiction, Social, Written | expert-annotated | found |



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




#### Ddisco.v2

A Danish Discourse dataset with values for coherence and source (Wikipedia or Reddit)
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/ddisco_cohesion`](https://huggingface.co/datasets/mteb/ddisco_cohesion) • **License:** cc-by-sa-3.0 • [Learn more →](https://aclanthology.org/2022.lrec-1.260/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | Non-fiction, Social, Written | expert-annotated | found |



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




#### DeepSentiPers

Persian Sentiment Analysis Dataset

**Dataset:** [`PartAI/DeepSentiPers`](https://huggingface.co/datasets/PartAI/DeepSentiPers) • **License:** not specified • [Learn more →](https://github.com/JoyeBright/DeepSentiPers)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Reviews | derived | found |



??? quote "Citation"


    ```bibtex

    ```




#### DeepSentiPers.v2

Persian Sentiment Analysis Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/deep_senti_pers`](https://huggingface.co/datasets/mteb/deep_senti_pers) • **License:** not specified • [Learn more →](https://github.com/JoyeBright/DeepSentiPers)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Reviews | derived | found |



??? quote "Citation"


    ```bibtex

    ```




#### DefinitionClassificationLegalBenchClassification

This task consists of determining whether or not a sentence from a Supreme Court opinion offers a definition of a term.

**Dataset:** [`mteb/DefinitionClassificationLegalBenchClassification`](https://huggingface.co/datasets/mteb/DefinitionClassificationLegalBenchClassification) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### DigikalamagClassification

A total of 8,515 articles scraped from Digikala Online Magazine. This dataset includes seven different classes.

**Dataset:** [`mteb/DigikalamagClassification`](https://huggingface.co/datasets/mteb/DigikalamagClassification) • **License:** not specified • [Learn more →](https://hooshvare.github.io/docs/datasets/tc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Web | derived | found |



??? quote "Citation"


    ```bibtex

    ```




#### Diversity1LegalBenchClassification

Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 1).

**Dataset:** [`mteb/Diversity1LegalBenchClassification`](https://huggingface.co/datasets/mteb/Diversity1LegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### Diversity2LegalBenchClassification

Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 2).

**Dataset:** [`mteb/Diversity2LegalBenchClassification`](https://huggingface.co/datasets/mteb/Diversity2LegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### Diversity3LegalBenchClassification

Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 3).

**Dataset:** [`mteb/Diversity3LegalBenchClassification`](https://huggingface.co/datasets/mteb/Diversity3LegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### Diversity4LegalBenchClassification

Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 4).

**Dataset:** [`mteb/Diversity4LegalBenchClassification`](https://huggingface.co/datasets/mteb/Diversity4LegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### Diversity5LegalBenchClassification

Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 5).

**Dataset:** [`mteb/Diversity5LegalBenchClassification`](https://huggingface.co/datasets/mteb/Diversity5LegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### Diversity6LegalBenchClassification

Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 6).

**Dataset:** [`mteb/Diversity6LegalBenchClassification`](https://huggingface.co/datasets/mteb/Diversity6LegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### DutchBookReviewSentimentClassification

A Dutch book review for sentiment classification.

**Dataset:** [`mteb/DutchBookReviewSentimentClassification`](https://huggingface.co/datasets/mteb/DutchBookReviewSentimentClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://github.com/benjaminvdb/DBRD)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nld | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{DBLP:journals/corr/abs-1910-00896,
      archiveprefix = {arXiv},
      author = {Benjamin, van der Burgh and
    Suzan, Verberne},
      bibsource = {dblp computer science bibliography, https://dblp.org},
      biburl = {https://dblp.org/rec/journals/corr/abs-1910-00896.bib},
      eprint = {1910.00896},
      journal = {CoRR},
      timestamp = {Fri, 04 Oct 2019 12:28:06 +0200},
      title = {The merits of Universal Language Model Fine-tuning for Small Datasets
    - a case with Dutch book reviews},
      url = {http://arxiv.org/abs/1910.00896},
      volume = {abs/1910.00896},
      year = {2019},
    }

    ```




#### DutchBookReviewSentimentClassification.v2

A Dutch book review for sentiment classification.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/dutch_book_review_sentiment`](https://huggingface.co/datasets/mteb/dutch_book_review_sentiment) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://github.com/benjaminvdb/DBRD)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nld | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{DBLP:journals/corr/abs-1910-00896,
      archiveprefix = {arXiv},
      author = {Benjamin, van der Burgh and
    Suzan, Verberne},
      bibsource = {dblp computer science bibliography, https://dblp.org},
      biburl = {https://dblp.org/rec/journals/corr/abs-1910-00896.bib},
      eprint = {1910.00896},
      journal = {CoRR},
      timestamp = {Fri, 04 Oct 2019 12:28:06 +0200},
      title = {The merits of Universal Language Model Fine-tuning for Small Datasets
    - a case with Dutch book reviews},
      url = {http://arxiv.org/abs/1910.00896},
      volume = {abs/1910.00896},
      year = {2019},
    }

    ```




#### EmotionClassification

Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise.

**Dataset:** [`mteb/emotion`](https://huggingface.co/datasets/mteb/emotion) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/D18-1404)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{saravia-etal-2018-carer,
      abstract = {Emotions are expressed in nuanced ways, which varies by collective or individual experiences, knowledge, and beliefs. Therefore, to understand emotion, as conveyed through text, a robust mechanism capable of capturing and modeling different linguistic nuances and phenomena is needed. We propose a semi-supervised, graph-based algorithm to produce rich structural descriptors which serve as the building blocks for constructing contextualized affect representations from text. The pattern-based representations are further enriched with word embeddings and evaluated through several emotion recognition tasks. Our experimental results demonstrate that the proposed method outperforms state-of-the-art techniques on emotion recognition tasks.},
      address = {Brussels, Belgium},
      author = {Saravia, Elvis  and
    Liu, Hsien-Chi Toby  and
    Huang, Yen-Hao  and
    Wu, Junlin  and
    Chen, Yi-Shin},
      booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
      doi = {10.18653/v1/D18-1404},
      editor = {Riloff, Ellen  and
    Chiang, David  and
    Hockenmaier, Julia  and
    Tsujii, Jun{'}ichi},
      month = oct # {-} # nov,
      pages = {3687--3697},
      publisher = {Association for Computational Linguistics},
      title = {{CARER}: Contextualized Affect Representations for Emotion Recognition},
      url = {https://aclanthology.org/D18-1404},
      year = {2018},
    }

    ```




#### EmotionClassification.v2

Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/emotion`](https://huggingface.co/datasets/mteb/emotion) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/D18-1404)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{saravia-etal-2018-carer,
      abstract = {Emotions are expressed in nuanced ways, which varies by collective or individual experiences, knowledge, and beliefs. Therefore, to understand emotion, as conveyed through text, a robust mechanism capable of capturing and modeling different linguistic nuances and phenomena is needed. We propose a semi-supervised, graph-based algorithm to produce rich structural descriptors which serve as the building blocks for constructing contextualized affect representations from text. The pattern-based representations are further enriched with word embeddings and evaluated through several emotion recognition tasks. Our experimental results demonstrate that the proposed method outperforms state-of-the-art techniques on emotion recognition tasks.},
      address = {Brussels, Belgium},
      author = {Saravia, Elvis  and
    Liu, Hsien-Chi Toby  and
    Huang, Yen-Hao  and
    Wu, Junlin  and
    Chen, Yi-Shin},
      booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
      doi = {10.18653/v1/D18-1404},
      editor = {Riloff, Ellen  and
    Chiang, David  and
    Hockenmaier, Julia  and
    Tsujii, Jun{'}ichi},
      month = oct # {-} # nov,
      pages = {3687--3697},
      publisher = {Association for Computational Linguistics},
      title = {{CARER}: Contextualized Affect Representations for Emotion Recognition},
      url = {https://aclanthology.org/D18-1404},
      year = {2018},
    }

    ```




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




#### EstonianValenceClassification

Dataset containing annotated Estonian news data from the Postimees and Õhtuleht newspapers.

**Dataset:** [`kardosdrur/estonian-valence`](https://huggingface.co/datasets/kardosdrur/estonian-valence) • **License:** cc-by-4.0 • [Learn more →](https://figshare.com/articles/dataset/Estonian_Valence_Corpus_Eesti_valentsikorpus/24517054)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | est | News, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{Pajupuu2023,
      author = {Hille Pajupuu and Jaan Pajupuu and Rene Altrov and Kairi Tamuri},
      doi = {10.6084/m9.figshare.24517054.v1},
      month = {11},
      title = {{Estonian Valence Corpus  / Eesti valentsikorpus}},
      url = {https://figshare.com/articles/dataset/Estonian_Valence_Corpus_Eesti_valentsikorpus/24517054},
      year = {2023},
    }

    ```




#### EstonianValenceClassification.v2

Dataset containing annotated Estonian news data from the Postimees and Õhtuleht newspapers.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/estonian_valence`](https://huggingface.co/datasets/mteb/estonian_valence) • **License:** cc-by-4.0 • [Learn more →](https://figshare.com/articles/dataset/Estonian_Valence_Corpus_Eesti_valentsikorpus/24517054)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | est | News, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{Pajupuu2023,
      author = {Hille Pajupuu and Jaan Pajupuu and Rene Altrov and Kairi Tamuri},
      doi = {10.6084/m9.figshare.24517054.v1},
      month = {11},
      title = {{Estonian Valence Corpus  / Eesti valentsikorpus}},
      url = {https://figshare.com/articles/dataset/Estonian_Valence_Corpus_Eesti_valentsikorpus/24517054},
      year = {2023},
    }

    ```




#### FaIntentClassification

Questions in 4 different categories that a user might ask their voice assistant to do

**Dataset:** [`MCINext/FaIntent`](https://huggingface.co/datasets/MCINext/FaIntent) • **License:** gpl-3.0 • [Learn more →](https://github.com/HalflingWizard/FA-Intent-Classification-and-Slot-Filling)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | fas | Spoken | derived | found |



??? quote "Citation"


    ```bibtex

    ```




#### FilipinoHateSpeechClassification

Filipino Twitter dataset for sentiment classification.

**Dataset:** [`mteb/FilipinoHateSpeechClassification`](https://huggingface.co/datasets/mteb/FilipinoHateSpeechClassification) • **License:** not specified • [Learn more →](https://pcj.csp.org.ph/index.php/pcj/issue/download/29/PCJ%20V14%20N1%20pp1-14%202019)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fil | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{Cabasag-2019-hate-speech,
      author = {Neil Vicente Cabasag, Vicente Raphael Chan, Sean Christian Lim, Mark Edward Gonzales, and Charibeth Cheng},
      journal = {Philippine Computing Journal},
      month = {August},
      number = {1},
      title = {Hate speech in Philippine election-related tweets: Automatic detection and classification using natural language processing.},
      volume = {XIV},
      year = {2019},
    }

    ```




#### FilipinoHateSpeechClassification.v2

Filipino Twitter dataset for sentiment classification.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/filipino_hate_speech`](https://huggingface.co/datasets/mteb/filipino_hate_speech) • **License:** not specified • [Learn more →](https://pcj.csp.org.ph/index.php/pcj/issue/download/29/PCJ%20V14%20N1%20pp1-14%202019)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fil | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{Cabasag-2019-hate-speech,
      author = {Neil Vicente Cabasag, Vicente Raphael Chan, Sean Christian Lim, Mark Edward Gonzales, and Charibeth Cheng},
      journal = {Philippine Computing Journal},
      month = {August},
      number = {1},
      title = {Hate speech in Philippine election-related tweets: Automatic detection and classification using natural language processing.},
      volume = {XIV},
      year = {2019},
    }

    ```




#### FilipinoShopeeReviewsClassification

The Shopee reviews tl 15 dataset is constructed by randomly taking 2100 training samples and 450 samples for testing and validation for each review star from 1 to 5. In total, there are 10500 training samples and 2250 each in validation and testing samples.

**Dataset:** [`scaredmeow/shopee-reviews-tl-stars`](https://huggingface.co/datasets/scaredmeow/shopee-reviews-tl-stars) • **License:** mpl-2.0 • [Learn more →](https://uijrt.com/articles/v4/i8/UIJRTV4I80009.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fil | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{riegoenhancement,
      author = {Riego, Neil Christian R. and Villarba, Danny Bell and Sison, Ariel Antwaun Rolando C. and Pineda, Fernandez C. and Lagunzad, Herminiño C.},
      issue = {08},
      journal = {United International Journal for Research & Technology},
      pages = {72--82},
      title = {Enhancement to Low-Resource Text Classification via Sequential Transfer Learning},
      volume = {04},
    }

    ```




#### FinToxicityClassification


        This dataset is a DeepL -based machine translated version of the Jigsaw toxicity dataset for Finnish. The dataset is originally from a Kaggle competition https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data.
        The original dataset poses a multi-label text classification problem and includes the labels identity_attack, insult, obscene, severe_toxicity, threat and toxicity.
        Here adapted for toxicity classification, which is the most represented class.


**Dataset:** [`TurkuNLP/jigsaw_toxicity_pred_fi`](https://huggingface.co/datasets/TurkuNLP/jigsaw_toxicity_pred_fi) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.68)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | fin | News, Written | derived | machine-translated |



??? quote "Citation"


    ```bibtex

    @inproceedings{eskelinen-etal-2023-toxicity,
      author = {Eskelinen, Anni  and
    Silvala, Laura  and
    Ginter, Filip  and
    Pyysalo, Sampo  and
    Laippala, Veronika},
      booktitle = {Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)},
      month = may,
      title = {Toxicity Detection in {F}innish Using Machine Translation},
      year = {2023},
    }

    ```




#### FinToxicityClassification.v2


        This dataset is a DeepL -based machine translated version of the Jigsaw toxicity dataset for Finnish. The dataset is originally from a Kaggle competition https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data.
        The original dataset poses a multi-label text classification problem and includes the labels identity_attack, insult, obscene, severe_toxicity, threat and toxicity.
        Here adapted for toxicity classification, which is the most represented class.

        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/fin_toxicity`](https://huggingface.co/datasets/mteb/fin_toxicity) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.68)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | fin | News, Written | derived | machine-translated |



??? quote "Citation"


    ```bibtex

    @inproceedings{eskelinen-etal-2023-toxicity,
      author = {Eskelinen, Anni  and
    Silvala, Laura  and
    Ginter, Filip  and
    Pyysalo, Sampo  and
    Laippala, Veronika},
      booktitle = {Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)},
      month = may,
      title = {Toxicity Detection in {F}innish Using Machine Translation},
      year = {2023},
    }

    ```




#### FinancialPhrasebankClassification

Polar sentiment dataset of sentences from financial news, categorized by sentiment into positive, negative, or neutral.

**Dataset:** [`mteb/FinancialPhrasebankClassification`](https://huggingface.co/datasets/mteb/FinancialPhrasebankClassification) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://arxiv.org/abs/1307.5336)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Financial, News, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{Malo2014GoodDO,
      author = {P. Malo and A. Sinha and P. Korhonen and J. Wallenius and P. Takala},
      journal = {Journal of the Association for Information Science and Technology},
      title = {Good debt or bad debt: Detecting semantic orientations in economic texts},
      volume = {65},
      year = {2014},
    }

    ```




#### FinancialPhrasebankClassification.v2

Polar sentiment dataset of sentences from financial news, categorized by sentiment into positive, negative, or neutral.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/financial_phrasebank`](https://huggingface.co/datasets/mteb/financial_phrasebank) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://arxiv.org/abs/1307.5336)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Financial, News, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{Malo2014GoodDO,
      author = {P. Malo and A. Sinha and P. Korhonen and J. Wallenius and P. Takala},
      journal = {Journal of the Association for Information Science and Technology},
      title = {Good debt or bad debt: Detecting semantic orientations in economic texts},
      volume = {65},
      year = {2014},
    }

    ```




#### FrenchBookReviews

It is a French book reviews dataset containing a huge number of reader reviews on French books. Each review is pared with a rating that ranges from 0.5 to 5 (with 0.5 increment).

**Dataset:** [`Abirate/french_book_reviews`](https://huggingface.co/datasets/Abirate/french_book_reviews) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/Abirate/french_book_reviews)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fra | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex


    ```




#### FrenchBookReviews.v2

It is a French book reviews dataset containing a huge number of reader reviews on French books. Each review is pared with a rating that ranges from 0.5 to 5 (with 0.5 increment).
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/french_book_reviews`](https://huggingface.co/datasets/mteb/french_book_reviews) • **License:** cc0-1.0 • [Learn more →](https://huggingface.co/datasets/Abirate/french_book_reviews)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fra | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex


    ```




#### FrenkEnClassification

English subset of the FRENK dataset

**Dataset:** [`mteb/FrenkEnClassification`](https://huggingface.co/datasets/mteb/FrenkEnClassification) • **License:** not specified • [Learn more →](https://arxiv.org/abs/1906.02045)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{ljubešić2019frenk,
      archiveprefix = {arXiv},
      author = {Nikola Ljubešić and Darja Fišer and Tomaž Erjavec},
      eprint = {1906.02045},
      primaryclass = {cs.CL},
      title = {The FRENK Datasets of Socially Unacceptable Discourse in Slovene and English},
      url = {https://arxiv.org/abs/1906.02045},
      year = {2019},
    }

    ```




#### FrenkEnClassification.v2

English subset of the FRENK dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/frenk_en`](https://huggingface.co/datasets/mteb/frenk_en) • **License:** not specified • [Learn more →](https://arxiv.org/abs/1906.02045)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{ljubešić2019frenk,
      archiveprefix = {arXiv},
      author = {Nikola Ljubešić and Darja Fišer and Tomaž Erjavec},
      eprint = {1906.02045},
      primaryclass = {cs.CL},
      title = {The FRENK Datasets of Socially Unacceptable Discourse in Slovene and English},
      url = {https://arxiv.org/abs/1906.02045},
      year = {2019},
    }

    ```




#### FrenkHrClassification

Croatian subset of the FRENK dataset

**Dataset:** [`mteb/FrenkHrClassification`](https://huggingface.co/datasets/mteb/FrenkHrClassification) • **License:** not specified • [Learn more →](https://arxiv.org/abs/1906.02045)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | hrv | Social, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{ljubešić2019frenk,
      archiveprefix = {arXiv},
      author = {Nikola Ljubešić and Darja Fišer and Tomaž Erjavec},
      eprint = {1906.02045},
      primaryclass = {cs.CL},
      title = {The FRENK Datasets of Socially Unacceptable Discourse in Slovene and English},
      url = {https://arxiv.org/abs/1906.02045},
      year = {2019},
    }

    ```




#### FrenkHrClassification.v2

Croatian subset of the FRENK dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/frenk_hr`](https://huggingface.co/datasets/mteb/frenk_hr) • **License:** not specified • [Learn more →](https://arxiv.org/abs/1906.02045)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | hrv | Social, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{ljubešić2019frenk,
      archiveprefix = {arXiv},
      author = {Nikola Ljubešić and Darja Fišer and Tomaž Erjavec},
      eprint = {1906.02045},
      primaryclass = {cs.CL},
      title = {The FRENK Datasets of Socially Unacceptable Discourse in Slovene and English},
      url = {https://arxiv.org/abs/1906.02045},
      year = {2019},
    }

    ```




#### FrenkSlClassification

Slovenian subset of the FRENK dataset. Also available on HuggingFace dataset hub: English subset, Croatian subset.

**Dataset:** [`mteb/FrenkSlClassification`](https://huggingface.co/datasets/mteb/FrenkSlClassification) • **License:** not specified • [Learn more →](https://arxiv.org/pdf/1906.02045)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | slv | Social, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{ljubešić2019frenk,
      archiveprefix = {arXiv},
      author = {Nikola Ljubešić and Darja Fišer and Tomaž Erjavec},
      eprint = {1906.02045},
      primaryclass = {cs.CL},
      title = {The FRENK Datasets of Socially Unacceptable Discourse in Slovene and English},
      url = {https://arxiv.org/abs/1906.02045},
      year = {2019},
    }

    ```




#### FrenkSlClassification.v2

Slovenian subset of the FRENK dataset. Also available on HuggingFace dataset hub: English subset, Croatian subset.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/frenk_sl`](https://huggingface.co/datasets/mteb/frenk_sl) • **License:** not specified • [Learn more →](https://arxiv.org/pdf/1906.02045)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | slv | Social, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{ljubešić2019frenk,
      archiveprefix = {arXiv},
      author = {Nikola Ljubešić and Darja Fišer and Tomaž Erjavec},
      eprint = {1906.02045},
      primaryclass = {cs.CL},
      title = {The FRENK Datasets of Socially Unacceptable Discourse in Slovene and English},
      url = {https://arxiv.org/abs/1906.02045},
      year = {2019},
    }

    ```




#### FunctionOfDecisionSectionLegalBenchClassification

The task is to classify a paragraph extracted from a written court decision into one of seven possible categories:
            1. Facts - The paragraph describes the faction background that led up to the present lawsuit.
            2. Procedural History - The paragraph describes the course of litigation that led to the current proceeding before the court.
            3. Issue - The paragraph describes the legal or factual issue that must be resolved by the court.
            4. Rule - The paragraph describes a rule of law relevant to resolving the issue.
            5. Analysis - The paragraph analyzes the legal issue by applying the relevant legal principles to the facts of the present dispute.
            6. Conclusion - The paragraph presents a conclusion of the court.
            7. Decree - The paragraph constitutes a decree resolving the dispute.


**Dataset:** [`mteb/FunctionOfDecisionSectionLegalBenchClassification`](https://huggingface.co/datasets/mteb/FunctionOfDecisionSectionLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### GeoreviewClassification

Review classification (5-point scale) based on Yandex Georeview dataset

**Dataset:** [`mteb/GeoreviewClassification`](https://huggingface.co/datasets/mteb/GeoreviewClassification) • **License:** mit • [Learn more →](https://github.com/yandex/geo-reviews-dataset-2023)

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



??? quote "Citation"


    ```bibtex

    @inproceedings{stefanovitch-etal-2022-resources,
      abstract = {This paper presents, to the best of our knowledge, the first ever publicly available annotated dataset for sentiment classification and semantic polarity dictionary for Georgian. The characteristics of these resources and the process of their creation are described in detail. The results of various experiments on the performance of both lexicon- and machine learning-based models for Georgian sentiment classification are also reported. Both 3-label (positive, neutral, negative) and 4-label settings (same labels + mixed) are considered. The machine learning models explored include, i.a., logistic regression, SVMs, and transformed-based models. We also explore transfer learning- and translation-based (to a well-supported language) approaches. The obtained results for Georgian are on par with the state-of-the-art results in sentiment classification for well studied languages when using training data of comparable size.},
      address = {Marseille, France},
      author = {Stefanovitch, Nicolas  and
    Piskorski, Jakub  and
    Kharazi, Sopho},
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
      pages = {1613--1621},
      publisher = {European Language Resources Association},
      title = {Resources and Experiments on Sentiment Classification for {G}eorgian},
      url = {https://aclanthology.org/2022.lrec-1.173},
      year = {2022},
    }

    ```




#### GermanPoliticiansTwitterSentimentClassification

GermanPoliticiansTwitterSentiment is a dataset of German tweets categorized with their sentiment (3 classes).

**Dataset:** [`Alienmaster/german_politicians_twitter_sentiment`](https://huggingface.co/datasets/Alienmaster/german_politicians_twitter_sentiment) • **License:** not specified • [Learn more →](https://aclanthology.org/2022.konvens-1.9)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu | Government, Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{schmidt-etal-2022-sentiment,
      address = {Potsdam, Germany},
      author = {Schmidt, Thomas  and
    Fehle, Jakob  and
    Weissenbacher, Maximilian  and
    Richter, Jonathan  and
    Gottschalk, Philipp  and
    Wolff, Christian},
      booktitle = {Proceedings of the 18th Conference on Natural Language Processing (KONVENS 2022)},
      editor = {Schaefer, Robin  and
    Bai, Xiaoyu  and
    Stede, Manfred  and
    Zesch, Torsten},
      month = {12--15 } # sep,
      pages = {74--87},
      publisher = {KONVENS 2022 Organizers},
      title = {Sentiment Analysis on {T}witter for the Major {G}erman Parties during the 2021 {G}erman Federal Election},
      url = {https://aclanthology.org/2022.konvens-1.9},
      year = {2022},
    }

    ```




#### GermanPoliticiansTwitterSentimentClassification.v2

GermanPoliticiansTwitterSentiment is a dataset of German tweets categorized with their sentiment (3 classes).
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/german_politicians_twitter_sentiment`](https://huggingface.co/datasets/mteb/german_politicians_twitter_sentiment) • **License:** not specified • [Learn more →](https://aclanthology.org/2022.konvens-1.9)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu | Government, Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{schmidt-etal-2022-sentiment,
      address = {Potsdam, Germany},
      author = {Schmidt, Thomas  and
    Fehle, Jakob  and
    Weissenbacher, Maximilian  and
    Richter, Jonathan  and
    Gottschalk, Philipp  and
    Wolff, Christian},
      booktitle = {Proceedings of the 18th Conference on Natural Language Processing (KONVENS 2022)},
      editor = {Schaefer, Robin  and
    Bai, Xiaoyu  and
    Stede, Manfred  and
    Zesch, Torsten},
      month = {12--15 } # sep,
      pages = {74--87},
      publisher = {KONVENS 2022 Organizers},
      title = {Sentiment Analysis on {T}witter for the Major {G}erman Parties during the 2021 {G}erman Federal Election},
      url = {https://aclanthology.org/2022.konvens-1.9},
      year = {2022},
    }

    ```




#### GreekLegalCodeClassification

Greek Legal Code Dataset for Classification. (subset = chapter)

**Dataset:** [`AI-team-UoA/greek_legal_code`](https://huggingface.co/datasets/AI-team-UoA/greek_legal_code) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2109.15298)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ell | Legal, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{papaloukas-etal-2021-glc,
      address = {Punta Cana, Dominican Republic},
      author = {Papaloukas, Christos and Chalkidis, Ilias and Athinaios, Konstantinos and Pantazi, Despina-Athanasia and Koubarakis, Manolis},
      booktitle = {Proceedings of the Natural Legal Language Processing Workshop 2021},
      doi = {10.48550/arXiv.2109.15298},
      pages = {63--75},
      publisher = {Association for Computational Linguistics},
      title = {Multi-granular Legal Topic Classification on Greek Legislation},
      url = {https://arxiv.org/abs/2109.15298},
      year = {2021},
    }

    ```




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



#### HUMEEmotionClassification

Human evaluation subset of Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise.

**Dataset:** [`mteb/HUMEEmotionClassification`](https://huggingface.co/datasets/mteb/HUMEEmotionClassification) • **License:** not specified • [Learn more →](https://www.aclweb.org/anthology/D18-1404)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | eng | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{saravia-etal-2018-carer,
      abstract = {Emotions are expressed in nuanced ways, which varies by collective or individual experiences, knowledge, and beliefs. Therefore, to understand emotion, as conveyed through text, a robust mechanism capable of capturing and modeling different linguistic nuances and phenomena is needed. We propose a semi-supervised, graph-based algorithm to produce rich structural descriptors which serve as the building blocks for constructing contextualized affect representations from text. The pattern-based representations are further enriched with word embeddings and evaluated through several emotion recognition tasks. Our experimental results demonstrate that the proposed method outperforms state-of-the-art techniques on emotion recognition tasks.},
      address = {Brussels, Belgium},
      author = {Saravia, Elvis  and
    Liu, Hsien-Chi Toby  and
    Huang, Yen-Hao  and
    Wu, Junlin  and
    Chen, Yi-Shin},
      booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
      doi = {10.18653/v1/D18-1404},
      editor = {Riloff, Ellen  and
    Chiang, David  and
    Hockenmaier, Julia  and
    Tsujii, Jun{'}ichi},
      month = oct # {-} # nov,
      pages = {3687--3697},
      publisher = {Association for Computational Linguistics},
      title = {{CARER}: Contextualized Affect Representations for Emotion Recognition},
      url = {https://aclanthology.org/D18-1404},
      year = {2018},
    }

    ```




#### HUMEMultilingualSentimentClassification

Human evaluation subset of Sentiment classification dataset with binary (positive vs negative sentiment) labels. Includes 4 languages.

**Dataset:** [`mteb/HUMEMultilingualSentimentClassification`](https://huggingface.co/datasets/mteb/HUMEMultilingualSentimentClassification) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb/multilingual-sentiment-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | ara, eng, nob, rus | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{mollanorozy-etal-2023-cross,
      address = {Dubrovnik, Croatia},
      author = {Mollanorozy, Sepideh  and
    Tanti, Marc  and
    Nissim, Malvina},
      booktitle = {Proceedings of the 5th Workshop on Research in Computational Linguistic Typology and Multilingual NLP},
      doi = {10.18653/v1/2023.sigtyp-1.9},
      editor = {Beinborn, Lisa  and
    Goswami, Koustava  and
    Murado{\\u{g}}lu, Saliha  and
    Sorokin, Alexey  and
    Kumar, Ritesh  and
    Shcherbakov, Andreas  and
    Ponti, Edoardo M.  and
    Cotterell, Ryan  and
    Vylomova, Ekaterina},
      month = may,
      pages = {89--95},
      publisher = {Association for Computational Linguistics},
      title = {Cross-lingual Transfer Learning with \{P\}ersian},
      url = {https://aclanthology.org/2023.sigtyp-1.9},
      year = {2023},
    }

    ```




#### HUMEToxicConversationsClassification

Human evaluation subset of Collection of comments from the Civil Comments platform together with annotations if the comment is toxic or not.

**Dataset:** [`mteb/HUMEToxicConversationsClassification`](https://huggingface.co/datasets/mteb/HUMEToxicConversationsClassification) • **License:** cc-by-4.0 • [Learn more →](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | eng | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{jigsaw-unintended-bias-in-toxicity-classification,
      author = {cjadams and Daniel Borkan and inversion and Jeffrey Sorensen and Lucas Dixon and Lucy Vasserman and nithum},
      publisher = {Kaggle},
      title = {Jigsaw Unintended Bias in Toxicity Classification},
      url = {https://kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification},
      year = {2019},
    }

    ```




#### HUMETweetSentimentExtractionClassification

Human evaluation subset of Tweet Sentiment Extraction dataset.

**Dataset:** [`mteb/HUMETweetSentimentExtractionClassification`](https://huggingface.co/datasets/mteb/HUMETweetSentimentExtractionClassification) • **License:** not specified • [Learn more →](https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | eng | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{tweet-sentiment-extraction,
      author = {Maggie, Phil Culliton, Wei Chen},
      publisher = {Kaggle},
      title = {Tweet Sentiment Extraction},
      url = {https://kaggle.com/competitions/tweet-sentiment-extraction},
      year = {2020},
    }

    ```




#### HateSpeechPortugueseClassification

HateSpeechPortugueseClassification is a dataset of Portuguese tweets categorized with their sentiment (2 classes).

**Dataset:** [`mteb/HateSpeechPortugueseClassification`](https://huggingface.co/datasets/mteb/HateSpeechPortugueseClassification) • **License:** not specified • [Learn more →](https://aclanthology.org/W19-3510)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | por | Social, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{fortuna-etal-2019-hierarchically,
      address = {Florence, Italy},
      author = {Fortuna, Paula  and
    Rocha da Silva, Jo{\~a}o  and
    Soler-Company, Juan  and
    Wanner, Leo  and
    Nunes, S{\'e}rgio},
      booktitle = {Proceedings of the Third Workshop on Abusive Language Online},
      doi = {10.18653/v1/W19-3510},
      editor = {Roberts, Sarah T.  and
    Tetreault, Joel  and
    Prabhakaran, Vinodkumar  and
    Waseem, Zeerak},
      month = aug,
      pages = {94--104},
      publisher = {Association for Computational Linguistics},
      title = {A Hierarchically-Labeled {P}ortuguese Hate Speech Dataset},
      url = {https://aclanthology.org/W19-3510},
      year = {2019},
    }

    ```




#### HeadlineClassification

Headline rubric classification based on the paraphraser plus dataset.

**Dataset:** [`ai-forever/headline-classification`](https://huggingface.co/datasets/ai-forever/headline-classification) • **License:** mit • [Learn more →](https://aclanthology.org/2020.ngt-1.6/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{gudkov-etal-2020-automatically,
      abstract = {The article is focused on automatic development and ranking of a large corpus for Russian paraphrase generation which proves to be the first corpus of such type in Russian computational linguistics. Existing manually annotated paraphrase datasets for Russian are limited to small-sized ParaPhraser corpus and ParaPlag which are suitable for a set of NLP tasks, such as paraphrase and plagiarism detection, sentence similarity and relatedness estimation, etc. Due to size restrictions, these datasets can hardly be applied in end-to-end text generation solutions. Meanwhile, paraphrase generation requires a large amount of training data. In our study we propose a solution to the problem: we collect, rank and evaluate a new publicly available headline paraphrase corpus (ParaPhraser Plus), and then perform text generation experiments with manual evaluation on automatically ranked corpora using the Universal Transformer architecture.},
      address = {Online},
      author = {Gudkov, Vadim  and
    Mitrofanova, Olga  and
    Filippskikh, Elizaveta},
      booktitle = {Proceedings of the Fourth Workshop on Neural Generation and Translation},
      doi = {10.18653/v1/2020.ngt-1.6},
      editor = {Birch, Alexandra  and
    Finch, Andrew  and
    Hayashi, Hiroaki  and
    Heafield, Kenneth  and
    Junczys-Dowmunt, Marcin  and
    Konstas, Ioannis  and
    Li, Xian  and
    Neubig, Graham  and
    Oda, Yusuke},
      month = jul,
      pages = {54--59},
      publisher = {Association for Computational Linguistics},
      title = {Automatically Ranked {R}ussian Paraphrase Corpus for Text Generation},
      url = {https://aclanthology.org/2020.ngt-1.6},
      year = {2020},
    }

    ```




#### HeadlineClassification.v2

Headline rubric classification based on the paraphraser plus dataset.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/headline`](https://huggingface.co/datasets/mteb/headline) • **License:** mit • [Learn more →](https://aclanthology.org/2020.ngt-1.6/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{gudkov-etal-2020-automatically,
      abstract = {The article is focused on automatic development and ranking of a large corpus for Russian paraphrase generation which proves to be the first corpus of such type in Russian computational linguistics. Existing manually annotated paraphrase datasets for Russian are limited to small-sized ParaPhraser corpus and ParaPlag which are suitable for a set of NLP tasks, such as paraphrase and plagiarism detection, sentence similarity and relatedness estimation, etc. Due to size restrictions, these datasets can hardly be applied in end-to-end text generation solutions. Meanwhile, paraphrase generation requires a large amount of training data. In our study we propose a solution to the problem: we collect, rank and evaluate a new publicly available headline paraphrase corpus (ParaPhraser Plus), and then perform text generation experiments with manual evaluation on automatically ranked corpora using the Universal Transformer architecture.},
      address = {Online},
      author = {Gudkov, Vadim  and
    Mitrofanova, Olga  and
    Filippskikh, Elizaveta},
      booktitle = {Proceedings of the Fourth Workshop on Neural Generation and Translation},
      doi = {10.18653/v1/2020.ngt-1.6},
      editor = {Birch, Alexandra  and
    Finch, Andrew  and
    Hayashi, Hiroaki  and
    Heafield, Kenneth  and
    Junczys-Dowmunt, Marcin  and
    Konstas, Ioannis  and
    Li, Xian  and
    Neubig, Graham  and
    Oda, Yusuke},
      month = jul,
      pages = {54--59},
      publisher = {Association for Computational Linguistics},
      title = {Automatically Ranked {R}ussian Paraphrase Corpus for Text Generation},
      url = {https://aclanthology.org/2020.ngt-1.6},
      year = {2020},
    }

    ```




#### HebrewSentimentAnalysis

HebrewSentiment is a data set consists of 12,804 user comments to posts on the official Facebook page of Israel’s president, Mr. Reuven Rivlin. In October 2015, we used the open software application Netvizz (Rieder, 2013) to scrape all the comments to all of the president’s posts in the period of June – August 2014, the first three months of Rivlin’s presidency.2 While the president’s posts aimed at reconciling tensions and called for tolerance and empathy, the sentiment expressed in the comments to the president’s posts was polarized between citizens who warmly thanked the president, and citizens that fiercely critiqued his policy.

**Dataset:** [`mteb/HebrewSentimentAnalysis`](https://huggingface.co/datasets/mteb/HebrewSentimentAnalysis) • **License:** mit • [Learn more →](https://huggingface.co/datasets/hebrew_sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | heb | Reviews, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{amram-etal-2018-representations,
      address = {Santa Fe, New Mexico, USA},
      author = {Amram, Adam and Ben David, Anat and Tsarfaty, Reut},
      booktitle = {Proceedings of the 27th International Conference on Computational Linguistics},
      month = aug,
      pages = {2242--2252},
      publisher = {Association for Computational Linguistics},
      title = {Representations and Architectures in Neural Sentiment Analysis for Morphologically Rich Languages: A Case Study from {M}odern {H}ebrew},
      url = {https://www.aclweb.org/anthology/C18-1190},
      year = {2018},
    }

    ```




#### HebrewSentimentAnalysis.v2

HebrewSentiment is a data set consists of 12,804 user comments to posts on the official Facebook page of Israel’s president, Mr. Reuven Rivlin. In October 2015, we used the open software application Netvizz (Rieder, 2013) to scrape all the comments to all of the president’s posts in the period of June – August 2014, the first three months of Rivlin’s presidency.2 While the president’s posts aimed at reconciling tensions and called for tolerance and empathy, the sentiment expressed in the comments to the president’s posts was polarized between citizens who warmly thanked the president, and citizens that fiercely critiqued his policy.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/hebrew_sentiment_analysis`](https://huggingface.co/datasets/mteb/hebrew_sentiment_analysis) • **License:** mit • [Learn more →](https://huggingface.co/datasets/hebrew_sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | heb | Reviews, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{amram-etal-2018-representations,
      address = {Santa Fe, New Mexico, USA},
      author = {Amram, Adam and Ben David, Anat and Tsarfaty, Reut},
      booktitle = {Proceedings of the 27th International Conference on Computational Linguistics},
      month = aug,
      pages = {2242--2252},
      publisher = {Association for Computational Linguistics},
      title = {Representations and Architectures in Neural Sentiment Analysis for Morphologically Rich Languages: A Case Study from {M}odern {H}ebrew},
      url = {https://www.aclweb.org/anthology/C18-1190},
      year = {2018},
    }

    ```




#### HinDialectClassification

HinDialect: 26 Hindi-related languages and dialects of the Indic Continuum in North India

**Dataset:** [`mlexplorer008/hin_dialect_classification`](https://huggingface.co/datasets/mlexplorer008/hin_dialect_classification) • **License:** cc-by-sa-4.0 • [Learn more →](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4839)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | anp, awa, ben, bgc, bhb, ... (21) | Social, Spoken, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{11234/1-4839,
      author = {Bafna, Niyati and {\v Z}abokrtsk{\'y}, Zden{\v e}k and Espa{\~n}a-Bonet, Cristina and van Genabith, Josef and Kumar, Lalit "Samyak Lalit" and Suman, Sharda and Shivay, Rahul},
      copyright = {Creative Commons - Attribution-{NonCommercial}-{ShareAlike} 4.0 International ({CC} {BY}-{NC}-{SA} 4.0)},
      note = {{LINDAT}/{CLARIAH}-{CZ} digital library at the Institute of Formal and Applied Linguistics ({{\'U}FAL}), Faculty of Mathematics and Physics, Charles University},
      title = {{HinDialect} 1.1: 26 Hindi-related languages and dialects of the Indic Continuum in North India},
      url = {http://hdl.handle.net/11234/1-4839},
      year = {2022},
    }

    ```




#### HindiDiscourseClassification

A Hindi Discourse dataset in Hindi with values for coherence.

**Dataset:** [`mteb/HindiDiscourseClassification`](https://huggingface.co/datasets/mteb/HindiDiscourseClassification) • **License:** mit • [Learn more →](https://aclanthology.org/2020.lrec-1.149/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | hin | Fiction, Social, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{dhanwal-etal-2020-annotated,
      address = {Marseille, France},
      author = {Dhanwal, Swapnil  and
    Dutta, Hritwik  and
    Nankani, Hitesh  and
    Shrivastava, Nilay  and
    Kumar, Yaman  and
    Li, Junyi Jessy  and
    Mahata, Debanjan  and
    Gosangi, Rakesh  and
    Zhang, Haimin  and
    Shah, Rajiv Ratn  and
    Stent, Amanda},
      booktitle = {Proceedings of the 12th Language Resources and Evaluation Conference},
      isbn = {979-10-95546-34-4},
      language = {English},
      month = may,
      publisher = {European Language Resources Association},
      title = {An Annotated Dataset of Discourse Modes in {H}indi Stories},
      url = {https://www.aclweb.org/anthology/2020.lrec-1.149},
      year = {2020},
    }

    ```




#### HindiDiscourseClassification.v2

A Hindi Discourse dataset in Hindi with values for coherence.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/hindi_discourse`](https://huggingface.co/datasets/mteb/hindi_discourse) • **License:** mit • [Learn more →](https://aclanthology.org/2020.lrec-1.149/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | hin | Fiction, Social, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{dhanwal-etal-2020-annotated,
      address = {Marseille, France},
      author = {Dhanwal, Swapnil  and
    Dutta, Hritwik  and
    Nankani, Hitesh  and
    Shrivastava, Nilay  and
    Kumar, Yaman  and
    Li, Junyi Jessy  and
    Mahata, Debanjan  and
    Gosangi, Rakesh  and
    Zhang, Haimin  and
    Shah, Rajiv Ratn  and
    Stent, Amanda},
      booktitle = {Proceedings of the 12th Language Resources and Evaluation Conference},
      isbn = {979-10-95546-34-4},
      language = {English},
      month = may,
      publisher = {European Language Resources Association},
      title = {An Annotated Dataset of Discourse Modes in {H}indi Stories},
      url = {https://www.aclweb.org/anthology/2020.lrec-1.149},
      year = {2020},
    }

    ```




#### HotelReviewSentimentClassification

HARD is a dataset of Arabic hotel reviews collected from the Booking.com website.

**Dataset:** [`mteb/HotelReviewSentimentClassification`](https://huggingface.co/datasets/mteb/HotelReviewSentimentClassification) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-67056-0_3)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{elnagar2018hotel,
      author = {Elnagar, Ashraf and Khalifa, Yasmin S and Einea, Anas},
      journal = {Intelligent natural language processing: Trends and applications},
      pages = {35--52},
      publisher = {Springer},
      title = {Hotel Arabic-reviews dataset construction for sentiment analysis applications},
      year = {2018},
    }

    ```




#### HotelReviewSentimentClassification.v2

HARD is a dataset of Arabic hotel reviews collected from the Booking.com website.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/HotelReviewSentimentClassification`](https://huggingface.co/datasets/mteb/HotelReviewSentimentClassification) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-67056-0_3)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{elnagar2018hotel,
      author = {Elnagar, Ashraf and Khalifa, Yasmin S and Einea, Anas},
      journal = {Intelligent natural language processing: Trends and applications},
      pages = {35--52},
      publisher = {Springer},
      title = {Hotel Arabic-reviews dataset construction for sentiment analysis applications},
      year = {2018},
    }

    ```




#### IFlyTek

Long Text classification for the description of Apps

**Dataset:** [`C-MTEB/IFlyTek-classification`](https://huggingface.co/datasets/C-MTEB/IFlyTek-classification) • **License:** not specified • [Learn more →](https://www.cluebenchmarks.com/introduce.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @inproceedings{xu-etal-2020-clue,
      abstract = {The advent of natural language understanding (NLU) benchmarks for English, such as GLUE and SuperGLUE allows new NLU models to be evaluated across a diverse set of tasks. These comprehensive benchmarks have facilitated a broad range of research and applications in natural language processing (NLP). The problem, however, is that most such benchmarks are limited to English, which has made it difficult to replicate many of the successes in English NLU for other languages. To help remedy this issue, we introduce the first large-scale Chinese Language Understanding Evaluation (CLUE) benchmark. CLUE is an open-ended, community-driven project that brings together 9 tasks spanning several well-established single-sentence/sentence-pair classification tasks, as well as machine reading comprehension, all on original Chinese text. To establish results on these tasks, we report scores using an exhaustive set of current state-of-the-art pre-trained Chinese models (9 in total). We also introduce a number of supplementary datasets and additional tools to help facilitate further progress on Chinese NLU. Our benchmark is released at https://www.cluebenchmarks.com},
      address = {Barcelona, Spain (Online)},
      author = {Xu, Liang  and
    Hu, Hai and
    Zhang, Xuanwei and
    Li, Lu and
    Cao, Chenjie and
    Li, Yudong and
    Xu, Yechen and
    Sun, Kai and
    Yu, Dian and
    Yu, Cong and
    Tian, Yin and
    Dong, Qianqian and
    Liu, Weitang and
    Shi, Bo and
    Cui, Yiming and
    Li, Junyi and
    Zeng, Jun and
    Wang, Rongzhao and
    Xie, Weijian and
    Li, Yanting and
    Patterson, Yina and
    Tian, Zuoyu and
    Zhang, Yiwen and
    Zhou, He and
    Liu, Shaoweihua and
    Zhao, Zhe and
    Zhao, Qipeng and
    Yue, Cong and
    Zhang, Xinrui and
    Yang, Zhengliang and
    Richardson, Kyle and
    Lan, Zhenzhong },
      booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
      doi = {10.18653/v1/2020.coling-main.419},
      month = dec,
      pages = {4762--4772},
      publisher = {International Committee on Computational Linguistics},
      title = {{CLUE}: A {C}hinese Language Understanding Evaluation Benchmark},
      url = {https://aclanthology.org/2020.coling-main.419},
      year = {2020},
    }

    ```




#### IFlyTek.v2

Long Text classification for the description of Apps
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/i_fly_tek`](https://huggingface.co/datasets/mteb/i_fly_tek) • **License:** not specified • [Learn more →](https://www.cluebenchmarks.com/introduce.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @inproceedings{xu-etal-2020-clue,
      abstract = {The advent of natural language understanding (NLU) benchmarks for English, such as GLUE and SuperGLUE allows new NLU models to be evaluated across a diverse set of tasks. These comprehensive benchmarks have facilitated a broad range of research and applications in natural language processing (NLP). The problem, however, is that most such benchmarks are limited to English, which has made it difficult to replicate many of the successes in English NLU for other languages. To help remedy this issue, we introduce the first large-scale Chinese Language Understanding Evaluation (CLUE) benchmark. CLUE is an open-ended, community-driven project that brings together 9 tasks spanning several well-established single-sentence/sentence-pair classification tasks, as well as machine reading comprehension, all on original Chinese text. To establish results on these tasks, we report scores using an exhaustive set of current state-of-the-art pre-trained Chinese models (9 in total). We also introduce a number of supplementary datasets and additional tools to help facilitate further progress on Chinese NLU. Our benchmark is released at https://www.cluebenchmarks.com},
      address = {Barcelona, Spain (Online)},
      author = {Xu, Liang  and
    Hu, Hai and
    Zhang, Xuanwei and
    Li, Lu and
    Cao, Chenjie and
    Li, Yudong and
    Xu, Yechen and
    Sun, Kai and
    Yu, Dian and
    Yu, Cong and
    Tian, Yin and
    Dong, Qianqian and
    Liu, Weitang and
    Shi, Bo and
    Cui, Yiming and
    Li, Junyi and
    Zeng, Jun and
    Wang, Rongzhao and
    Xie, Weijian and
    Li, Yanting and
    Patterson, Yina and
    Tian, Zuoyu and
    Zhang, Yiwen and
    Zhou, He and
    Liu, Shaoweihua and
    Zhao, Zhe and
    Zhao, Qipeng and
    Yue, Cong and
    Zhang, Xinrui and
    Yang, Zhengliang and
    Richardson, Kyle and
    Lan, Zhenzhong },
      booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
      doi = {10.18653/v1/2020.coling-main.419},
      month = dec,
      pages = {4762--4772},
      publisher = {International Committee on Computational Linguistics},
      title = {{CLUE}: A {C}hinese Language Understanding Evaluation Benchmark},
      url = {https://aclanthology.org/2020.coling-main.419},
      year = {2020},
    }

    ```




#### ImdbClassification

Large Movie Review Dataset

**Dataset:** [`mteb/imdb`](https://huggingface.co/datasets/mteb/imdb) • **License:** not specified • [Learn more →](http://www.aclweb.org/anthology/P11-1015)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{maas-etal-2011-learning,
      address = {Portland, Oregon, USA},
      author = {Maas, Andrew L.  and
    Daly, Raymond E.  and
    Pham, Peter T.  and
    Huang, Dan  and
    Ng, Andrew Y.  and
    Potts, Christopher},
      booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
      editor = {Lin, Dekang  and
    Matsumoto, Yuji  and
    Mihalcea, Rada},
      month = jun,
      pages = {142--150},
      publisher = {Association for Computational Linguistics},
      title = {Learning Word Vectors for Sentiment Analysis},
      url = {https://aclanthology.org/P11-1015},
      year = {2011},
    }

    ```




#### ImdbClassification.v2

Large Movie Review Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/imdb`](https://huggingface.co/datasets/mteb/imdb) • **License:** not specified • [Learn more →](http://www.aclweb.org/anthology/P11-1015)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{maas-etal-2011-learning,
      address = {Portland, Oregon, USA},
      author = {Maas, Andrew L.  and
    Daly, Raymond E.  and
    Pham, Peter T.  and
    Huang, Dan  and
    Ng, Andrew Y.  and
    Potts, Christopher},
      booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
      editor = {Lin, Dekang  and
    Matsumoto, Yuji  and
    Mihalcea, Rada},
      month = jun,
      pages = {142--150},
      publisher = {Association for Computational Linguistics},
      title = {Learning Word Vectors for Sentiment Analysis},
      url = {https://aclanthology.org/P11-1015},
      year = {2011},
    }

    ```




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




#### InappropriatenessClassification

Inappropriateness identification in the form of binary classification

**Dataset:** [`ai-forever/inappropriateness-classification`](https://huggingface.co/datasets/ai-forever/inappropriateness-classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/2021.bsnlp-1.4)

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




#### InappropriatenessClassification.v2

Inappropriateness identification in the form of binary classification
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/inappropriateness`](https://huggingface.co/datasets/mteb/inappropriateness) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/2021.bsnlp-1.4)

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




#### InappropriatenessClassificationv2

Inappropriateness identification in the form of binary classification

**Dataset:** [`mteb/InappropriatenessClassificationv2`](https://huggingface.co/datasets/mteb/InappropriatenessClassificationv2) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://aclanthology.org/2021.bsnlp-1.4)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | rus | Social, Web, Written | human-annotated | found |



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




#### IndicLangClassification

A language identification test set for native-script as well as Romanized text which spans 22 Indic languages.

**Dataset:** [`ai4bharat/Bhasha-Abhijnaanam`](https://huggingface.co/datasets/ai4bharat/Bhasha-Abhijnaanam) • **License:** cc0-1.0 • [Learn more →](https://arxiv.org/abs/2305.15814)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | asm, ben, brx, doi, gom, ... (22) | Non-fiction, Web, Written | expert-annotated | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{madhani-etal-2023-bhasa,
      address = {Toronto, Canada},
      author = {Madhani, Yash  and
    Khapra, Mitesh M.  and
    Kunchukuttan, Anoop},
      booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
      doi = {10.18653/v1/2023.acl-short.71},
      editor = {Rogers, Anna  and
    Boyd-Graber, Jordan  and
    Okazaki, Naoaki},
      month = jul,
      pages = {816--826},
      publisher = {Association for Computational Linguistics},
      title = {Bhasa-Abhijnaanam: Native-script and romanized Language Identification for 22 {I}ndic languages},
      url = {https://aclanthology.org/2023.acl-short.71},
      year = {2023},
    }

    ```




#### IndicNLPNewsClassification

A News classification dataset in multiple Indian regional languages.

**Dataset:** [`Sakshamrzt/IndicNLP-Multilingual`](https://huggingface.co/datasets/Sakshamrzt/IndicNLP-Multilingual) • **License:** cc-by-nc-4.0 • [Learn more →](https://github.com/AI4Bharat/indicnlp_corpus#indicnlp-news-article-classification-dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | guj, kan, mal, mar, ori, ... (8) | News, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{kunchukuttan2020indicnlpcorpus,
      author = {Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
      journal = {arXiv preprint arXiv:2005.00085},
      title = {AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
      year = {2020},
    }

    ```




#### IndicSentimentClassification

A new, multilingual, and n-way parallel dataset for sentiment analysis in 13 Indic languages.

**Dataset:** [`mteb/IndicSentiment`](https://huggingface.co/datasets/mteb/IndicSentiment) • **License:** cc0-1.0 • [Learn more →](https://arxiv.org/abs/2212.05409)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | asm, ben, brx, guj, hin, ... (13) | Reviews, Written | human-annotated | machine-translated and verified |



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




#### IndonesianIdClickbaitClassification

The CLICK-ID dataset is a collection of Indonesian news headlines that was collected from 12 local online news publishers.

**Dataset:** [`manandey/id_clickbait`](https://huggingface.co/datasets/manandey/id_clickbait) • **License:** cc-by-4.0 • [Learn more →](http://www.sciencedirect.com/science/article/pii/S2352340920311252)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ind | News, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{WILLIAM2020106231,
      abstract = {News analysis is a popular task in Natural Language Processing (NLP). In particular, the problem of clickbait in news analysis has gained attention in recent years [1, 2]. However, the majority of the tasks has been focused on English news, in which there is already a rich representative resource. For other languages, such as Indonesian, there is still a lack of resource for clickbait tasks. Therefore, we introduce the CLICK-ID dataset of Indonesian news headlines extracted from 12 Indonesian online news publishers. It is comprised of 15,000 annotated headlines with clickbait and non-clickbait labels. Using the CLICK-ID dataset, we then developed an Indonesian clickbait classification model achieving favourable performance. We believe that this corpus will be useful for replicable experiments in clickbait detection or other experiments in NLP areas.},
      author = {Andika William and Yunita Sari},
      doi = {https://doi.org/10.1016/j.dib.2020.106231},
      issn = {2352-3409},
      journal = {Data in Brief},
      keywords = {Indonesian, Natural Language Processing, News articles, Clickbait, Text-classification},
      pages = {106231},
      title = {CLICK-ID: A novel dataset for Indonesian clickbait headlines},
      url = {http://www.sciencedirect.com/science/article/pii/S2352340920311252},
      volume = {32},
      year = {2020},
    }

    ```




#### IndonesianIdClickbaitClassification.v2

The CLICK-ID dataset is a collection of Indonesian news headlines that was collected from 12 local online news publishers.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/indonesian_id_clickbait`](https://huggingface.co/datasets/mteb/indonesian_id_clickbait) • **License:** cc-by-4.0 • [Learn more →](http://www.sciencedirect.com/science/article/pii/S2352340920311252)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ind | News, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{WILLIAM2020106231,
      abstract = {News analysis is a popular task in Natural Language Processing (NLP). In particular, the problem of clickbait in news analysis has gained attention in recent years [1, 2]. However, the majority of the tasks has been focused on English news, in which there is already a rich representative resource. For other languages, such as Indonesian, there is still a lack of resource for clickbait tasks. Therefore, we introduce the CLICK-ID dataset of Indonesian news headlines extracted from 12 Indonesian online news publishers. It is comprised of 15,000 annotated headlines with clickbait and non-clickbait labels. Using the CLICK-ID dataset, we then developed an Indonesian clickbait classification model achieving favourable performance. We believe that this corpus will be useful for replicable experiments in clickbait detection or other experiments in NLP areas.},
      author = {Andika William and Yunita Sari},
      doi = {https://doi.org/10.1016/j.dib.2020.106231},
      issn = {2352-3409},
      journal = {Data in Brief},
      keywords = {Indonesian, Natural Language Processing, News articles, Clickbait, Text-classification},
      pages = {106231},
      title = {CLICK-ID: A novel dataset for Indonesian clickbait headlines},
      url = {http://www.sciencedirect.com/science/article/pii/S2352340920311252},
      volume = {32},
      year = {2020},
    }

    ```




#### IndonesianMongabayConservationClassification

Conservation dataset that was collected from mongabay.co.id contains topic-classification task (multi-label format) and sentiment classification. This task only covers sentiment analysis (positive, neutral negative)

**Dataset:** [`Datasaur/mongabay-experiment`](https://huggingface.co/datasets/Datasaur/mongabay-experiment) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.sealp-1.4/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ind | Web, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{fransiska-etal-2023-utilizing,
      address = {Nusa Dua, Bali, Indonesia},
      author = {Fransiska, Mega  and
    Pitaloka, Diah  and
    Saripudin, Saripudin  and
    Putra, Satrio  and
    Sutawika*, Lintang},
      booktitle = {Proceedings of the First Workshop in South East Asian Language Processing},
      doi = {10.18653/v1/2023.sealp-1.4},
      editor = {Wijaya, Derry  and
    Aji, Alham Fikri  and
    Vania, Clara  and
    Winata, Genta Indra  and
    Purwarianti, Ayu},
      month = nov,
      pages = {30--54},
      publisher = {Association for Computational Linguistics},
      title = {Utilizing Weak Supervision to Generate {I}ndonesian Conservation Datasets},
      url = {https://aclanthology.org/2023.sealp-1.4},
      year = {2023},
    }

    ```




#### IndonesianMongabayConservationClassification.v2

Conservation dataset that was collected from mongabay.co.id contains topic-classification task (multi-label format) and sentiment classification. This task only covers sentiment analysis (positive, neutral negative)
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/indonesian_mongabay_conservation`](https://huggingface.co/datasets/mteb/indonesian_mongabay_conservation) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.sealp-1.4/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ind | Web, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{fransiska-etal-2023-utilizing,
      address = {Nusa Dua, Bali, Indonesia},
      author = {Fransiska, Mega  and
    Pitaloka, Diah  and
    Saripudin, Saripudin  and
    Putra, Satrio  and
    Sutawika*, Lintang},
      booktitle = {Proceedings of the First Workshop in South East Asian Language Processing},
      doi = {10.18653/v1/2023.sealp-1.4},
      editor = {Wijaya, Derry  and
    Aji, Alham Fikri  and
    Vania, Clara  and
    Winata, Genta Indra  and
    Purwarianti, Ayu},
      month = nov,
      pages = {30--54},
      publisher = {Association for Computational Linguistics},
      title = {Utilizing Weak Supervision to Generate {I}ndonesian Conservation Datasets},
      url = {https://aclanthology.org/2023.sealp-1.4},
      year = {2023},
    }

    ```




#### InsurancePolicyInterpretationLegalBenchClassification

Given an insurance claim and policy, determine whether the claim is covered by the policy.

**Dataset:** [`mteb/InsurancePolicyInterpretationLegalBenchClassification`](https://huggingface.co/datasets/mteb/InsurancePolicyInterpretationLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### InternationalCitizenshipQuestionsLegalBenchClassification

Answer questions about citizenship law from across the world. Dataset was made using the GLOBALCIT citizenship law dataset, by constructing questions about citizenship law as Yes or No questions.

**Dataset:** [`mteb/InternationalCitizenshipQuestionsLegalBenchClassification`](https://huggingface.co/datasets/mteb/InternationalCitizenshipQuestionsLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @misc{vink2023globalcit,
      author = {Vink, Maarten and van der Baaren, Luuk and Bauböck, Rainer and Džankić, Jelena and Honohan, Iseult and Manby, Bronwen},
      howpublished = {https://hdl.handle.net/1814/73190},
      publisher = {Global Citizenship Observatory},
      title = {GLOBALCIT Citizenship Law Dataset, v2.0, Country-Year-Mode Data (Acquisition)},
      year = {2023},
    }

    ```




#### IsiZuluNewsClassification

isiZulu News Classification Dataset

**Dataset:** [`isaacchung/isizulu-news`](https://huggingface.co/datasets/isaacchung/isizulu-news) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/dsfsi/za-isizulu-siswati-news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | zul | News, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{Madodonga_Marivate_Adendorff_2023,
      author = {Madodonga, Andani and Marivate, Vukosi and Adendorff, Matthew},
      doi = {10.55492/dhasa.v4i01.4449},
      month = {Jan.},
      title = {Izindaba-Tindzaba: Machine learning news categorisation for Long and Short Text for isiZulu and Siswati},
      url = {https://upjournals.up.ac.za/index.php/dhasa/article/view/4449},
      volume = {4},
      year = {2023},
    }

    ```




#### IsiZuluNewsClassification.v2

isiZulu News Classification Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/isi_zulu_news`](https://huggingface.co/datasets/mteb/isi_zulu_news) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/dsfsi/za-isizulu-siswati-news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | zul | News, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{Madodonga_Marivate_Adendorff_2023,
      author = {Madodonga, Andani and Marivate, Vukosi and Adendorff, Matthew},
      doi = {10.55492/dhasa.v4i01.4449},
      month = {Jan.},
      title = {Izindaba-Tindzaba: Machine learning news categorisation for Long and Short Text for isiZulu and Siswati},
      url = {https://upjournals.up.ac.za/index.php/dhasa/article/view/4449},
      volume = {4},
      year = {2023},
    }

    ```




#### ItaCaseholdClassification

An Italian Dataset consisting of 1101 pairs of judgments and their official holdings between the years 2019 and 2022 from the archives of Italian Administrative Justice categorized with 64 subjects.

**Dataset:** [`itacasehold/itacasehold`](https://huggingface.co/datasets/itacasehold/itacasehold) • **License:** apache-2.0 • [Learn more →](https://doi.org/10.1145/3594536.3595177)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ita | Government, Legal, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{10.1145/3594536.3595177,
      abstract = {Legal holdings are used in Italy as a critical component of the legal system, serving to establish legal precedents, provide guidance for future legal decisions, and ensure consistency and predictability in the interpretation and application of the law. They are written by domain experts who describe in a clear and concise manner the principle of law applied in the judgments.We introduce a legal holding extraction method based on Italian-LEGAL-BERT to automatically extract legal holdings from Italian cases. In addition, we present ITA-CaseHold, a benchmark dataset for Italian legal summarization. We conducted several experiments using this dataset, as a valuable baseline for future research on this topic.},
      address = {New York, NY, USA},
      author = {Licari, Daniele and Bushipaka, Praveen and Marino, Gabriele and Comand\'{e}, Giovanni and Cucinotta, Tommaso},
      booktitle = {Proceedings of the Nineteenth International Conference on Artificial Intelligence and Law},
      doi = {10.1145/3594536.3595177},
      isbn = {9798400701979},
      keywords = {Italian-LEGAL-BERT, Holding Extraction, Extractive Text Summarization, Benchmark Dataset},
      location = {<conf-loc>, <city>Braga</city>, <country>Portugal</country>, </conf-loc>},
      numpages = {9},
      pages = {148–156},
      publisher = {Association for Computing Machinery},
      series = {ICAIL '23},
      title = {Legal Holding Extraction from Italian Case Documents using Italian-LEGAL-BERT Text Summarization},
      url = {https://doi.org/10.1145/3594536.3595177},
      year = {2023},
    }

    ```




#### Itacola

An Italian Corpus of Linguistic Acceptability taken from linguistic literature with a binary annotation made by the original authors themselves.

**Dataset:** [`mteb/Itacola`](https://huggingface.co/datasets/mteb/Itacola) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.findings-emnlp.250/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ita | Non-fiction, Spoken, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{trotta-etal-2021-monolingual-cross,
      address = {Punta Cana, Dominican Republic},
      author = {Trotta, Daniela  and
    Guarasci, Raffaele  and
    Leonardelli, Elisa  and
    Tonelli, Sara},
      booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2021},
      doi = {10.18653/v1/2021.findings-emnlp.250},
      month = nov,
      pages = {2929--2940},
      publisher = {Association for Computational Linguistics},
      title = {Monolingual and Cross-Lingual Acceptability Judgments with the {I}talian {C}o{LA} corpus},
      url = {https://aclanthology.org/2021.findings-emnlp.250},
      year = {2021},
    }

    ```




#### Itacola.v2

An Italian Corpus of Linguistic Acceptability taken from linguistic literature with a binary annotation made by the original authors themselves.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/italian_linguistic_acceptability`](https://huggingface.co/datasets/mteb/italian_linguistic_acceptability) • **License:** not specified • [Learn more →](https://aclanthology.org/2021.findings-emnlp.250/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ita | Non-fiction, Spoken, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{trotta-etal-2021-monolingual-cross,
      address = {Punta Cana, Dominican Republic},
      author = {Trotta, Daniela  and
    Guarasci, Raffaele  and
    Leonardelli, Elisa  and
    Tonelli, Sara},
      booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2021},
      doi = {10.18653/v1/2021.findings-emnlp.250},
      month = nov,
      pages = {2929--2940},
      publisher = {Association for Computational Linguistics},
      title = {Monolingual and Cross-Lingual Acceptability Judgments with the {I}talian {C}o{LA} corpus},
      url = {https://aclanthology.org/2021.findings-emnlp.250},
      year = {2021},
    }

    ```




#### JCrewBlockerLegalBenchClassification

The J.Crew Blocker, also known as the J.Crew Protection, is a provision included in leveraged loan documents to prevent companies from removing security by transferring intellectual property (IP) into new subsidiaries and raising additional debt. The task consists of detemining whether the J.Crew Blocker is present in the document.

**Dataset:** [`mteb/JCrewBlockerLegalBenchClassification`](https://huggingface.co/datasets/mteb/JCrewBlockerLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### JCrewBlockerLegalBenchClassification.v2

The J.Crew Blocker, also known as the J.Crew Protection, is a provision included in leveraged loan documents to prevent companies from removing security by transferring intellectual property (IP) into new subsidiaries and raising additional debt. The task consists of detemining whether the J.Crew Blocker is present in the document.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/j_crew_blocker_legal_bench`](https://huggingface.co/datasets/mteb/j_crew_blocker_legal_bench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### JDReview

review for iphone

**Dataset:** [`C-MTEB/JDReview-classification`](https://huggingface.co/datasets/C-MTEB/JDReview-classification) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @article{xiao2023c,
      author = {Xiao, Shitao and Liu, Zheng and Zhang, Peitian and Muennighof, Niklas},
      journal = {arXiv preprint arXiv:2309.07597},
      title = {C-pack: Packaged resources to advance general chinese embedding},
      year = {2023},
    }

    ```




#### JDReview.v2

review for iphone
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/jd_review`](https://huggingface.co/datasets/mteb/jd_review) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @article{xiao2023c,
      author = {Xiao, Shitao and Liu, Zheng and Zhang, Peitian and Muennighof, Niklas},
      journal = {arXiv preprint arXiv:2309.07597},
      title = {C-pack: Packaged resources to advance general chinese embedding},
      year = {2023},
    }

    ```




#### JapaneseSentimentClassification

Japanese sentiment classification dataset with binary
(positive vs negative sentiment) labels. This version reverts
the morphological analysis from the original multilingual dataset
to restore natural Japanese text without artificial spaces.


**Dataset:** [`mteb/JapaneseSentimentClassification`](https://huggingface.co/datasets/mteb/JapaneseSentimentClassification) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb/multilingual-sentiment-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | jpn | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{mollanorozy-etal-2023-cross,
      address = {Dubrovnik, Croatia},
      author = {Mollanorozy, Sepideh  and
    Tanti, Marc  and
    Nissim, Malvina},
      booktitle = {Proceedings of the 5th Workshop on Research in Computational Linguistic Typology and Multilingual NLP},
      doi = {10.18653/v1/2023.sigtyp-1.9},
      editor = {Beinborn, Lisa  and
    Goswami, Koustava  and
    Murado{\\u{g}}lu, Saliha  and
    Sorokin, Alexey  and
    Shcherbakov, Andreas  and
    Ponti, Edoardo M.  and
    Cotterell, Ryan  and
    Vylomova, Ekaterina},
      month = may,
      pages = {89--95},
      publisher = {Association for Computational Linguistics},
      title = {Cross-lingual Transfer Learning with \{P\}ersian},
      url = {https://aclanthology.org/2023.sigtyp-1.9},
      year = {2023},
    }

    ```




#### JavaneseIMDBClassification

Large Movie Review Dataset translated to Javanese. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets.

**Dataset:** [`mteb/JavaneseIMDBClassification`](https://huggingface.co/datasets/mteb/JavaneseIMDBClassification) • **License:** mit • [Learn more →](https://github.com/w11wo/nlp-datasets#javanese-imdb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | jav | Reviews, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{wongso2021causal,
      author = {Wongso, Wilson and Setiawan, David Samuel and Suhartono, Derwin},
      booktitle = {2021 International Conference on Advanced Computer Science and Information Systems (ICACSIS)},
      organization = {IEEE},
      pages = {1--7},
      title = {Causal and Masked Language Modeling of Javanese Language using Transformer-based Architectures},
      year = {2021},
    }

    ```




#### JavaneseIMDBClassification.v2

Large Movie Review Dataset translated to Javanese. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/javanese_imdb`](https://huggingface.co/datasets/mteb/javanese_imdb) • **License:** mit • [Learn more →](https://github.com/w11wo/nlp-datasets#javanese-imdb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | jav | Reviews, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{wongso2021causal,
      author = {Wongso, Wilson and Setiawan, David Samuel and Suhartono, Derwin},
      booktitle = {2021 International Conference on Advanced Computer Science and Information Systems (ICACSIS)},
      organization = {IEEE},
      pages = {1--7},
      title = {Causal and Masked Language Modeling of Javanese Language using Transformer-based Architectures},
      year = {2021},
    }

    ```




#### KLUE-TC

Topic classification dataset of human-annotated news headlines. Part of the Korean Language Understanding Evaluation (KLUE).

**Dataset:** [`klue/klue`](https://huggingface.co/datasets/klue/klue) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2105.09680)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | News, Written | human-annotated | found |



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




#### KLUE-TC.v2

Topic classification dataset of human-annotated news headlines. Part of the Korean Language Understanding Evaluation (KLUE).
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/klue_tc`](https://huggingface.co/datasets/mteb/klue_tc) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2105.09680)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | News, Written | human-annotated | found |



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




#### KannadaNewsClassification

The Kannada news dataset contains only the headlines of news article in three categories: Entertainment, Tech, and Sports. The data set contains around 6300 news article headlines which are collected from Kannada news websites. The data set has been cleaned and contains train and test set using which can be used to benchmark topic classification models in Kannada.

**Dataset:** [`Akash190104/kannada_news_classification`](https://huggingface.co/datasets/Akash190104/kannada_news_classification) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/goru001/nlp-for-kannada)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kan | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{kunchukuttan2020indicnlpcorpus,
      author = {Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
      journal = {arXiv preprint arXiv:2005.00085},
      title = {AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
      year = {2020},
    }

    ```




#### KannadaNewsClassification.v2

The Kannada news dataset contains only the headlines of news article in three categories: Entertainment, Tech, and Sports. The data set contains around 6300 news article headlines which are collected from Kannada news websites. The data set has been cleaned and contains train and test set using which can be used to benchmark topic classification models in Kannada.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/kannada_news`](https://huggingface.co/datasets/mteb/kannada_news) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/goru001/nlp-for-kannada)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kan | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{kunchukuttan2020indicnlpcorpus,
      author = {Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
      journal = {arXiv preprint arXiv:2005.00085},
      title = {AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
      year = {2020},
    }

    ```




#### KinopoiskClassification

Kinopoisk review sentiment classification

**Dataset:** [`ai-forever/kinopoisk-sentiment-classification`](https://huggingface.co/datasets/ai-forever/kinopoisk-sentiment-classification) • **License:** not specified • [Learn more →](https://www.dialog-21.ru/media/1226/blinovpd.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{blinov2013research,
      author = {Blinov, PD and Klekovkina, Maria and Kotelnikov, Eugeny and Pestov, Oleg},
      journal = {Computational Linguistics and Intellectual Technologies},
      number = {12},
      pages = {48--58},
      title = {Research of lexical approach and machine learning methods for sentiment analysis},
      volume = {2},
      year = {2013},
    }

    ```




#### KorFin

The KorFin-ASC is an extension of KorFin-ABSA, which is a financial sentiment analysis dataset including 8818 samples with (aspect, polarity) pairs annotated. The samples were collected from KLUE-TC and analyst reports from Naver Finance.

**Dataset:** [`amphora/korfin-asc`](https://huggingface.co/datasets/amphora/korfin-asc) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/amphora/korfin-asc)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | Financial, News, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{son2023removing,
      author = {Son, Guijin and Lee, Hanwool and Kang, Nahyeon and Hahm, Moonjeong},
      journal = {arXiv preprint arXiv:2301.03136},
      title = {Removing Non-Stationary Knowledge From Pre-Trained Language Models for Entity-Level Sentiment Classification in Finance},
      year = {2023},
    }

    ```




#### KorHateClassification

The dataset was created to provide the first human-labeled Korean corpus for
        toxic speech detection from a Korean online entertainment news aggregator. Recently,
        two young Korean celebrities suffered from a series of tragic incidents that led to two
        major Korean web portals to close the comments section on their platform. However, this only
        serves as a temporary solution, and the fundamental issue has not been solved yet. This dataset
        hopes to improve Korean hate speech detection. Annotation was performed by 32 annotators,
        consisting of 29 annotators from the crowdsourcing platform DeepNatural AI and three NLP researchers.


**Dataset:** [`mteb/KorHateClassification`](https://huggingface.co/datasets/mteb/KorHateClassification) • **License:** cc-by-sa-4.0 • [Learn more →](https://paperswithcode.com/dataset/korean-hatespeech-dataset)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | Social, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{moon2020beep,
      archiveprefix = {arXiv},
      author = {Jihyung Moon and Won Ik Cho and Junbum Lee},
      eprint = {2005.12503},
      primaryclass = {cs.CL},
      title = {BEEP! Korean Corpus of Online News Comments for Toxic Speech Detection},
      year = {2020},
    }

    ```




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



??? quote "Citation"


    ```bibtex

    @misc{moon2020beep,
      archiveprefix = {arXiv},
      author = {Jihyung Moon and Won Ik Cho and Junbum Lee},
      eprint = {2005.12503},
      primaryclass = {cs.CL},
      title = {BEEP! Korean Corpus of Online News Comments for Toxic Speech Detection},
      year = {2020},
    }

    ```




#### KorSarcasmClassification


        The Korean Sarcasm Dataset was created to detect sarcasm in text, which can significantly alter the original
        meaning of a sentence. 9319 tweets were collected from Twitter and labeled for sarcasm or not_sarcasm. These
        tweets were gathered by querying for: irony sarcastic, and
        sarcasm.
        The dataset was created by gathering HTML data from Twitter. Queries for hashtags that include sarcasm
        and variants of it were used to return tweets. It was preprocessed by removing the keyword
        hashtag, urls and mentions of the user to preserve anonymity.


**Dataset:** [`mteb/KorSarcasmClassification`](https://huggingface.co/datasets/mteb/KorSarcasmClassification) • **License:** mit • [Learn more →](https://github.com/SpellOnYou/korean-sarcasm)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kor | Social, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{kim2019kocasm,
      author = {Kim, Jiwon and Cho, Won Ik},
      howpublished = {https://github.com/SpellOnYou/korean-sarcasm},
      journal = {GitHub repository},
      publisher = {GitHub},
      title = {Kocasm: Korean Automatic Sarcasm Detection},
      year = {2019},
    }

    ```




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



??? quote "Citation"


    ```bibtex

    @misc{kim2019kocasm,
      author = {Kim, Jiwon and Cho, Won Ik},
      howpublished = {https://github.com/SpellOnYou/korean-sarcasm},
      journal = {GitHub repository},
      publisher = {GitHub},
      title = {Kocasm: Korean Automatic Sarcasm Detection},
      year = {2019},
    }

    ```




#### KurdishSentimentClassification

Kurdish Sentiment Dataset

**Dataset:** [`asparius/Kurdish-Sentiment`](https://huggingface.co/datasets/asparius/Kurdish-Sentiment) • **License:** cc-by-4.0 • [Learn more →](https://link.springer.com/article/10.1007/s10579-023-09716-6)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kur | Web, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{article,
      author = {Badawi, Soran and Kazemi, Arefeh and Rezaie, Vali},
      doi = {10.1007/s10579-023-09716-6},
      journal = {Language Resources and Evaluation},
      month = {01},
      pages = {1-20},
      title = {KurdiSent: a corpus for kurdish sentiment analysis},
      year = {2024},
    }

    ```




#### KurdishSentimentClassification.v2

Kurdish Sentiment Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/kurdish_sentiment`](https://huggingface.co/datasets/mteb/kurdish_sentiment) • **License:** cc-by-4.0 • [Learn more →](https://link.springer.com/article/10.1007/s10579-023-09716-6)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | kur | Web, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{article,
      author = {Badawi, Soran and Kazemi, Arefeh and Rezaie, Vali},
      doi = {10.1007/s10579-023-09716-6},
      journal = {Language Resources and Evaluation},
      month = {01},
      pages = {1-20},
      title = {KurdiSent: a corpus for kurdish sentiment analysis},
      year = {2024},
    }

    ```




#### LanguageClassification

A language identification dataset for 20 languages.

**Dataset:** [`papluca/language-identification`](https://huggingface.co/datasets/papluca/language-identification) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/papluca/language-identification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara, bul, cmn, deu, ell, ... (20) | Fiction, Government, Non-fiction, Reviews, Web, ... (6) | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{conneau2018xnli,
      author = {Conneau, Alexis
    and Rinott, Ruty
    and Lample, Guillaume
    and Williams, Adina
    and Bowman, Samuel R.
    and Schwenk, Holger
    and Stoyanov, Veselin},
      booktitle = {Proceedings of the 2018 Conference on Empirical Methods
    in Natural Language Processing},
      location = {Brussels, Belgium},
      publisher = {Association for Computational Linguistics},
      title = {XNLI: Evaluating Cross-lingual Sentence Representations},
      year = {2018},
    }

    ```




#### LccSentimentClassification

The leipzig corpora collection, annotated for sentiment

**Dataset:** [`DDSC/lcc`](https://huggingface.co/datasets/DDSC/lcc) • **License:** cc-by-4.0 • [Learn more →](https://github.com/fnielsen/lcc-sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan | News, Web, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{quasthoff-etal-2006-corpus,
      abstract = {A simple and flexible schema for storing and presenting monolingual language resources is proposed. In this format, data for 18 different languages is already available in various sizes. The data is provided free of charge for online use and download. The main target is to ease the application of algorithms for monolingual and interlingual studies.},
      address = {Genoa, Italy},
      author = {Quasthoff, Uwe  and
    Richter, Matthias  and
    Biemann, Christian},
      booktitle = {Proceedings of the Fifth International Conference on Language Resources and Evaluation ({LREC}{'}06)},
      editor = {Calzolari, Nicoletta  and
    Choukri, Khalid  and
    Gangemi, Aldo  and
    Maegaard, Bente  and
    Mariani, Joseph  and
    Odijk, Jan  and
    Tapias, Daniel},
      month = may,
      publisher = {European Language Resources Association (ELRA)},
      title = {Corpus Portal for Search in Monolingual Corpora},
      url = {http://www.lrec-conf.org/proceedings/lrec2006/pdf/641_pdf.pdf},
      year = {2006},
    }

    ```




#### LearnedHandsBenefitsLegalBenchClassification

This is a binary classification task in which the model must determine if a user's legal post discusses public benefits and social services that people can get from the government, like for food, disability, old age, housing, medical help, unemployment, child care, or other social needs.

**Dataset:** [`mteb/LearnedHandsBenefitsLegalBenchClassification`](https://huggingface.co/datasets/mteb/LearnedHandsBenefitsLegalBenchClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @dataset{learned_hands,
      author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
      note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
      title = {LearnedHands Dataset},
      url = {https://spot.suffolklitlab.org/data/#learnedhands},
      urldate = {2022-05-21},
      year = {2022},
    }

    ```




#### LearnedHandsBusinessLegalBenchClassification

This is a binary classification task in which the model must determine if a user's legal question discusses issues faced by people who run small businesses or nonprofits, including around incorporation, licenses, taxes, regulations, and other concerns. It also includes options when there are disasters, bankruptcies, or other problems.

**Dataset:** [`mteb/LearnedHandsBusinessLegalBenchClassification`](https://huggingface.co/datasets/mteb/LearnedHandsBusinessLegalBenchClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @dataset{learned_hands,
      author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
      note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
      title = {LearnedHands Dataset},
      url = {https://spot.suffolklitlab.org/data/#learnedhands},
      urldate = {2022-05-21},
      year = {2022},
    }

    ```




#### LearnedHandsConsumerLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues people face regarding money, insurance, consumer goods and contracts, taxes, and small claims about quality of service.

**Dataset:** [`mteb/LearnedHandsConsumerLegalBenchClassification`](https://huggingface.co/datasets/mteb/LearnedHandsConsumerLegalBenchClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @dataset{learned_hands,
      author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
      note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
      title = {LearnedHands Dataset},
      url = {https://spot.suffolklitlab.org/data/#learnedhands},
      urldate = {2022-05-21},
      year = {2022},
    }

    ```




#### LearnedHandsCourtsLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses the logistics of how a person can interact with a lawyer or the court system. It applies to situations about procedure, rules, how to file lawsuits, how to hire lawyers, how to represent oneself, and other practical matters about dealing with these systems.

**Dataset:** [`mteb/LearnedHandsCourtsLegalBenchClassification`](https://huggingface.co/datasets/mteb/LearnedHandsCourtsLegalBenchClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @dataset{learned_hands,
      author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
      note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
      title = {LearnedHands Dataset},
      url = {https://spot.suffolklitlab.org/data/#learnedhands},
      urldate = {2022-05-21},
      year = {2022},
    }

    ```




#### LearnedHandsCrimeLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues in the criminal system including when people are charged with crimes, go to a criminal trial, go to prison, or are a victim of a crime.

**Dataset:** [`mteb/LearnedHandsCrimeLegalBenchClassification`](https://huggingface.co/datasets/mteb/LearnedHandsCrimeLegalBenchClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @dataset{learned_hands,
      author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
      note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
      title = {LearnedHands Dataset},
      url = {https://spot.suffolklitlab.org/data/#learnedhands},
      urldate = {2022-05-21},
      year = {2022},
    }

    ```




#### LearnedHandsDivorceLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues around filing for divorce, separation, or annulment, getting spousal support, splitting money and property, and following the court processes.

**Dataset:** [`mteb/LearnedHandsDivorceLegalBenchClassification`](https://huggingface.co/datasets/mteb/LearnedHandsDivorceLegalBenchClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @dataset{learned_hands,
      author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
      note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
      title = {LearnedHands Dataset},
      url = {https://spot.suffolklitlab.org/data/#learnedhands},
      urldate = {2022-05-21},
      year = {2022},
    }

    ```




#### LearnedHandsDomesticViolenceLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses dealing with domestic violence and abuse, including getting protective orders, enforcing them, understanding abuse, reporting abuse, and getting resources and status if there is abuse.

**Dataset:** [`mteb/LearnedHandsDomesticViolenceLegalBenchClassification`](https://huggingface.co/datasets/mteb/LearnedHandsDomesticViolenceLegalBenchClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @dataset{learned_hands,
      author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
      note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
      title = {LearnedHands Dataset},
      url = {https://spot.suffolklitlab.org/data/#learnedhands},
      urldate = {2022-05-21},
      year = {2022},
    }

    ```




#### LearnedHandsEducationLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues around school, including accommodations for special needs, discrimination, student debt, discipline, and other issues in education.

**Dataset:** [`mteb/LearnedHandsEducationLegalBenchClassification`](https://huggingface.co/datasets/mteb/LearnedHandsEducationLegalBenchClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @dataset{learned_hands,
      author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
      note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
      title = {LearnedHands Dataset},
      url = {https://spot.suffolklitlab.org/data/#learnedhands},
      urldate = {2022-05-21},
      year = {2022},
    }

    ```




#### LearnedHandsEmploymentLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues related to working at a job, including discrimination and harassment, worker's compensation, workers rights, unions, getting paid, pensions, being fired, and more.

**Dataset:** [`mteb/LearnedHandsEmploymentLegalBenchClassification`](https://huggingface.co/datasets/mteb/LearnedHandsEmploymentLegalBenchClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @dataset{learned_hands,
      author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
      note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
      title = {LearnedHands Dataset},
      url = {https://spot.suffolklitlab.org/data/#learnedhands},
      urldate = {2022-05-21},
      year = {2022},
    }

    ```




#### LearnedHandsEstatesLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses planning for end-of-life, possible incapacitation, and other special circumstances that would prevent a person from making decisions about their own well-being, finances, and property. This includes issues around wills, powers of attorney, advance directives, trusts, guardianships, conservatorships, and other estate issues that people and families deal with.

**Dataset:** [`mteb/LearnedHandsEstatesLegalBenchClassification`](https://huggingface.co/datasets/mteb/LearnedHandsEstatesLegalBenchClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @dataset{learned_hands,
      author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
      note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
      title = {LearnedHands Dataset},
      url = {https://spot.suffolklitlab.org/data/#learnedhands},
      urldate = {2022-05-21},
      year = {2022},
    }

    ```




#### LearnedHandsFamilyLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues that arise within a family, like divorce, adoption, name change, guardianship, domestic violence, child custody, and other issues.

**Dataset:** [`mteb/LearnedHandsFamilyLegalBenchClassification`](https://huggingface.co/datasets/mteb/LearnedHandsFamilyLegalBenchClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @dataset{learned_hands,
      author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
      note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
      title = {LearnedHands Dataset},
      url = {https://spot.suffolklitlab.org/data/#learnedhands},
      urldate = {2022-05-21},
      year = {2022},
    }

    ```




#### LearnedHandsHealthLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues with accessing health services, paying for medical care, getting public benefits for health care, protecting one's rights in medical settings, and other issues related to health.

**Dataset:** [`mteb/LearnedHandsHealthLegalBenchClassification`](https://huggingface.co/datasets/mteb/LearnedHandsHealthLegalBenchClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @dataset{learned_hands,
      author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
      note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
      title = {LearnedHands Dataset},
      url = {https://spot.suffolklitlab.org/data/#learnedhands},
      urldate = {2022-05-21},
      year = {2022},
    }

    ```




#### LearnedHandsHousingLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses issues with paying your rent or mortgage, landlord-tenant issues, housing subsidies and public housing, eviction, and other problems with your apartment, mobile home, or house.

**Dataset:** [`mteb/LearnedHandsHousingLegalBenchClassification`](https://huggingface.co/datasets/mteb/LearnedHandsHousingLegalBenchClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @dataset{learned_hands,
      author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
      note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
      title = {LearnedHands Dataset},
      url = {https://spot.suffolklitlab.org/data/#learnedhands},
      urldate = {2022-05-21},
      year = {2022},
    }

    ```




#### LearnedHandsImmigrationLegalBenchClassification

This is a binary classification task in which the model must determine if a user's post discusses visas, asylum, green cards, citizenship, migrant work and benefits, and other issues faced by people who are not full citizens in the US.

**Dataset:** [`mteb/LearnedHandsImmigrationLegalBenchClassification`](https://huggingface.co/datasets/mteb/LearnedHandsImmigrationLegalBenchClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @dataset{learned_hands,
      author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
      note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
      title = {LearnedHands Dataset},
      url = {https://spot.suffolklitlab.org/data/#learnedhands},
      urldate = {2022-05-21},
      year = {2022},
    }

    ```




#### LearnedHandsTortsLegalBenchClassification

This is a binary classification task in which the model must determine if a user's legal question discusses problems that one person has with another person (or animal), like when there is a car accident, a dog bite, bullying or possible harassment, or neighbors treating each other badly.

**Dataset:** [`mteb/LearnedHandsTortsLegalBenchClassification`](https://huggingface.co/datasets/mteb/LearnedHandsTortsLegalBenchClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @dataset{learned_hands,
      author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
      note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
      title = {LearnedHands Dataset},
      url = {https://spot.suffolklitlab.org/data/#learnedhands},
      urldate = {2022-05-21},
      year = {2022},
    }

    ```




#### LearnedHandsTrafficLegalBenchClassification

This is a binary classification task in which the model must determine if a user's legal post discusses problems with traffic and parking tickets, fees, driver's licenses, and other issues experienced with the traffic system. It also concerns issues with car accidents and injuries, cars' quality, repairs, purchases, and other contracts.

**Dataset:** [`mteb/LearnedHandsTrafficLegalBenchClassification`](https://huggingface.co/datasets/mteb/LearnedHandsTrafficLegalBenchClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @dataset{learned_hands,
      author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
      note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
      title = {LearnedHands Dataset},
      url = {https://spot.suffolklitlab.org/data/#learnedhands},
      urldate = {2022-05-21},
      year = {2022},
    }

    ```




#### LegalReasoningCausalityLegalBenchClassification

Given an excerpt from a district court opinion, classify if it relies on statistical evidence in its reasoning.

**Dataset:** [`mteb/LegalReasoningCausalityLegalBenchClassification`](https://huggingface.co/datasets/mteb/LegalReasoningCausalityLegalBenchClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### LegalReasoningCausalityLegalBenchClassification.v2

Given an excerpt from a district court opinion, classify if it relies on statistical evidence in its reasoning.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/legal_reasoning_causality_legal_bench`](https://huggingface.co/datasets/mteb/legal_reasoning_causality_legal_bench) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




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


**Dataset:** [`mteb/MAUDLegalBenchClassification`](https://huggingface.co/datasets/mteb/MAUDLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @article{wang2023maud,
      author = {Wang, Steven H and Scardigli, Antoine and Tang, Leonard and Chen, Wei and Levkin, Dimitry and Chen, Anya and Ball, Spencer and Woodside, Thomas and Zhang, Oliver and Hendrycks, Dan},
      journal = {arXiv preprint arXiv:2301.00876},
      title = {MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding},
      year = {2023},
    }

    ```




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

    @article{wang2023maud,
      author = {Wang, Steven H and Scardigli, Antoine and Tang, Leonard and Chen, Wei and Levkin, Dimitry and Chen, Anya and Ball, Spencer and Woodside, Thomas and Zhang, Oliver and Hendrycks, Dan},
      journal = {arXiv preprint arXiv:2301.00876},
      title = {MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding},
      year = {2023},
    }

    ```




#### MTOPDomainClassification

MTOP: Multilingual Task-Oriented Semantic Parsing

**Dataset:** [`mteb/MTOPDomainClassification`](https://huggingface.co/datasets/mteb/MTOPDomainClassification) • **License:** not specified • [Learn more →](https://arxiv.org/pdf/2008.09335.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu, eng, fra, hin, spa, ... (6) | Spoken, Spoken | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{li-etal-2021-mtop,
      abstract = {Scaling semantic parsing models for task-oriented dialog systems to new languages is often expensive and time-consuming due to the lack of available datasets. Available datasets suffer from several shortcomings: a) they contain few languages b) they contain small amounts of labeled examples per language c) they are based on the simple intent and slot detection paradigm for non-compositional queries. In this paper, we present a new multilingual dataset, called MTOP, comprising of 100k annotated utterances in 6 languages across 11 domains. We use this dataset and other publicly available datasets to conduct a comprehensive benchmarking study on using various state-of-the-art multilingual pre-trained models for task-oriented semantic parsing. We achieve an average improvement of +6.3 points on Slot F1 for the two existing multilingual datasets, over best results reported in their experiments. Furthermore, we demonstrate strong zero-shot performance using pre-trained models combined with automatic translation and alignment, and a proposed distant supervision method to reduce the noise in slot label projection.},
      address = {Online},
      author = {Li, Haoran  and
    Arora, Abhinav  and
    Chen, Shuohui  and
    Gupta, Anchit  and
    Gupta, Sonal  and
    Mehdad, Yashar},
      booktitle = {Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume},
      doi = {10.18653/v1/2021.eacl-main.257},
      editor = {Merlo, Paola  and
    Tiedemann, Jorg  and
    Tsarfaty, Reut},
      month = apr,
      pages = {2950--2962},
      publisher = {Association for Computational Linguistics},
      title = {{MTOP}: A Comprehensive Multilingual Task-Oriented Semantic Parsing Benchmark},
      url = {https://aclanthology.org/2021.eacl-main.257},
      year = {2021},
    }

    ```




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




#### MTOPIntentClassification

MTOP: Multilingual Task-Oriented Semantic Parsing

**Dataset:** [`mteb/MTOPIntentClassification`](https://huggingface.co/datasets/mteb/MTOPIntentClassification) • **License:** not specified • [Learn more →](https://arxiv.org/pdf/2008.09335.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu, eng, fra, hin, spa, ... (6) | Spoken, Spoken | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{li-etal-2021-mtop,
      abstract = {Scaling semantic parsing models for task-oriented dialog systems to new languages is often expensive and time-consuming due to the lack of available datasets. Available datasets suffer from several shortcomings: a) they contain few languages b) they contain small amounts of labeled examples per language c) they are based on the simple intent and slot detection paradigm for non-compositional queries. In this paper, we present a new multilingual dataset, called MTOP, comprising of 100k annotated utterances in 6 languages across 11 domains. We use this dataset and other publicly available datasets to conduct a comprehensive benchmarking study on using various state-of-the-art multilingual pre-trained models for task-oriented semantic parsing. We achieve an average improvement of +6.3 points on Slot F1 for the two existing multilingual datasets, over best results reported in their experiments. Furthermore, we demonstrate strong zero-shot performance using pre-trained models combined with automatic translation and alignment, and a proposed distant supervision method to reduce the noise in slot label projection.},
      address = {Online},
      author = {Li, Haoran  and
    Arora, Abhinav  and
    Chen, Shuohui  and
    Gupta, Anchit  and
    Gupta, Sonal  and
    Mehdad, Yashar},
      booktitle = {Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume},
      doi = {10.18653/v1/2021.eacl-main.257},
      editor = {Merlo, Paola  and
    Tiedemann, Jorg  and
    Tsarfaty, Reut},
      month = apr,
      pages = {2950--2962},
      publisher = {Association for Computational Linguistics},
      title = {{MTOP}: A Comprehensive Multilingual Task-Oriented Semantic Parsing Benchmark},
      url = {https://aclanthology.org/2021.eacl-main.257},
      year = {2021},
    }

    ```




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




#### MacedonianTweetSentimentClassification

An Macedonian dataset for tweet sentiment classification.

**Dataset:** [`isaacchung/macedonian-tweet-sentiment-classification`](https://huggingface.co/datasets/isaacchung/macedonian-tweet-sentiment-classification) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://aclanthology.org/R15-1034/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | mkd | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{jovanoski-etal-2015-sentiment,
      address = {Hissar, Bulgaria},
      author = {Jovanoski, Dame  and
    Pachovski, Veno  and
    Nakov, Preslav},
      booktitle = {Proceedings of the International Conference Recent Advances in Natural Language Processing},
      editor = {Mitkov, Ruslan  and
    Angelova, Galia  and
    Bontcheva, Kalina},
      month = sep,
      pages = {249--257},
      publisher = {INCOMA Ltd. Shoumen, BULGARIA},
      title = {Sentiment Analysis in {T}witter for {M}acedonian},
      url = {https://aclanthology.org/R15-1034},
      year = {2015},
    }

    ```




#### MacedonianTweetSentimentClassification.v2

An Macedonian dataset for tweet sentiment classification.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/macedonian_tweet_sentiment`](https://huggingface.co/datasets/mteb/macedonian_tweet_sentiment) • **License:** cc-by-nc-sa-3.0 • [Learn more →](https://aclanthology.org/R15-1034/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | mkd | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{jovanoski-etal-2015-sentiment,
      address = {Hissar, Bulgaria},
      author = {Jovanoski, Dame  and
    Pachovski, Veno  and
    Nakov, Preslav},
      booktitle = {Proceedings of the International Conference Recent Advances in Natural Language Processing},
      editor = {Mitkov, Ruslan  and
    Angelova, Galia  and
    Bontcheva, Kalina},
      month = sep,
      pages = {249--257},
      publisher = {INCOMA Ltd. Shoumen, BULGARIA},
      title = {Sentiment Analysis in {T}witter for {M}acedonian},
      url = {https://aclanthology.org/R15-1034},
      year = {2015},
    }

    ```




#### MalayalamNewsClassification

A Malayalam dataset for 3-class classification of Malayalam news articles

**Dataset:** [`mlexplorer008/malayalam_news_classification`](https://huggingface.co/datasets/mlexplorer008/malayalam_news_classification) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-malyalam)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | mal | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{kunchukuttan2020indicnlpcorpus,
      author = {Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
      journal = {arXiv preprint arXiv:2005.00085},
      title = {AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
      year = {2020},
    }

    ```




#### MalayalamNewsClassification.v2

A Malayalam dataset for 3-class classification of Malayalam news articles
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/malayalam_news`](https://huggingface.co/datasets/mteb/malayalam_news) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-malyalam)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | mal | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{kunchukuttan2020indicnlpcorpus,
      author = {Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
      journal = {arXiv preprint arXiv:2005.00085},
      title = {AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
      year = {2020},
    }

    ```




#### MarathiNewsClassification

A Marathi dataset for 3-class classification of Marathi news articles

**Dataset:** [`mlexplorer008/marathi_news_classification`](https://huggingface.co/datasets/mlexplorer008/marathi_news_classification) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-marathi)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | mar | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{kunchukuttan2020indicnlpcorpus,
      author = {Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
      journal = {arXiv preprint arXiv:2005.00085},
      title = {AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
      year = {2020},
    }

    ```




#### MarathiNewsClassification.v2

A Marathi dataset for 3-class classification of Marathi news articles
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/marathi_news`](https://huggingface.co/datasets/mteb/marathi_news) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-marathi)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | mar | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{kunchukuttan2020indicnlpcorpus,
      author = {Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
      journal = {arXiv preprint arXiv:2005.00085},
      title = {AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
      year = {2020},
    }

    ```




#### MasakhaNEWSClassification

MasakhaNEWS is the largest publicly available dataset for news topic classification in 16 languages widely spoken in Africa. The train/validation/test sets are available for all the 16 languages.

**Dataset:** [`mteb/masakhanews`](https://huggingface.co/datasets/mteb/masakhanews) • **License:** cc-by-nc-4.0 • [Learn more →](https://arxiv.org/abs/2304.09972)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | amh, eng, fra, hau, ibo, ... (16) | News, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{adelani2023masakhanews,
      archiveprefix = {arXiv},
      author = {David Ifeoluwa Adelani and Marek Masiak and Israel Abebe Azime and Jesujoba Alabi and Atnafu Lambebo Tonja and Christine Mwase and Odunayo Ogundepo and Bonaventure F. P. Dossou and Akintunde Oladipo and Doreen Nixdorf and Chris Chinenye Emezue and sana al-azzawi and Blessing Sibanda and Davis David and Lolwethu Ndolela and Jonathan Mukiibi and Tunde Ajayi and Tatiana Moteu and Brian Odhiambo and Abraham Owodunni and Nnaemeka Obiefuna and Muhidin Mohamed and Shamsuddeen Hassan Muhammad and Teshome Mulugeta Ababu and Saheed Abdullahi Salahudeen and Mesay Gemeda Yigezu and Tajuddeen Gwadabe and Idris Abdulmumin and Mahlet Taye and Oluwabusayo Awoyomi and Iyanuoluwa Shode and Tolulope Adelani and Habiba Abdulganiyu and Abdul-Hakeem Omotayo and Adetola Adeeko and Abeeb Afolabi and Anuoluwapo Aremu and Olanrewaju Samuel and Clemencia Siro and Wangari Kimotho and Onyekachi Ogbu and Chinedu Mbonu and Chiamaka Chukwuneke and Samuel Fanijo and Jessica Ojo and Oyinkansola Awosan and Tadesse Kebede and Toadoum Sari Sakayo and Pamela Nyatsine and Freedmore Sidume and Oreen Yousuf and Mardiyyah Oduwole and Tshinu Tshinu and Ussen Kimanuka and Thina Diko and Siyanda Nxakama and Sinodos Nigusse and Abdulmejid Johar and Shafie Mohamed and Fuad Mire Hassan and Moges Ahmed Mehamed and Evrard Ngabire and Jules Jules and Ivan Ssenkungu and Pontus Stenetorp},
      eprint = {2304.09972},
      primaryclass = {cs.CL},
      title = {MasakhaNEWS: News Topic Classification for African languages},
      year = {2023},
    }

    ```




#### MassiveIntentClassification

MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages

**Dataset:** [`mteb/amazon_massive_intent`](https://huggingface.co/datasets/mteb/amazon_massive_intent) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2204.08582)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | afr, amh, ara, aze, ben, ... (50) | Spoken | human-annotated | human-translated and localized |



??? quote "Citation"


    ```bibtex

    @misc{fitzgerald2022massive,
      archiveprefix = {arXiv},
      author = {Jack FitzGerald and Christopher Hench and Charith Peris and Scott Mackie and Kay Rottmann and Ana Sanchez and Aaron Nash and Liam Urbach and Vishesh Kakarala and Richa Singh and Swetha Ranganath and Laurie Crist and Misha Britan and Wouter Leeuwis and Gokhan Tur and Prem Natarajan},
      eprint = {2204.08582},
      primaryclass = {cs.CL},
      title = {MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages},
      year = {2022},
    }

    ```




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




#### MassiveScenarioClassification

MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages

**Dataset:** [`mteb/amazon_massive_scenario`](https://huggingface.co/datasets/mteb/amazon_massive_scenario) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/2204.08582)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | afr, amh, ara, aze, ben, ... (50) | Spoken | human-annotated | human-translated and localized |



??? quote "Citation"


    ```bibtex

    @misc{fitzgerald2022massive,
      archiveprefix = {arXiv},
      author = {Jack FitzGerald and Christopher Hench and Charith Peris and Scott Mackie and Kay Rottmann and Ana Sanchez and Aaron Nash and Liam Urbach and Vishesh Kakarala and Richa Singh and Swetha Ranganath and Laurie Crist and Misha Britan and Wouter Leeuwis and Gokhan Tur and Prem Natarajan},
      eprint = {2204.08582},
      primaryclass = {cs.CL},
      title = {MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages},
      year = {2022},
    }

    ```




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




#### Moroco

The Moldavian and Romanian Dialectal Corpus. The MOROCO data set contains Moldavian and Romanian samples of text collected from the news domain. The samples belong to one of the following six topics: (0) culture, (1) finance, (2) politics, (3) science, (4) sports, (5) tech

**Dataset:** [`mteb/Moroco`](https://huggingface.co/datasets/mteb/Moroco) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/moroco)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ron | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{Butnaru-ACL-2019,
      author = {Andrei M. Butnaru and Radu Tudor Ionescu},
      booktitle = {Proceedings of ACL},
      pages = {688--698},
      title = {{MOROCO: The Moldavian and Romanian Dialectal Corpus}},
      year = {2019},
    }

    ```




#### Moroco.v2

The Moldavian and Romanian Dialectal Corpus. The MOROCO data set contains Moldavian and Romanian samples of text collected from the news domain. The samples belong to one of the following six topics: (0) culture, (1) finance, (2) politics, (3) science, (4) sports, (5) tech
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/moroco`](https://huggingface.co/datasets/mteb/moroco) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/moroco)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ron | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{Butnaru-ACL-2019,
      author = {Andrei M. Butnaru and Radu Tudor Ionescu},
      booktitle = {Proceedings of ACL},
      pages = {688--698},
      title = {{MOROCO: The Moldavian and Romanian Dialectal Corpus}},
      year = {2019},
    }

    ```




#### MovieReviewSentimentClassification

The Allociné dataset is a French-language dataset for sentiment analysis that contains movie reviews produced by the online community of the Allociné.fr website.

**Dataset:** [`tblard/allocine`](https://huggingface.co/datasets/tblard/allocine) • **License:** mit • [Learn more →](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fra | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @software{blard2020,
      author = {Théophile Blard},
      title = {French sentiment analysis with BERT},
      url = {https://github.com/TheophileBlard/french-sentiment-analysis-with-bert},
      year = {2020},
    }

    ```




#### MovieReviewSentimentClassification.v2

The Allociné dataset is a French-language dataset for sentiment analysis that contains movie reviews produced by the online community of the Allociné.fr website.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/movie_review_sentiment`](https://huggingface.co/datasets/mteb/movie_review_sentiment) • **License:** mit • [Learn more →](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fra | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @software{blard2020,
      author = {Théophile Blard},
      title = {French sentiment analysis with BERT},
      url = {https://github.com/TheophileBlard/french-sentiment-analysis-with-bert},
      year = {2020},
    }

    ```




#### MultiHateClassification

Hate speech detection dataset with binary
                       (hateful vs non-hateful) labels. Includes 25+ distinct types of hate
                       and challenging non-hate, and 11 languages.


**Dataset:** [`mteb/multi-hatecheck`](https://huggingface.co/datasets/mteb/multi-hatecheck) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2022.woah-1.15/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara, cmn, deu, eng, fra, ... (11) | Constructed, Written | expert-annotated | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{rottger-etal-2021-hatecheck,
      abstract = {Detecting online hate is a difficult task that even state-of-the-art models struggle with. Typically, hate speech detection models are evaluated by measuring their performance on held-out test data using metrics such as accuracy and F1 score. However, this approach makes it difficult to identify specific model weak points. It also risks overestimating generalisable model performance due to increasingly well-evidenced systematic gaps and biases in hate speech datasets. To enable more targeted diagnostic insights, we introduce HateCheck, a suite of functional tests for hate speech detection models. We specify 29 model functionalities motivated by a review of previous research and a series of interviews with civil society stakeholders. We craft test cases for each functionality and validate their quality through a structured annotation process. To illustrate HateCheck{'}s utility, we test near-state-of-the-art transformer models as well as two popular commercial models, revealing critical model weaknesses.},
      address = {Online},
      author = {R{\"o}ttger, Paul  and
    Vidgen, Bertie  and
    Nguyen, Dong  and
    Waseem, Zeerak  and
    Margetts, Helen  and
    Pierrehumbert, Janet},
      booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
      doi = {10.18653/v1/2021.acl-long.4},
      editor = {Zong, Chengqing  and
    Xia, Fei  and
    Li, Wenjie  and
    Navigli, Roberto},
      month = aug,
      pages = {41--58},
      publisher = {Association for Computational Linguistics},
      title = {{H}ate{C}heck: Functional Tests for Hate Speech Detection Models},
      url = {https://aclanthology.org/2021.acl-long.4},
      year = {2021},
    }

    @inproceedings{rottger-etal-2022-multilingual,
      abstract = {Hate speech detection models are typically evaluated on held-out test sets. However, this risks painting an incomplete and potentially misleading picture of model performance because of increasingly well-documented systematic gaps and biases in hate speech datasets. To enable more targeted diagnostic insights, recent research has thus introduced functional tests for hate speech detection models. However, these tests currently only exist for English-language content, which means that they cannot support the development of more effective models in other languages spoken by billions across the world. To help address this issue, we introduce Multilingual HateCheck (MHC), a suite of functional tests for multilingual hate speech detection models. MHC covers 34 functionalities across ten languages, which is more languages than any other hate speech dataset. To illustrate MHC{'}s utility, we train and test a high-performing multilingual hate speech detection model, and reveal critical model weaknesses for monolingual and cross-lingual applications.},
      address = {Seattle, Washington (Hybrid)},
      author = {R{\"o}ttger, Paul  and
    Seelawi, Haitham  and
    Nozza, Debora  and
    Talat, Zeerak  and
    Vidgen, Bertie},
      booktitle = {Proceedings of the Sixth Workshop on Online Abuse and Harms (WOAH)},
      doi = {10.18653/v1/2022.woah-1.15},
      editor = {Narang, Kanika  and
    Mostafazadeh Davani, Aida  and
    Mathias, Lambert  and
    Vidgen, Bertie  and
    Talat, Zeerak},
      month = jul,
      pages = {154--169},
      publisher = {Association for Computational Linguistics},
      title = {Multilingual {H}ate{C}heck: Functional Tests for Multilingual Hate Speech Detection Models},
      url = {https://aclanthology.org/2022.woah-1.15},
      year = {2022},
    }

    ```




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

Sentiment classification dataset with binary (positive vs negative sentiment) labels. Includes 30 languages and dialects.

**Dataset:** [`mteb/multilingual-sentiment-classification`](https://huggingface.co/datasets/mteb/multilingual-sentiment-classification) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/mteb/multilingual-sentiment-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara, bam, bul, cmn, cym, ... (31) | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{mollanorozy-etal-2023-cross,
      address = {Dubrovnik, Croatia},
      author = {Mollanorozy, Sepideh  and
    Tanti, Marc  and
    Nissim, Malvina},
      booktitle = {Proceedings of the 5th Workshop on Research in Computational Linguistic Typology and Multilingual NLP},
      doi = {10.18653/v1/2023.sigtyp-1.9},
      editor = {Beinborn, Lisa  and
    Goswami, Koustava  and
    Murado{\\u{g}}lu, Saliha  and
    Sorokin, Alexey  and
    Kumar, Ritesh  and
    Shcherbakov, Andreas  and
    Ponti, Edoardo M.  and
    Cotterell, Ryan  and
    Vylomova, Ekaterina},
      month = may,
      pages = {89--95},
      publisher = {Association for Computational Linguistics},
      title = {Cross-lingual Transfer Learning with \{P\}ersian},
      url = {https://aclanthology.org/2023.sigtyp-1.9},
      year = {2023},
    }

    ```




#### MyanmarNews

The Myanmar News dataset on Hugging Face contains news articles in Burmese. It is designed for tasks such as text classification, sentiment analysis, and language modeling. The dataset includes a variety of news topics in 4 categorie, providing a rich resource for natural language processing applications involving Burmese which is a low resource language.

**Dataset:** [`mteb/MyanmarNews`](https://huggingface.co/datasets/mteb/MyanmarNews) • **License:** gpl-3.0 • [Learn more →](https://huggingface.co/datasets/myanmar_news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | mya | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{Khine2017,
      author = {A. H. Khine and K. T. Nwet and K. M. Soe},
      booktitle = {15th Proceedings of International Conference on Computer Applications},
      month = {February},
      pages = {401--408},
      title = {Automatic Myanmar News Classification},
      year = {2017},
    }

    ```




#### MyanmarNews.v2

The Myanmar News dataset on Hugging Face contains news articles in Burmese. It is designed for tasks such as text classification, sentiment analysis, and language modeling. The dataset includes a variety of news topics in 4 categorie, providing a rich resource for natural language processing applications involving Burmese which is a low resource language.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/myanmar_news`](https://huggingface.co/datasets/mteb/myanmar_news) • **License:** gpl-3.0 • [Learn more →](https://huggingface.co/datasets/myanmar_news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | mya | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{Khine2017,
      author = {A. H. Khine and K. T. Nwet and K. M. Soe},
      booktitle = {15th Proceedings of International Conference on Computer Applications},
      month = {February},
      pages = {401--408},
      title = {Automatic Myanmar News Classification},
      year = {2017},
    }

    ```




#### NLPTwitterAnalysisClassification

Twitter Analysis Classification

**Dataset:** [`hamedhf/nlp_twitter_analysis`](https://huggingface.co/datasets/hamedhf/nlp_twitter_analysis) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/hamedhf/nlp_twitter_analysis/tree/main)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Social | derived | found |



??? quote "Citation"


    ```bibtex

    ```




#### NLPTwitterAnalysisClassification.v2

Twitter Analysis Classification
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/nlp_twitter_analysis`](https://huggingface.co/datasets/mteb/nlp_twitter_analysis) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/hamedhf/nlp_twitter_analysis/tree/main)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Social | derived | found |



??? quote "Citation"


    ```bibtex

    ```




#### NYSJudicialEthicsLegalBenchClassification

Answer questions on judicial ethics from the New York State Unified Court System Advisory Committee.

**Dataset:** [`mteb/NYSJudicialEthicsLegalBenchClassification`](https://huggingface.co/datasets/mteb/NYSJudicialEthicsLegalBenchClassification) • **License:** mit • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### NaijaSenti

NaijaSenti is the first large-scale human-annotated Twitter sentiment dataset for the four most widely spoken languages in Nigeria — Hausa, Igbo, Nigerian-Pidgin, and Yorùbá — consisting of around 30,000 annotated tweets per language, including a significant fraction of code-mixed tweets.

**Dataset:** [`mteb/NaijaSenti`](https://huggingface.co/datasets/mteb/NaijaSenti) • **License:** cc-by-4.0 • [Learn more →](https://github.com/hausanlp/NaijaSenti)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | hau, ibo, pcm, yor | Social, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{muhammad-etal-2022-naijasenti,
      address = {Marseille, France},
      author = {Muhammad, Shamsuddeen Hassan  and
    Adelani, David Ifeoluwa  and
    Ruder, Sebastian  and
    Ahmad, Ibrahim Sa{'}id  and
    Abdulmumin, Idris  and
    Bello, Bello Shehu  and
    Choudhury, Monojit  and
    Emezue, Chris Chinenye  and
    Abdullahi, Saheed Salahudeen  and
    Aremu, Anuoluwapo  and
    Jorge, Al{\'\i}pio  and
    Brazdil, Pavel},
      booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
      month = jun,
      pages = {590--602},
      publisher = {European Language Resources Association},
      title = {{N}aija{S}enti: A {N}igerian {T}witter Sentiment Corpus for Multilingual Sentiment Analysis},
      url = {https://aclanthology.org/2022.lrec-1.63},
      year = {2022},
    }

    ```




#### NepaliNewsClassification

A Nepali dataset for 7500 news articles

**Dataset:** [`bpHigh/iNLTK_Nepali_News_Dataset`](https://huggingface.co/datasets/bpHigh/iNLTK_Nepali_News_Dataset) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/goru001/nlp-for-nepali)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nep | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{arora-2020-inltk,
      abstract = {We present iNLTK, an open-source NLP library consisting of pre-trained language models and out-of-the-box support for Data Augmentation, Textual Similarity, Sentence Embeddings, Word Embeddings, Tokenization and Text Generation in 13 Indic Languages. By using pre-trained models from iNLTK for text classification on publicly available datasets, we significantly outperform previously reported results. On these datasets, we also show that by using pre-trained models and data augmentation from iNLTK, we can achieve more than 95{\%} of the previous best performance by using less than 10{\%} of the training data. iNLTK is already being widely used by the community and has 40,000+ downloads, 600+ stars and 100+ forks on GitHub.},
      address = {Online},
      author = {Arora, Gaurav},
      booktitle = {Proceedings of Second Workshop for NLP Open Source Software (NLP-OSS)},
      doi = {10.18653/v1/2020.nlposs-1.10},
      editor = {Park, Eunjeong L.  and
    Hagiwara, Masato  and
    Milajevs, Dmitrijs  and
    Liu, Nelson F.  and
    Chauhan, Geeticka  and
    Tan, Liling},
      month = nov,
      pages = {66--71},
      publisher = {Association for Computational Linguistics},
      title = {i{NLTK}: Natural Language Toolkit for Indic Languages},
      url = {https://aclanthology.org/2020.nlposs-1.10},
      year = {2020},
    }

    ```




#### NepaliNewsClassification.v2

A Nepali dataset for 7500 news articles
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/nepali_news`](https://huggingface.co/datasets/mteb/nepali_news) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/goru001/nlp-for-nepali)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nep | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{arora-2020-inltk,
      abstract = {We present iNLTK, an open-source NLP library consisting of pre-trained language models and out-of-the-box support for Data Augmentation, Textual Similarity, Sentence Embeddings, Word Embeddings, Tokenization and Text Generation in 13 Indic Languages. By using pre-trained models from iNLTK for text classification on publicly available datasets, we significantly outperform previously reported results. On these datasets, we also show that by using pre-trained models and data augmentation from iNLTK, we can achieve more than 95{\%} of the previous best performance by using less than 10{\%} of the training data. iNLTK is already being widely used by the community and has 40,000+ downloads, 600+ stars and 100+ forks on GitHub.},
      address = {Online},
      author = {Arora, Gaurav},
      booktitle = {Proceedings of Second Workshop for NLP Open Source Software (NLP-OSS)},
      doi = {10.18653/v1/2020.nlposs-1.10},
      editor = {Park, Eunjeong L.  and
    Hagiwara, Masato  and
    Milajevs, Dmitrijs  and
    Liu, Nelson F.  and
    Chauhan, Geeticka  and
    Tan, Liling},
      month = nov,
      pages = {66--71},
      publisher = {Association for Computational Linguistics},
      title = {i{NLTK}: Natural Language Toolkit for Indic Languages},
      url = {https://aclanthology.org/2020.nlposs-1.10},
      year = {2020},
    }

    ```




#### NewsClassification

Large News Classification Dataset

**Dataset:** [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/1509.01626)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | News, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{NIPS2015_250cf8b5,
      author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
      booktitle = {Advances in Neural Information Processing Systems},
      editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
      pages = {},
      publisher = {Curran Associates, Inc.},
      title = {Character-level Convolutional Networks for Text Classification},
      url = {https://proceedings.neurips.cc/paper_files/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf},
      volume = {28},
      year = {2015},
    }

    ```




#### NewsClassification.v2

Large News Classification Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/news`](https://huggingface.co/datasets/mteb/news) • **License:** apache-2.0 • [Learn more →](https://arxiv.org/abs/1509.01626)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | News, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{NIPS2015_250cf8b5,
      author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
      booktitle = {Advances in Neural Information Processing Systems},
      editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
      pages = {},
      publisher = {Curran Associates, Inc.},
      title = {Character-level Convolutional Networks for Text Classification},
      url = {https://proceedings.neurips.cc/paper_files/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf},
      volume = {28},
      year = {2015},
    }

    ```




#### NoRecClassification

A Norwegian dataset for sentiment classification on review

**Dataset:** [`mteb/norec_classification`](https://huggingface.co/datasets/mteb/norec_classification) • **License:** cc-by-nc-4.0 • [Learn more →](https://aclanthology.org/L18-1661/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nob | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{velldal-etal-2018-norec,
      address = {Miyazaki, Japan},
      author = {Velldal, Erik  and
    {\\O}vrelid, Lilja  and
    Bergem, Eivind Alexander  and
    Stadsnes, Cathrine  and
    Touileb, Samia  and
    J{\\o}rgensen, Fredrik},
      booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)},
      editor = {Calzolari, Nicoletta  and
    Choukri, Khalid  and
    Cieri, Christopher  and
    Declerck, Thierry  and
    Goggi, Sara  and
    Hasida, Koiti  and
    Isahara, Hitoshi  and
    Maegaard, Bente  and
    Mariani, Joseph  and
    Mazo, H{\\'e}l{\\`e}ne  and
    Moreno, Asuncion  and
    Odijk, Jan  and
    Piperidis, Stelios  and
    Tokunaga, Takenobu},
      month = may,
      publisher = {European Language Resources Association (ELRA)},
      title = {{N}o{R}e{C}: The {N}orwegian Review Corpus},
      url = {https://aclanthology.org/L18-1661},
      year = {2018},
    }

    ```




#### NoRecClassification.v2

A Norwegian dataset for sentiment classification on review
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/no_rec`](https://huggingface.co/datasets/mteb/no_rec) • **License:** cc-by-nc-4.0 • [Learn more →](https://aclanthology.org/L18-1661/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nob | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{velldal-etal-2018-norec,
      address = {Miyazaki, Japan},
      author = {Velldal, Erik  and
    {\\O}vrelid, Lilja  and
    Bergem, Eivind Alexander  and
    Stadsnes, Cathrine  and
    Touileb, Samia  and
    J{\\o}rgensen, Fredrik},
      booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)},
      editor = {Calzolari, Nicoletta  and
    Choukri, Khalid  and
    Cieri, Christopher  and
    Declerck, Thierry  and
    Goggi, Sara  and
    Hasida, Koiti  and
    Isahara, Hitoshi  and
    Maegaard, Bente  and
    Mariani, Joseph  and
    Mazo, H{\\'e}l{\\`e}ne  and
    Moreno, Asuncion  and
    Odijk, Jan  and
    Piperidis, Stelios  and
    Tokunaga, Takenobu},
      month = may,
      publisher = {European Language Resources Association (ELRA)},
      title = {{N}o{R}e{C}: The {N}orwegian Review Corpus},
      url = {https://aclanthology.org/L18-1661},
      year = {2018},
    }

    ```




#### NordicLangClassification

A dataset for Nordic language identification.

**Dataset:** [`mteb/NordicLangClassification`](https://huggingface.co/datasets/mteb/NordicLangClassification) • **License:** cc-by-sa-3.0 • [Learn more →](https://aclanthology.org/2021.vardial-1.8/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan, fao, isl, nno, nob, ... (6) | Encyclopaedic | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{haas-derczynski-2021-discriminating,
      abstract = {Automatic language identification is a challenging problem. Discriminating between closely related languages is especially difficult. This paper presents a machine learning approach for automatic language identification for the Nordic languages, which often suffer miscategorisation by existing state-of-the-art tools. Concretely we will focus on discrimination between six Nordic languages: Danish, Swedish, Norwegian (Nynorsk), Norwegian (Bokm{\aa}l), Faroese and Icelandic.},
      address = {Kiyv, Ukraine},
      author = {Haas, Ren{\'e}  and
    Derczynski, Leon},
      booktitle = {Proceedings of the Eighth Workshop on NLP for Similar Languages, Varieties and Dialects},
      editor = {Zampieri, Marcos  and
    Nakov, Preslav  and
    Ljube{\v{s}}i{\'c}, Nikola  and
    Tiedemann, J{\"o}rg  and
    Scherrer, Yves  and
    Jauhiainen, Tommi},
      month = apr,
      pages = {67--75},
      publisher = {Association for Computational Linguistics},
      title = {Discriminating Between Similar {N}ordic Languages},
      url = {https://aclanthology.org/2021.vardial-1.8},
      year = {2021},
    }

    ```




#### NorwegianParliamentClassification

Norwegian parliament speeches annotated for sentiment

**Dataset:** [`mteb/NorwegianParliamentClassification`](https://huggingface.co/datasets/mteb/NorwegianParliamentClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/NbAiLab/norwegian_parliament)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nob | Government, Spoken | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{kummervold-etal-2021-operationalizing,
      abstract = {In this work, we show the process of building a large-scale training set from digital and digitized collections at a national library. The resulting Bidirectional Encoder Representations from Transformers (BERT)-based language model for Norwegian outperforms multilingual BERT (mBERT) models in several token and sequence classification tasks for both Norwegian Bokm{\aa}l and Norwegian Nynorsk. Our model also improves the mBERT performance for other languages present in the corpus such as English, Swedish, and Danish. For languages not included in the corpus, the weights degrade moderately while keeping strong multilingual properties. Therefore, we show that building high-quality models within a memory institution using somewhat noisy optical character recognition (OCR) content is feasible, and we hope to pave the way for other memory institutions to follow.},
      address = {Reykjavik, Iceland (Online)},
      author = {Kummervold, Per E  and
    De la Rosa, Javier  and
    Wetjen, Freddy  and
    Brygfjeld, Svein Arne},
      booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
      editor = {Dobnik, Simon  and
    {\O}vrelid, Lilja},
      month = may # { 31--2 } # jun,
      pages = {20--29},
      publisher = {Link{\"o}ping University Electronic Press, Sweden},
      title = {Operationalizing a National Digital Library: The Case for a {N}orwegian Transformer Model},
      url = {https://aclanthology.org/2021.nodalida-main.3},
      year = {2021},
    }

    ```




#### NorwegianParliamentClassification.v2

Norwegian parliament speeches annotated for sentiment
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/norwegian_parliament`](https://huggingface.co/datasets/mteb/norwegian_parliament) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/NbAiLab/norwegian_parliament)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | nob | Government, Spoken | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{kummervold-etal-2021-operationalizing,
      abstract = {In this work, we show the process of building a large-scale training set from digital and digitized collections at a national library. The resulting Bidirectional Encoder Representations from Transformers (BERT)-based language model for Norwegian outperforms multilingual BERT (mBERT) models in several token and sequence classification tasks for both Norwegian Bokm{\aa}l and Norwegian Nynorsk. Our model also improves the mBERT performance for other languages present in the corpus such as English, Swedish, and Danish. For languages not included in the corpus, the weights degrade moderately while keeping strong multilingual properties. Therefore, we show that building high-quality models within a memory institution using somewhat noisy optical character recognition (OCR) content is feasible, and we hope to pave the way for other memory institutions to follow.},
      address = {Reykjavik, Iceland (Online)},
      author = {Kummervold, Per E  and
    De la Rosa, Javier  and
    Wetjen, Freddy  and
    Brygfjeld, Svein Arne},
      booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
      editor = {Dobnik, Simon  and
    {\O}vrelid, Lilja},
      month = may # { 31--2 } # jun,
      pages = {20--29},
      publisher = {Link{\"o}ping University Electronic Press, Sweden},
      title = {Operationalizing a National Digital Library: The Case for a {N}orwegian Transformer Model},
      url = {https://aclanthology.org/2021.nodalida-main.3},
      year = {2021},
    }

    ```




#### NusaParagraphEmotionClassification

NusaParagraphEmotionClassification is a multi-class emotion classification on 10 Indonesian languages from the NusaParagraph dataset.

**Dataset:** [`gentaiscool/nusaparagraph_emot`](https://huggingface.co/datasets/gentaiscool/nusaparagraph_emot) • **License:** apache-2.0 • [Learn more →](https://github.com/IndoNLP/nusa-writes)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | bbc, bew, bug, jav, mad, ... (10) | Fiction, Non-fiction, Written | human-annotated | found |



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




#### NusaParagraphTopicClassification

NusaParagraphTopicClassification is a multi-class topic classification on 10 Indonesian languages.

**Dataset:** [`gentaiscool/nusaparagraph_topic`](https://huggingface.co/datasets/gentaiscool/nusaparagraph_topic) • **License:** apache-2.0 • [Learn more →](https://github.com/IndoNLP/nusa-writes)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | bbc, bew, bug, jav, mad, ... (10) | Fiction, Non-fiction, Written | human-annotated | found |



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




#### NusaX-senti

NusaX is a high-quality multilingual parallel corpus that covers 12 languages, Indonesian, English, and 10 Indonesian local languages, namely Acehnese, Balinese, Banjarese, Buginese, Madurese, Minangkabau, Javanese, Ngaju, Sundanese, and Toba Batak. NusaX-Senti is a 3-labels (positive, neutral, negative) sentiment analysis dataset for 10 Indonesian local languages + Indonesian and English.

**Dataset:** [`mteb/NusaX-senti`](https://huggingface.co/datasets/mteb/NusaX-senti) • **License:** cc-by-sa-4.0 • [Learn more →](https://arxiv.org/abs/2205.15960)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ace, ban, bbc, bjn, bug, ... (12) | Constructed, Reviews, Social, Web, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{winata2022nusax,
      archiveprefix = {arXiv},
      author = {Winata, Genta Indra and Aji, Alham Fikri and Cahyawijaya,
    Samuel and Mahendra, Rahmad and Koto, Fajri and Romadhony,
    Ade and Kurniawan, Kemal and Moeljadi, David and Prasojo,
    Radityo Eko and Fung, Pascale and Baldwin, Timothy and Lau,
    Jey Han and Sennrich, Rico and Ruder, Sebastian},
      eprint = {2205.15960},
      primaryclass = {cs.CL},
      title = {NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages},
      year = {2022},
    }

    ```




#### OPP115DataRetentionLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describes how long user information is stored.

**Dataset:** [`mteb/OPP115DataRetentionLegalBenchClassification`](https://huggingface.co/datasets/mteb/OPP115DataRetentionLegalBenchClassification) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @inproceedings{wilson2016creation,
      author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
      booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      pages = {1330--1340},
      title = {The creation and analysis of a website privacy policy corpus},
      year = {2016},
    }

    ```




#### OPP115DataSecurityLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describes how user information is protected.

**Dataset:** [`mteb/OPP115DataSecurityLegalBenchClassification`](https://huggingface.co/datasets/mteb/OPP115DataSecurityLegalBenchClassification) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @inproceedings{wilson2016creation,
      author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
      booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      pages = {1330--1340},
      title = {The creation and analysis of a website privacy policy corpus},
      year = {2016},
    }

    ```




#### OPP115DataSecurityLegalBenchClassification.v2

Given a clause from a privacy policy, classify if the clause describes how user information is protected.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/opp115_data_security_legal_bench`](https://huggingface.co/datasets/mteb/opp115_data_security_legal_bench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @inproceedings{wilson2016creation,
      author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
      booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      pages = {1330--1340},
      title = {The creation and analysis of a website privacy policy corpus},
      year = {2016},
    }

    ```




#### OPP115DoNotTrackLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describes if and how Do Not Track signals for online tracking and advertising are honored.

**Dataset:** [`mteb/OPP115DoNotTrackLegalBenchClassification`](https://huggingface.co/datasets/mteb/OPP115DoNotTrackLegalBenchClassification) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @inproceedings{wilson2016creation,
      author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
      booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      pages = {1330--1340},
      title = {The creation and analysis of a website privacy policy corpus},
      year = {2016},
    }

    ```




#### OPP115DoNotTrackLegalBenchClassification.v2

Given a clause from a privacy policy, classify if the clause describes if and how Do Not Track signals for online tracking and advertising are honored.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/opp115_do_not_track_legal_bench`](https://huggingface.co/datasets/mteb/opp115_do_not_track_legal_bench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @inproceedings{wilson2016creation,
      author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
      booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      pages = {1330--1340},
      title = {The creation and analysis of a website privacy policy corpus},
      year = {2016},
    }

    ```




#### OPP115FirstPartyCollectionUseLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describes how and why a service provider collects user information.

**Dataset:** [`mteb/OPP115FirstPartyCollectionUseLegalBenchClassification`](https://huggingface.co/datasets/mteb/OPP115FirstPartyCollectionUseLegalBenchClassification) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @inproceedings{wilson2016creation,
      author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
      booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      pages = {1330--1340},
      title = {The creation and analysis of a website privacy policy corpus},
      year = {2016},
    }

    ```




#### OPP115InternationalAndSpecificAudiencesLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describe practices that pertain only to a specific group of users (e.g., children, Europeans, or California residents).

**Dataset:** [`mteb/OPP115InternationalAndSpecificAudiencesLegalBenchClassification`](https://huggingface.co/datasets/mteb/OPP115InternationalAndSpecificAudiencesLegalBenchClassification) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @inproceedings{wilson2016creation,
      author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
      booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      pages = {1330--1340},
      title = {The creation and analysis of a website privacy policy corpus},
      year = {2016},
    }

    ```




#### OPP115PolicyChangeLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describes if and how users will be informed about changes to the privacy policy.

**Dataset:** [`mteb/OPP115PolicyChangeLegalBenchClassification`](https://huggingface.co/datasets/mteb/OPP115PolicyChangeLegalBenchClassification) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @inproceedings{wilson2016creation,
      author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
      booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      pages = {1330--1340},
      title = {The creation and analysis of a website privacy policy corpus},
      year = {2016},
    }

    ```




#### OPP115ThirdPartySharingCollectionLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describe how user information may be shared with or collected by third parties.

**Dataset:** [`mteb/OPP115ThirdPartySharingCollectionLegalBenchClassification`](https://huggingface.co/datasets/mteb/OPP115ThirdPartySharingCollectionLegalBenchClassification) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @inproceedings{wilson2016creation,
      author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
      booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      pages = {1330--1340},
      title = {The creation and analysis of a website privacy policy corpus},
      year = {2016},
    }

    ```




#### OPP115UserAccessEditAndDeletionLegalBenchClassification

Given a clause from a privacy policy, classify if the clause describes if and how users may access, edit, or delete their information.

**Dataset:** [`mteb/OPP115UserAccessEditAndDeletionLegalBenchClassification`](https://huggingface.co/datasets/mteb/OPP115UserAccessEditAndDeletionLegalBenchClassification) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @inproceedings{wilson2016creation,
      author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
      booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      pages = {1330--1340},
      title = {The creation and analysis of a website privacy policy corpus},
      year = {2016},
    }

    ```




#### OPP115UserChoiceControlLegalBenchClassification

Given a clause fro ma privacy policy, classify if the clause describes the choices and control options available to users.

**Dataset:** [`mteb/OPP115UserChoiceControlLegalBenchClassification`](https://huggingface.co/datasets/mteb/OPP115UserChoiceControlLegalBenchClassification) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @inproceedings{wilson2016creation,
      author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
      booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      pages = {1330--1340},
      title = {The creation and analysis of a website privacy policy corpus},
      year = {2016},
    }

    ```




#### OPP115UserChoiceControlLegalBenchClassification.v2

Given a clause fro ma privacy policy, classify if the clause describes the choices and control options available to users.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/opp115_user_choice_control_legal_bench`](https://huggingface.co/datasets/mteb/opp115_user_choice_control_legal_bench) • **License:** cc-by-nc-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @inproceedings{wilson2016creation,
      author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
      booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      pages = {1330--1340},
      title = {The creation and analysis of a website privacy policy corpus},
      year = {2016},
    }

    ```




#### OdiaNewsClassification

A Odia dataset for 3-class classification of Odia news articles

**Dataset:** [`mlexplorer008/odia_news_classification`](https://huggingface.co/datasets/mlexplorer008/odia_news_classification) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-odia)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ory | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{kunchukuttan2020indicnlpcorpus,
      author = {Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
      journal = {arXiv preprint arXiv:2005.00085},
      title = {AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
      year = {2020},
    }

    ```




#### OdiaNewsClassification.v2

A Odia dataset for 3-class classification of Odia news articles
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/odia_news`](https://huggingface.co/datasets/mteb/odia_news) • **License:** mit • [Learn more →](https://github.com/goru001/nlp-for-odia)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | ory | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{kunchukuttan2020indicnlpcorpus,
      author = {Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
      journal = {arXiv preprint arXiv:2005.00085},
      title = {AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
      year = {2020},
    }

    ```




#### OnlineShopping

Sentiment Analysis of User Reviews on Online Shopping Websites

**Dataset:** [`C-MTEB/OnlineShopping-classification`](https://huggingface.co/datasets/C-MTEB/OnlineShopping-classification) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @article{xiao2023c,
      author = {Xiao, Shitao and Liu, Zheng and Zhang, Peitian and Muennighof, Niklas},
      journal = {arXiv preprint arXiv:2309.07597},
      title = {C-pack: Packaged resources to advance general chinese embedding},
      year = {2023},
    }

    ```




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


**Dataset:** [`mteb/OralArgumentQuestionPurposeLegalBenchClassification`](https://huggingface.co/datasets/mteb/OralArgumentQuestionPurposeLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




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

    ```




#### OverrulingLegalBenchClassification

This task consists of classifying whether or not a particular sentence of case law overturns the decision of a previous case.

**Dataset:** [`mteb/OverrulingLegalBenchClassification`](https://huggingface.co/datasets/mteb/OverrulingLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @inproceedings{zheng2021does,
      author = {Zheng, Lucia and Guha, Neel and Anderson, Brandon R and Henderson, Peter and Ho, Daniel E},
      booktitle = {Proceedings of the eighteenth international conference on artificial intelligence and law},
      pages = {159--168},
      title = {When does pretraining help? assessing self-supervised learning for law and the casehold dataset of 53,000+ legal holdings},
      year = {2021},
    }

    ```




#### OverrulingLegalBenchClassification.v2

This task consists of classifying whether or not a particular sentence of case law overturns the decision of a previous case.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/overruling_legal_bench`](https://huggingface.co/datasets/mteb/overruling_legal_bench) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @inproceedings{zheng2021does,
      author = {Zheng, Lucia and Guha, Neel and Anderson, Brandon R and Henderson, Peter and Ho, Daniel E},
      booktitle = {Proceedings of the eighteenth international conference on artificial intelligence and law},
      pages = {159--168},
      title = {When does pretraining help? assessing self-supervised learning for law and the casehold dataset of 53,000+ legal holdings},
      year = {2021},
    }

    ```




#### PAC

Polish Paraphrase Corpus

**Dataset:** [`mteb/PAC`](https://huggingface.co/datasets/mteb/PAC) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2211.13112.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Legal, Written | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @misc{augustyniak2022waydesigningcompilinglepiszcze,
      archiveprefix = {arXiv},
      author = {Łukasz Augustyniak and Kamil Tagowski and Albert Sawczyn and Denis Janiak and Roman Bartusiak and Adrian Szymczak and Marcin Wątroba and Arkadiusz Janz and Piotr Szymański and Mikołaj Morzy and Tomasz Kajdanowicz and Maciej Piasecki},
      eprint = {2211.13112},
      primaryclass = {cs.CL},
      title = {This is the way: designing and compiling LEPISZCZE, a comprehensive NLP benchmark for Polish},
      url = {https://arxiv.org/abs/2211.13112},
      year = {2022},
    }

    ```




#### PAC.v2

Polish Paraphrase Corpus
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/pac`](https://huggingface.co/datasets/mteb/pac) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2211.13112.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Legal, Written | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @misc{augustyniak2022waydesigningcompilinglepiszcze,
      archiveprefix = {arXiv},
      author = {Łukasz Augustyniak and Kamil Tagowski and Albert Sawczyn and Denis Janiak and Roman Bartusiak and Adrian Szymczak and Marcin Wątroba and Arkadiusz Janz and Piotr Szymański and Mikołaj Morzy and Tomasz Kajdanowicz and Maciej Piasecki},
      eprint = {2211.13112},
      primaryclass = {cs.CL},
      title = {This is the way: designing and compiling LEPISZCZE, a comprehensive NLP benchmark for Polish},
      url = {https://arxiv.org/abs/2211.13112},
      year = {2022},
    }

    ```




#### PROALegalBenchClassification

Given a statute, determine if the text contains an explicit private right of action. Given a privacy policy clause and a description of the clause, determine if the description is correct. A private right of action (PROA) exists when a statute empowers an ordinary individual (i.e., a private person) to legally enforce their rights by bringing an action in court. In short, a PROA creates the ability for an individual to sue someone in order to recover damages or halt some offending conduct. PROAs are ubiquitous in antitrust law (in which individuals harmed by anti-competitive behavior can sue offending firms for compensation) and environmental law (in which individuals can sue entities which release hazardous substances for damages).

**Dataset:** [`mteb/PROALegalBenchClassification`](https://huggingface.co/datasets/mteb/PROALegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### PatentClassification

Classification Dataset of Patents and Abstract

**Dataset:** [`mteb/PatentClassification`](https://huggingface.co/datasets/mteb/PatentClassification) • **License:** not specified • [Learn more →](https://aclanthology.org/P19-1212.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{sharma-etal-2019-bigpatent,
      abstract = {Most existing text summarization datasets are compiled from the news domain, where summaries have a flattened discourse structure. In such datasets, summary-worthy content often appears in the beginning of input articles. Moreover, large segments from input articles are present verbatim in their respective summaries. These issues impede the learning and evaluation of systems that can understand an article{'}s global content structure as well as produce abstractive summaries with high compression ratio. In this work, we present a novel dataset, BIGPATENT, consisting of 1.3 million records of U.S. patent documents along with human written abstractive summaries. Compared to existing summarization datasets, BIGPATENT has the following properties: i) summaries contain a richer discourse structure with more recurring entities, ii) salient content is evenly distributed in the input, and iii) lesser and shorter extractive fragments are present in the summaries. Finally, we train and evaluate baselines and popular learning models on BIGPATENT to shed light on new challenges and motivate future directions for summarization research.},
      address = {Florence, Italy},
      author = {Sharma, Eva  and
    Li, Chen  and
    Wang, Lu},
      booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
      doi = {10.18653/v1/P19-1212},
      editor = {Korhonen, Anna  and
    Traum, David  and
    M{\`a}rquez, Llu{\'\i}s},
      month = jul,
      pages = {2204--2213},
      publisher = {Association for Computational Linguistics},
      title = {{BIGPATENT}: A Large-Scale Dataset for Abstractive and Coherent Summarization},
      url = {https://aclanthology.org/P19-1212},
      year = {2019},
    }

    ```




#### PatentClassification.v2

Classification Dataset of Patents and Abstract
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/patent`](https://huggingface.co/datasets/mteb/patent) • **License:** not specified • [Learn more →](https://aclanthology.org/P19-1212.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{sharma-etal-2019-bigpatent,
      abstract = {Most existing text summarization datasets are compiled from the news domain, where summaries have a flattened discourse structure. In such datasets, summary-worthy content often appears in the beginning of input articles. Moreover, large segments from input articles are present verbatim in their respective summaries. These issues impede the learning and evaluation of systems that can understand an article{'}s global content structure as well as produce abstractive summaries with high compression ratio. In this work, we present a novel dataset, BIGPATENT, consisting of 1.3 million records of U.S. patent documents along with human written abstractive summaries. Compared to existing summarization datasets, BIGPATENT has the following properties: i) summaries contain a richer discourse structure with more recurring entities, ii) salient content is evenly distributed in the input, and iii) lesser and shorter extractive fragments are present in the summaries. Finally, we train and evaluate baselines and popular learning models on BIGPATENT to shed light on new challenges and motivate future directions for summarization research.},
      address = {Florence, Italy},
      author = {Sharma, Eva  and
    Li, Chen  and
    Wang, Lu},
      booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
      doi = {10.18653/v1/P19-1212},
      editor = {Korhonen, Anna  and
    Traum, David  and
    M{\`a}rquez, Llu{\'\i}s},
      month = jul,
      pages = {2204--2213},
      publisher = {Association for Computational Linguistics},
      title = {{BIGPATENT}: A Large-Scale Dataset for Abstractive and Coherent Summarization},
      url = {https://aclanthology.org/P19-1212},
      year = {2019},
    }

    ```




#### PerShopDomainClassification

PerSHOP - A Persian dataset for shopping dialogue systems modeling

**Dataset:** [`MCINext/pershop-classification`](https://huggingface.co/datasets/MCINext/pershop-classification) • **License:** not specified • [Learn more →](https://github.com/keyvanmahmoudi/PerSHOP)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | fas | Spoken | human-annotated | created |



??? quote "Citation"


    ```bibtex
    @article{mahmoudi2024pershop,
      author = {Mahmoudi, Keyvan and Faili, Heshaam},
      journal = {arXiv preprint arXiv:2401.00811},
      title = {PerSHOP--A Persian dataset for shopping dialogue systems modeling},
      year = {2024},
    }
    ```




#### PerShopIntentClassification

PerSHOP - A Persian dataset for shopping dialogue systems modeling

**Dataset:** [`MCINext/pershop-classification`](https://huggingface.co/datasets/MCINext/pershop-classification) • **License:** not specified • [Learn more →](https://github.com/keyvanmahmoudi/PerSHOP)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | fas | Spoken | human-annotated | created |



??? quote "Citation"


    ```bibtex
    @article{mahmoudi2024pershop,
      author = {Mahmoudi, Keyvan and Faili, Heshaam},
      journal = {arXiv preprint arXiv:2401.00811},
      title = {PerSHOP--A Persian dataset for shopping dialogue systems modeling},
      year = {2024},
    }
    ```




#### PersianFoodSentimentClassification

Persian Food Review Dataset

**Dataset:** [`asparius/Persian-Food-Sentiment`](https://huggingface.co/datasets/asparius/Persian-Food-Sentiment) • **License:** not specified • [Learn more →](https://hooshvare.github.io/docs/datasets/sa)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{ParsBERT,
      author = {Mehrdad Farahani, Mohammad Gharachorloo, Marzieh Farahani, Mohammad Manthouri},
      journal = {ArXiv},
      title = {ParsBERT: Transformer-based Model for Persian Language Understanding},
      volume = {abs/2005.12515},
      year = {2020},
    }

    ```




#### PersianTextEmotion

Emotion is a Persian dataset with six basic emotions: anger, fear, joy, love, sadness, and surprise.

**Dataset:** [`SeyedAli/Persian-Text-Emotion`](https://huggingface.co/datasets/SeyedAli/Persian-Text-Emotion) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/SeyedAli/Persian-Text-Emotion)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | not specified | derived | found |



??? quote "Citation"


    ```bibtex

    ```




#### PersianTextEmotion.v2

Emotion is a Persian dataset with six basic emotions: anger, fear, joy, love, sadness, and surprise.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/persian_text_emotion`](https://huggingface.co/datasets/mteb/persian_text_emotion) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/SeyedAli/Persian-Text-Emotion)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | not specified | derived | found |



??? quote "Citation"


    ```bibtex

    ```




#### PersonalJurisdictionLegalBenchClassification

Given a fact pattern describing the set of contacts between a plaintiff, defendant, and forum, determine if a court in that forum could excercise personal jurisdiction over the defendant.

**Dataset:** [`mteb/PersonalJurisdictionLegalBenchClassification`](https://huggingface.co/datasets/mteb/PersonalJurisdictionLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### PoemSentimentClassification

Poem Sentiment is a sentiment dataset of poem verses from Project Gutenberg.

**Dataset:** [`mteb/PoemSentimentClassification`](https://huggingface.co/datasets/mteb/PoemSentimentClassification) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2011.02686)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{sheng2020investigating,
      archiveprefix = {arXiv},
      author = {Emily Sheng and David Uthus},
      eprint = {2011.02686},
      primaryclass = {cs.CL},
      title = {Investigating Societal Biases in a Poetry Composition System},
      year = {2020},
    }

    ```




#### PoemSentimentClassification.v2

Poem Sentiment is a sentiment dataset of poem verses from Project Gutenberg.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/poem_sentiment`](https://huggingface.co/datasets/mteb/poem_sentiment) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2011.02686)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{sheng2020investigating,
      archiveprefix = {arXiv},
      author = {Emily Sheng and David Uthus},
      eprint = {2011.02686},
      primaryclass = {cs.CL},
      title = {Investigating Societal Biases in a Poetry Composition System},
      year = {2020},
    }

    ```




#### PolEmo2.0-IN

A collection of Polish online reviews from four domains: medicine, hotels, products and school. The PolEmo2.0-IN task is to predict the sentiment of in-domain (medicine and hotels) reviews.

**Dataset:** [`PL-MTEB/polemo2_in`](https://huggingface.co/datasets/PL-MTEB/polemo2_in) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/K19-1092.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Social, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{kocon-etal-2019-multi,
      abstract = {In this article we present an extended version of PolEmo {--} a corpus of consumer reviews from 4 domains: medicine, hotels, products and school. Current version (PolEmo 2.0) contains 8,216 reviews having 57,466 sentences. Each text and sentence was manually annotated with sentiment in 2+1 scheme, which gives a total of 197,046 annotations. We obtained a high value of Positive Specific Agreement, which is 0.91 for texts and 0.88 for sentences. PolEmo 2.0 is publicly available under a Creative Commons copyright license. We explored recent deep learning approaches for the recognition of sentiment, such as Bi-directional Long Short-Term Memory (BiLSTM) and Bidirectional Encoder Representations from Transformers (BERT).},
      address = {Hong Kong, China},
      author = {Koco{\'n}, Jan  and
    Mi{\l}kowski, Piotr  and
    Za{\'s}ko-Zieli{\'n}ska, Monika},
      booktitle = {Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)},
      doi = {10.18653/v1/K19-1092},
      month = nov,
      pages = {980--991},
      publisher = {Association for Computational Linguistics},
      title = {Multi-Level Sentiment Analysis of {P}ol{E}mo 2.0: Extended Corpus of Multi-Domain Consumer Reviews},
      url = {https://aclanthology.org/K19-1092},
      year = {2019},
    }

    ```




#### PolEmo2.0-IN.v2

A collection of Polish online reviews from four domains: medicine, hotels, products and school. The PolEmo2.0-IN task is to predict the sentiment of in-domain (medicine and hotels) reviews.

**Dataset:** [`mteb/pol_emo2_in`](https://huggingface.co/datasets/mteb/pol_emo2_in) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/K19-1092.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | pol | Social, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{kocon-etal-2019-multi,
      abstract = {In this article we present an extended version of PolEmo {--} a corpus of consumer reviews from 4 domains: medicine, hotels, products and school. Current version (PolEmo 2.0) contains 8,216 reviews having 57,466 sentences. Each text and sentence was manually annotated with sentiment in 2+1 scheme, which gives a total of 197,046 annotations. We obtained a high value of Positive Specific Agreement, which is 0.91 for texts and 0.88 for sentences. PolEmo 2.0 is publicly available under a Creative Commons copyright license. We explored recent deep learning approaches for the recognition of sentiment, such as Bi-directional Long Short-Term Memory (BiLSTM) and Bidirectional Encoder Representations from Transformers (BERT).},
      address = {Hong Kong, China},
      author = {Koco{\'n}, Jan  and
    Mi{\l}kowski, Piotr  and
    Za{\'s}ko-Zieli{\'n}ska, Monika},
      booktitle = {Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)},
      doi = {10.18653/v1/K19-1092},
      month = nov,
      pages = {980--991},
      publisher = {Association for Computational Linguistics},
      title = {Multi-Level Sentiment Analysis of {P}ol{E}mo 2.0: Extended Corpus of Multi-Domain Consumer Reviews},
      url = {https://aclanthology.org/K19-1092},
      year = {2019},
    }

    ```




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



??? quote "Citation"


    ```bibtex

    @article{kunchukuttan2020indicnlpcorpus,
      author = {Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
      journal = {arXiv preprint arXiv:2005.00085},
      title = {AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
      year = {2020},
    }

    ```




#### RestaurantReviewSentimentClassification

Dataset of 8364 restaurant reviews from qaym.com in Arabic for sentiment analysis

**Dataset:** [`hadyelsahar/ar_res_reviews`](https://huggingface.co/datasets/hadyelsahar/ar_res_reviews) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-18117-2_2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{elsahar2015building,
      author = {ElSahar, Hady and El-Beltagy, Samhaa R},
      booktitle = {International conference on intelligent text processing and computational linguistics},
      organization = {Springer},
      pages = {23--34},
      title = {Building large arabic multi-domain resources for sentiment analysis},
      year = {2015},
    }

    ```




#### RestaurantReviewSentimentClassification.v2

Dataset of 8156 restaurant reviews from qaym.com in Arabic for sentiment analysis
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/restaurant_review_sentiment`](https://huggingface.co/datasets/mteb/restaurant_review_sentiment) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-18117-2_2)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{elsahar2015building,
      author = {ElSahar, Hady and El-Beltagy, Samhaa R},
      booktitle = {International conference on intelligent text processing and computational linguistics},
      organization = {Springer},
      pages = {23--34},
      title = {Building large arabic multi-domain resources for sentiment analysis},
      year = {2015},
    }

    ```




#### RomanianReviewsSentiment

LaRoSeDa (A Large Romanian Sentiment Data Set) contains 15,000 reviews written in Romanian

**Dataset:** [`mteb/RomanianReviewsSentiment`](https://huggingface.co/datasets/mteb/RomanianReviewsSentiment) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2101.04197)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ron | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{tache2101clustering,
      author = {Anca Maria Tache and Mihaela Gaman and Radu Tudor Ionescu},
      journal = {ArXiv},
      title = {Clustering Word Embeddings with Self-Organizing Maps. Application on LaRoSeDa -- A Large Romanian Sentiment Data Set},
      year = {2021},
    }

    ```




#### RomanianReviewsSentiment.v2

LaRoSeDa (A Large Romanian Sentiment Data Set) contains 15,000 reviews written in Romanian
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/romanian_reviews_sentiment`](https://huggingface.co/datasets/mteb/romanian_reviews_sentiment) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/2101.04197)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ron | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{tache2101clustering,
      author = {Anca Maria Tache and Mihaela Gaman and Radu Tudor Ionescu},
      journal = {ArXiv},
      title = {Clustering Word Embeddings with Self-Organizing Maps. Application on LaRoSeDa -- A Large Romanian Sentiment Data Set},
      year = {2021},
    }

    ```




#### RomanianSentimentClassification

An Romanian dataset for sentiment classification.

**Dataset:** [`mteb/RomanianSentimentClassification`](https://huggingface.co/datasets/mteb/RomanianSentimentClassification) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2009.08712)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ron | Reviews, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{dumitrescu2020birth,
      author = {Dumitrescu, Stefan Daniel and Avram, Andrei-Marius and Pyysalo, Sampo},
      journal = {arXiv preprint arXiv:2009.08712},
      title = {The birth of Romanian BERT},
      year = {2020},
    }

    ```




#### RomanianSentimentClassification.v2

An Romanian dataset for sentiment classification.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/romanian_sentiment`](https://huggingface.co/datasets/mteb/romanian_sentiment) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2009.08712)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ron | Reviews, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{dumitrescu2020birth,
      author = {Dumitrescu, Stefan Daniel and Avram, Andrei-Marius and Pyysalo, Sampo},
      journal = {arXiv preprint arXiv:2009.08712},
      title = {The birth of Romanian BERT},
      year = {2020},
    }

    ```




#### RuNLUIntentClassification

Contains natural language data for human-robot interaction in home domain which we collected and annotated for evaluating NLU Services/platforms.

**Dataset:** [`mteb/RuNLUIntentClassification`](https://huggingface.co/datasets/mteb/RuNLUIntentClassification) • **License:** cc-by-4.0 • [Learn more →](https://arxiv.org/abs/1903.05566)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | rus | not specified | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{liu2019benchmarkingnaturallanguageunderstanding,
      archiveprefix = {arXiv},
      author = {Xingkun Liu and Arash Eshghi and Pawel Swietojanski and Verena Rieser},
      eprint = {1903.05566},
      primaryclass = {cs.CL},
      title = {Benchmarking Natural Language Understanding Services for building Conversational Agents},
      url = {https://arxiv.org/abs/1903.05566},
      year = {2019},
    }

    ```




#### RuReviewsClassification

Product review classification (3-point scale) based on RuRevies dataset

**Dataset:** [`ai-forever/ru-reviews-classification`](https://huggingface.co/datasets/ai-forever/ru-reviews-classification) • **License:** apache-2.0 • [Learn more →](https://github.com/sismetanin/rureviews)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{Smetanin-SA-2019,
      author = {Sergey Smetanin and Michail Komarov},
      booktitle = {2019 IEEE 21st Conference on Business Informatics (CBI)},
      doi = {10.1109/CBI.2019.00062},
      issn = {2378-1963},
      month = {July},
      number = {},
      pages = {482-486},
      title = {Sentiment Analysis of Product Reviews in Russian using Convolutional Neural Networks},
      volume = {01},
      year = {2019},
    }

    ```




#### RuReviewsClassification.v2

Product review classification (3-point scale) based on RuRevies dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/ru_reviews`](https://huggingface.co/datasets/mteb/ru_reviews) • **License:** apache-2.0 • [Learn more →](https://github.com/sismetanin/rureviews)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | rus | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{Smetanin-SA-2019,
      author = {Sergey Smetanin and Michail Komarov},
      booktitle = {2019 IEEE 21st Conference on Business Informatics (CBI)},
      doi = {10.1109/CBI.2019.00062},
      issn = {2378-1963},
      month = {July},
      number = {},
      pages = {482-486},
      title = {Sentiment Analysis of Product Reviews in Russian using Convolutional Neural Networks},
      volume = {01},
      year = {2019},
    }

    ```




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




#### RuSciBenchPubTypeClassification

This task involves classifying scientific papers (based on their title and abstract)
        into different publication types. The dataset identifies the following types:
        'Article', 'Conference proceedings', 'Survey', 'Miscellanea', 'Short message', 'Review', and 'Personalia'.
        This task is available for both Russian and English versions of the paper's title and abstract.

**Dataset:** [`mlsa-iai-msu-lab/ru_sci_bench_mteb`](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench_mteb) • **License:** mit • [Learn more →](https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng, rus | Academic, Non-fiction, Written | derived | found |



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

**Dataset:** [`mteb/SCDBPAccountabilityLegalBenchClassification`](https://huggingface.co/datasets/mteb/SCDBPAccountabilityLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{chilton2017limitations,
      author = {Chilton, Adam S and Sarfaty, Galit A},
      journal = {Stan. J. Int'l L.},
      pages = {1},
      publisher = {HeinOnline},
      title = {The limitations of supply chain disclosure regimes},
      volume = {53},
      year = {2017},
    }

    @misc{guha2023legalbench,
      archiveprefix = {arXiv},
      author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
      eprint = {2308.11462},
      primaryclass = {cs.CL},
      title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
      year = {2023},
    }

    ```




#### SCDBPAuditsLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer  performs any type of audit, or reserves the right to audit?'

**Dataset:** [`mteb/SCDBPAuditsLegalBenchClassification`](https://huggingface.co/datasets/mteb/SCDBPAuditsLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{chilton2017limitations,
      author = {Chilton, Adam S and Sarfaty, Galit A},
      journal = {Stan. J. Int'l L.},
      pages = {1},
      publisher = {HeinOnline},
      title = {The limitations of supply chain disclosure regimes},
      volume = {53},
      year = {2017},
    }

    @misc{guha2023legalbench,
      archiveprefix = {arXiv},
      author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
      eprint = {2308.11462},
      primaryclass = {cs.CL},
      title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
      year = {2023},
    }

    ```




#### SCDBPCertificationLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer  performs any type of audit, or reserves the right to audit?'

**Dataset:** [`mteb/SCDBPCertificationLegalBenchClassification`](https://huggingface.co/datasets/mteb/SCDBPCertificationLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{chilton2017limitations,
      author = {Chilton, Adam S and Sarfaty, Galit A},
      journal = {Stan. J. Int'l L.},
      pages = {1},
      publisher = {HeinOnline},
      title = {The limitations of supply chain disclosure regimes},
      volume = {53},
      year = {2017},
    }

    @misc{guha2023legalbench,
      archiveprefix = {arXiv},
      author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
      eprint = {2308.11462},
      primaryclass = {cs.CL},
      title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
      year = {2023},
    }

    ```




#### SCDBPTrainingLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer  provides training to employees on human trafficking and slavery? Broad policies such as ongoing dialogue on mitigating risks of human trafficking and slavery or increasing managers and purchasers knowledge about health, safety and labor practices qualify as training. Providing training to contractors who failed to comply with human trafficking laws counts as training.'

**Dataset:** [`mteb/SCDBPTrainingLegalBenchClassification`](https://huggingface.co/datasets/mteb/SCDBPTrainingLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{chilton2017limitations,
      author = {Chilton, Adam S and Sarfaty, Galit A},
      journal = {Stan. J. Int'l L.},
      pages = {1},
      publisher = {HeinOnline},
      title = {The limitations of supply chain disclosure regimes},
      volume = {53},
      year = {2017},
    }

    @misc{guha2023legalbench,
      archiveprefix = {arXiv},
      author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
      eprint = {2308.11462},
      primaryclass = {cs.CL},
      title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
      year = {2023},
    }

    ```




#### SCDBPVerificationLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer engages in verification and auditing as one practice, expresses that it may conduct an audit, or expressess that it is assessing supplier risks through a review of the US Dept. of Labor's List?'

**Dataset:** [`mteb/SCDBPVerificationLegalBenchClassification`](https://huggingface.co/datasets/mteb/SCDBPVerificationLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{chilton2017limitations,
      author = {Chilton, Adam S and Sarfaty, Galit A},
      journal = {Stan. J. Int'l L.},
      pages = {1},
      publisher = {HeinOnline},
      title = {The limitations of supply chain disclosure regimes},
      volume = {53},
      year = {2017},
    }

    @misc{guha2023legalbench,
      archiveprefix = {arXiv},
      author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
      eprint = {2308.11462},
      primaryclass = {cs.CL},
      title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
      year = {2023},
    }

    ```




#### SCDDAccountabilityLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose to what extent, if any, that the retail seller or manufacturer maintains internal accountability standards and procedures for employees or contractors failing to meet company standards regarding slavery and trafficking?'

**Dataset:** [`mteb/SCDDAccountabilityLegalBenchClassification`](https://huggingface.co/datasets/mteb/SCDDAccountabilityLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{chilton2017limitations,
      author = {Chilton, Adam S and Sarfaty, Galit A},
      journal = {Stan. J. Int'l L.},
      pages = {1},
      publisher = {HeinOnline},
      title = {The limitations of supply chain disclosure regimes},
      volume = {53},
      year = {2017},
    }

    @misc{guha2023legalbench,
      archiveprefix = {arXiv},
      author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
      eprint = {2308.11462},
      primaryclass = {cs.CL},
      title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
      year = {2023},
    }

    ```




#### SCDDAuditsLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose to what extent, if any, that the retail seller or manufacturer conducts audits of suppliers to evaluate supplier compliance with company standards for trafficking and slavery in supply chains? The disclosure shall specify if the verification was not an independent, unannounced audit.'

**Dataset:** [`mteb/SCDDAuditsLegalBenchClassification`](https://huggingface.co/datasets/mteb/SCDDAuditsLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{chilton2017limitations,
      author = {Chilton, Adam S and Sarfaty, Galit A},
      journal = {Stan. J. Int'l L.},
      pages = {1},
      publisher = {HeinOnline},
      title = {The limitations of supply chain disclosure regimes},
      volume = {53},
      year = {2017},
    }

    @misc{guha2023legalbench,
      archiveprefix = {arXiv},
      author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
      eprint = {2308.11462},
      primaryclass = {cs.CL},
      title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
      year = {2023},
    }

    ```




#### SCDDCertificationLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose to what extent, if any, that the retail seller or manufacturer requires direct suppliers to certify that materials incorporated into the product comply with the laws regarding slavery and human trafficking of the country or countries in which they are doing business?'

**Dataset:** [`mteb/SCDDCertificationLegalBenchClassification`](https://huggingface.co/datasets/mteb/SCDDCertificationLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{chilton2017limitations,
      author = {Chilton, Adam S and Sarfaty, Galit A},
      journal = {Stan. J. Int'l L.},
      pages = {1},
      publisher = {HeinOnline},
      title = {The limitations of supply chain disclosure regimes},
      volume = {53},
      year = {2017},
    }

    @misc{guha2023legalbench,
      archiveprefix = {arXiv},
      author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
      eprint = {2308.11462},
      primaryclass = {cs.CL},
      title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
      year = {2023},
    }

    ```




#### SCDDTrainingLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose to what extent, if any, that the retail seller or manufacturer provides company employees and management, who have direct responsibility for supply chain management, training on human trafficking and slavery, particularly with respect to mitigating risks within the supply chains of products?'

**Dataset:** [`mteb/SCDDTrainingLegalBenchClassification`](https://huggingface.co/datasets/mteb/SCDDTrainingLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{chilton2017limitations,
      author = {Chilton, Adam S and Sarfaty, Galit A},
      journal = {Stan. J. Int'l L.},
      pages = {1},
      publisher = {HeinOnline},
      title = {The limitations of supply chain disclosure regimes},
      volume = {53},
      year = {2017},
    }

    @misc{guha2023legalbench,
      archiveprefix = {arXiv},
      author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
      eprint = {2308.11462},
      primaryclass = {cs.CL},
      title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
      year = {2023},
    }

    ```




#### SCDDVerificationLegalBenchClassification

This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose to what extent, if any, that the retail seller or manufacturer engages in verification of product supply chains to evaluate and address risks of human trafficking and slavery? If the company conducts verification], the disclosure shall specify if the verification was not conducted by a third party.'

**Dataset:** [`mteb/SCDDVerificationLegalBenchClassification`](https://huggingface.co/datasets/mteb/SCDDVerificationLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{chilton2017limitations,
      author = {Chilton, Adam S and Sarfaty, Galit A},
      journal = {Stan. J. Int'l L.},
      pages = {1},
      publisher = {HeinOnline},
      title = {The limitations of supply chain disclosure regimes},
      volume = {53},
      year = {2017},
    }

    @misc{guha2023legalbench,
      archiveprefix = {arXiv},
      author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
      eprint = {2308.11462},
      primaryclass = {cs.CL},
      title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
      year = {2023},
    }

    ```




#### SDSEyeProtectionClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/SDSEyeProtectionClassification`](https://huggingface.co/datasets/BASF-AI/SDSEyeProtectionClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | LM-generated and reviewed | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    @inproceedings{pereira2020msds,
      author = {Pereira, Eliseu},
      booktitle = {15th Doctoral Symposium},
      pages = {42},
      title = {MSDS-OPP: Operator Procedures Prediction in Material Safety Data Sheets},
      year = {2020},
    }

    ```




#### SDSEyeProtectionClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sds_eye_protection`](https://huggingface.co/datasets/mteb/sds_eye_protection) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | LM-generated and reviewed | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    @inproceedings{pereira2020msds,
      author = {Pereira, Eliseu},
      booktitle = {15th Doctoral Symposium},
      pages = {42},
      title = {MSDS-OPP: Operator Procedures Prediction in Material Safety Data Sheets},
      year = {2020},
    }

    ```




#### SDSGlovesClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/SDSGlovesClassification`](https://huggingface.co/datasets/BASF-AI/SDSGlovesClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | LM-generated and reviewed | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    @inproceedings{pereira2020msds,
      author = {Pereira, Eliseu},
      booktitle = {15th Doctoral Symposium},
      pages = {42},
      title = {MSDS-OPP: Operator Procedures Prediction in Material Safety Data Sheets},
      year = {2020},
    }

    ```




#### SDSGlovesClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sds_gloves`](https://huggingface.co/datasets/mteb/sds_gloves) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | LM-generated and reviewed | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    @inproceedings{pereira2020msds,
      author = {Pereira, Eliseu},
      booktitle = {15th Doctoral Symposium},
      pages = {42},
      title = {MSDS-OPP: Operator Procedures Prediction in Material Safety Data Sheets},
      year = {2020},
    }

    ```




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



??? quote "Citation"


    ```bibtex

    @article{adelani2023sib,
      author = {Adelani, David Ifeoluwa and Liu, Hannah and Shen, Xiaoyu and Vassilyev, Nikita and Alabi, Jesujoba O and Mao, Yanke and Gao, Haonan and Lee, Annie En-Shiun},
      journal = {arXiv preprint arXiv:2309.07445},
      title = {SIB-200: A simple, inclusive, and big evaluation dataset for topic classification in 200+ languages and dialects},
      year = {2023},
    }

    ```




#### SIDClassification

SID Classification

**Dataset:** [`MCINext/sid-classification`](https://huggingface.co/datasets/MCINext/sid-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Academic | derived | found |



??? quote "Citation"


    ```bibtex

    ```




#### SIDClassification.v2

SID Classification
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sid`](https://huggingface.co/datasets/mteb/sid) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Academic | derived | found |



??? quote "Citation"


    ```bibtex

    ```




#### SanskritShlokasClassification

This data set contains ~500 Shlokas

**Dataset:** [`bpHigh/iNLTK_Sanskrit_Shlokas_Dataset`](https://huggingface.co/datasets/bpHigh/iNLTK_Sanskrit_Shlokas_Dataset) • **License:** cc-by-sa-4.0 • [Learn more →](https://github.com/goru001/nlp-for-sanskrit)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | san | Religious, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{arora-2020-inltk,
      abstract = {We present iNLTK, an open-source NLP library consisting of pre-trained language models and out-of-the-box support for Data Augmentation, Textual Similarity, Sentence Embeddings, Word Embeddings, Tokenization and Text Generation in 13 Indic Languages. By using pre-trained models from iNLTK for text classification on publicly available datasets, we significantly outperform previously reported results. On these datasets, we also show that by using pre-trained models and data augmentation from iNLTK, we can achieve more than 95{\%} of the previous best performance by using less than 10{\%} of the training data. iNLTK is already being widely used by the community and has 40,000+ downloads, 600+ stars and 100+ forks on GitHub.},
      address = {Online},
      author = {Arora, Gaurav},
      booktitle = {Proceedings of Second Workshop for NLP Open Source Software (NLP-OSS)},
      doi = {10.18653/v1/2020.nlposs-1.10},
      editor = {Park, Eunjeong L.  and
    Hagiwara, Masato  and
    Milajevs, Dmitrijs  and
    Liu, Nelson F.  and
    Chauhan, Geeticka  and
    Tan, Liling},
      month = nov,
      pages = {66--71},
      publisher = {Association for Computational Linguistics},
      title = {i{NLTK}: Natural Language Toolkit for Indic Languages},
      url = {https://aclanthology.org/2020.nlposs-1.10},
      year = {2020},
    }

    ```




#### SardiStanceClassification

SardiStance is a unique dataset designed for the task of stance detection in Italian tweets. It consists of tweets related to the Sardines movement, providing a valuable resource for researchers and practitioners in the field of NLP.

**Dataset:** [`MattiaSangermano/SardiStance`](https://huggingface.co/datasets/MattiaSangermano/SardiStance) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://github.com/mirkolai/evalita-sardistance)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ita | Social | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{cignarella2020sardistance,
      author = {Cignarella, Alessandra Teresa and Lai, Mirko and Bosco, Cristina and Patti, Viviana and Rosso, Paolo and others},
      booktitle = {CEUR WORKSHOP PROCEEDINGS},
      organization = {Ceur},
      pages = {1--10},
      title = {Sardistance@ evalita2020: Overview of the task on stance detection in italian tweets},
      year = {2020},
    }

    ```




#### ScalaClassification

ScaLa a linguistic acceptability dataset for the mainland Scandinavian languages automatically constructed from dependency annotations in Universal Dependencies Treebanks.
        Published as part of 'ScandEval: A Benchmark for Scandinavian Natural Language Processing'

**Dataset:** [`mteb/multilingual-scala-classification`](https://huggingface.co/datasets/mteb/multilingual-scala-classification) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan, nno, nob, swe | Blog, Fiction, News, Non-fiction, Spoken, ... (7) | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{nielsen-2023-scandeval,
      address = {T{\'o}rshavn, Faroe Islands},
      author = {Nielsen, Dan},
      booktitle = {Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)},
      editor = {Alum{\"a}e, Tanel  and
    Fishel, Mark},
      month = may,
      pages = {185--201},
      publisher = {University of Tartu Library},
      title = {{S}cand{E}val: A Benchmark for {S}candinavian Natural Language Processing},
      url = {https://aclanthology.org/2023.nodalida-1.20},
      year = {2023},
    }

    ```




#### ScandiSentClassification

The corpus is crawled from se.trustpilot.com, no.trustpilot.com, dk.trustpilot.com, fi.trustpilot.com and trustpilot.com.

**Dataset:** [`mteb/scandisent`](https://huggingface.co/datasets/mteb/scandisent) • **License:** openrail • [Learn more →](https://github.com/timpal0l/ScandiSent)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | dan, eng, fin, nob, swe | Reviews, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{isbister-etal-2021-stop,
      address = {Reykjavik, Iceland (Online)},
      author = {Isbister, Tim  and
    Carlsson, Fredrik  and
    Sahlgren, Magnus},
      booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
      editor = {Dobnik, Simon  and
    {\O}vrelid, Lilja},
      month = may # { 31--2 } # jun,
      pages = {385--390},
      publisher = {Link{\"o}ping University Electronic Press, Sweden},
      title = {Should we Stop Training More Monolingual Models, and Simply Use Machine Translation Instead?},
      url = {https://aclanthology.org/2021.nodalida-main.42/},
      year = {2021},
    }

    ```




#### SentiRuEval2016

Russian sentiment analysis evaluation SentiRuEval-2016 devoted to reputation monitoring of banks and telecom companies in Twitter. We describe the task, data, the procedure of data preparation, and participants’ results.

**Dataset:** [`mteb/SentiRuEval2016`](https://huggingface.co/datasets/mteb/SentiRuEval2016) • **License:** not specified • [Learn more →](https://github.com/mokoron/sentirueval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | rus | not specified | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{loukachevitch2016sentirueval,
      author = {Loukachevitch, NV and Rubtsova, Yu V},
      booktitle = {Computational Linguistics and Intellectual Technologies},
      pages = {416--426},
      title = {SentiRuEval-2016: overcoming time gap and data sparsity in tweet sentiment analysis},
      year = {2016},
    }

    ```




#### SentiRuEval2016.v2

Russian sentiment analysis evaluation SentiRuEval-2016 devoted to reputation monitoring of banks and telecom companies in Twitter. We describe the task, data, the procedure of data preparation, and participants’ results.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/senti_ru_eval2016`](https://huggingface.co/datasets/mteb/senti_ru_eval2016) • **License:** not specified • [Learn more →](https://github.com/mokoron/sentirueval)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | rus | not specified | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{loukachevitch2016sentirueval,
      author = {Loukachevitch, NV and Rubtsova, Yu V},
      booktitle = {Computational Linguistics and Intellectual Technologies},
      pages = {416--426},
      title = {SentiRuEval-2016: overcoming time gap and data sparsity in tweet sentiment analysis},
      year = {2016},
    }

    ```




#### SentimentAnalysisHindi

Hindi Sentiment Analysis Dataset

**Dataset:** [`OdiaGenAI/sentiment_analysis_hindi`](https://huggingface.co/datasets/OdiaGenAI/sentiment_analysis_hindi) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/OdiaGenAI/sentiment_analysis_hindi)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | hin | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{OdiaGenAI,
      author = {Shantipriya Parida and Sambit Sekhar and Soumendra Kumar Sahoo and Swateek Jena and Abhijeet Parida and Satya Ranjan Dash and Guneet Singh Kohli},
      howpublished = {{https://huggingface.co/OdiaGenAI}},
      journal = {Hugging Face repository},
      publisher = {Hugging Face},
      title = {OdiaGenAI: Generative AI and LLM Initiative for the Odia Language},
      year = {2023},
    }

    ```




#### SentimentAnalysisHindi.v2

Hindi Sentiment Analysis Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sentiment_analysis_hindi`](https://huggingface.co/datasets/mteb/sentiment_analysis_hindi) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/OdiaGenAI/sentiment_analysis_hindi)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | hin | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{OdiaGenAI,
      author = {Shantipriya Parida and Sambit Sekhar and Soumendra Kumar Sahoo and Swateek Jena and Abhijeet Parida and Satya Ranjan Dash and Guneet Singh Kohli},
      howpublished = {{https://huggingface.co/OdiaGenAI}},
      journal = {Hugging Face repository},
      publisher = {Hugging Face},
      title = {OdiaGenAI: Generative AI and LLM Initiative for the Odia Language},
      year = {2023},
    }

    ```




#### SentimentDKSF

The Sentiment DKSF (Digikala/Snappfood comments) is a dataset for sentiment analysis.

**Dataset:** [`hezarai/sentiment-dksf`](https://huggingface.co/datasets/hezarai/sentiment-dksf) • **License:** not specified • [Learn more →](https://github.com/hezarai/hezar)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Reviews | derived | found |



??? quote "Citation"


    ```bibtex

    ```




#### SentimentDKSF.v2

The Sentiment DKSF (Digikala/Snappfood comments) is a dataset for sentiment analysis.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sentiment_dksf`](https://huggingface.co/datasets/mteb/sentiment_dksf) • **License:** not specified • [Learn more →](https://github.com/hezarai/hezar)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Reviews | derived | found |



??? quote "Citation"


    ```bibtex

    ```




#### SinhalaNewsClassification

This file contains news texts (sentences) belonging to 5 different news categories (political, business, technology, sports and Entertainment). The original dataset was released by Nisansa de Silva (Sinhala Text Classification: Observations from the Perspective of a Resource Poor Language, 2015).

**Dataset:** [`NLPC-UOM/Sinhala-News-Category-classification`](https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Category-classification) • **License:** mit • [Learn more →](https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Category-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | sin | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{deSilva2015,
      author = {Nisansa de Silva},
      journal = {Year of Publication},
      title = {Sinhala Text Classification: Observations from the Perspective of a Resource Poor Language},
      year = {2015},
    }

    @article{dhananjaya2022,
      author = {Dhananjaya et al.},
      journal = {Year of Publication},
      title = {BERTifying Sinhala - A Comprehensive Analysis of Pre-trained Language Models for Sinhala Text Classification},
      year = {2022},
    }

    ```




#### SinhalaNewsClassification.v2

This file contains news texts (sentences) belonging to 5 different news categories (political, business, technology, sports and Entertainment). The original dataset was released by Nisansa de Silva (Sinhala Text Classification: Observations from the Perspective of a Resource Poor Language, 2015).
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sinhala_news`](https://huggingface.co/datasets/mteb/sinhala_news) • **License:** mit • [Learn more →](https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Category-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | sin | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{deSilva2015,
      author = {Nisansa de Silva},
      journal = {Year of Publication},
      title = {Sinhala Text Classification: Observations from the Perspective of a Resource Poor Language},
      year = {2015},
    }

    @article{dhananjaya2022,
      author = {Dhananjaya et al.},
      journal = {Year of Publication},
      title = {BERTifying Sinhala - A Comprehensive Analysis of Pre-trained Language Models for Sinhala Text Classification},
      year = {2022},
    }

    ```




#### SinhalaNewsSourceClassification

This dataset contains Sinhala news headlines extracted from 9 news sources (websites) (Sri Lanka Army, Dinamina, GossipLanka, Hiru, ITN, Lankapuwath, NewsLK, Newsfirst, World Socialist Web Site-Sinhala).

**Dataset:** [`NLPC-UOM/Sinhala-News-Source-classification`](https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Source-classification) • **License:** mit • [Learn more →](https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Source-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | sin | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{dhananjaya2022,
      author = {Dhananjaya et al.},
      journal = {Year of Publication},
      title = {BERTifying Sinhala - A Comprehensive Analysis of Pre-trained Language Models for Sinhala Text Classification},
      year = {2022},
    }

    ```




#### SinhalaNewsSourceClassification.v2

This dataset contains Sinhala news headlines extracted from 9 news sources (websites) (Sri Lanka Army, Dinamina, GossipLanka, Hiru, ITN, Lankapuwath, NewsLK, Newsfirst, World Socialist Web Site-Sinhala).
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/sinhala_news_source`](https://huggingface.co/datasets/mteb/sinhala_news_source) • **License:** mit • [Learn more →](https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Source-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | sin | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{dhananjaya2022,
      author = {Dhananjaya et al.},
      journal = {Year of Publication},
      title = {BERTifying Sinhala - A Comprehensive Analysis of Pre-trained Language Models for Sinhala Text Classification},
      year = {2022},
    }

    ```




#### SiswatiNewsClassification

Siswati News Classification Dataset

**Dataset:** [`isaacchung/siswati-news`](https://huggingface.co/datasets/isaacchung/siswati-news) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/dsfsi/za-isizulu-siswati-news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ssw | News, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{Madodonga_Marivate_Adendorff_2023,
      author = {Madodonga, Andani and Marivate, Vukosi and Adendorff, Matthew},
      doi = {10.55492/dhasa.v4i01.4449},
      month = {Jan.},
      title = {Izindaba-Tindzaba: Machine learning news categorisation for Long and Short Text for isiZulu and Siswati},
      url = {https://upjournals.up.ac.za/index.php/dhasa/article/view/4449},
      volume = {4},
      year = {2023},
    }

    ```




#### SiswatiNewsClassification.v2

Siswati News Classification Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/siswati_news`](https://huggingface.co/datasets/mteb/siswati_news) • **License:** cc-by-sa-4.0 • [Learn more →](https://huggingface.co/datasets/dsfsi/za-isizulu-siswati-news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ssw | News, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @article{Madodonga_Marivate_Adendorff_2023,
      author = {Madodonga, Andani and Marivate, Vukosi and Adendorff, Matthew},
      doi = {10.55492/dhasa.v4i01.4449},
      month = {Jan.},
      title = {Izindaba-Tindzaba: Machine learning news categorisation for Long and Short Text for isiZulu and Siswati},
      url = {https://upjournals.up.ac.za/index.php/dhasa/article/view/4449},
      volume = {4},
      year = {2023},
    }

    ```




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



??? quote "Citation"


    ```bibtex

    @article{vstefanik2023resources,
      author = {{\v{S}}tef{\'a}nik, Michal and Kadl{\v{c}}{\'\i}k, Marek and Gramacki, Piotr and Sojka, Petr},
      journal = {arXiv preprint arXiv:2304.01922},
      title = {Resources and Few-shot Learners for In-context Learning in Slavic Languages},
      year = {2023},
    }

    ```




#### SlovakMovieReviewSentimentClassification.v2

User reviews of movies on the CSFD movie database, with 2 sentiment classes (positive, negative)
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/slovak_movie_review_sentiment`](https://huggingface.co/datasets/mteb/slovak_movie_review_sentiment) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/pdf/2304.01922)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | svk | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{vstefanik2023resources,
      author = {{\v{S}}tef{\'a}nik, Michal and Kadl{\v{c}}{\'\i}k, Marek and Gramacki, Piotr and Sojka, Petr},
      journal = {arXiv preprint arXiv:2304.01922},
      title = {Resources and Few-shot Learners for In-context Learning in Slavic Languages},
      year = {2023},
    }

    ```




#### SouthAfricanLangClassification

A language identification test set for 11 South African Languages.

**Dataset:** [`mlexplorer008/south_african_language_identification`](https://huggingface.co/datasets/mlexplorer008/south_african_language_identification) • **License:** mit • [Learn more →](https://www.kaggle.com/competitions/south-african-language-identification/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | afr, eng, nbl, nso, sot, ... (11) | Non-fiction, Web, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{south-african-language-identification,
      author = {ExploreAI Academy, Joanne M},
      publisher = {Kaggle},
      title = {South African Language Identification},
      url = {https://kaggle.com/competitions/south-african-language-identification},
      year = {2022},
    }

    ```




#### SpanishNewsClassification

A Spanish dataset for news classification. The dataset includes articles from reputable Spanish news sources spanning 12 different categories.

**Dataset:** [`MarcOrfilaCarreras/spanish-news`](https://huggingface.co/datasets/MarcOrfilaCarreras/spanish-news) • **License:** mit • [Learn more →](https://huggingface.co/datasets/MarcOrfilaCarreras/spanish-news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | spa | News, Written | derived | found |



??? quote "Citation"


    ```bibtex


    ```




#### SpanishNewsClassification.v2

A Spanish dataset for news classification. The dataset includes articles from reputable Spanish news sources spanning 12 different categories.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/spanish_news`](https://huggingface.co/datasets/mteb/spanish_news) • **License:** mit • [Learn more →](https://huggingface.co/datasets/MarcOrfilaCarreras/spanish-news)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | spa | News, Written | derived | found |



??? quote "Citation"


    ```bibtex


    ```




#### SpanishSentimentClassification

A Spanish dataset for sentiment classification.

**Dataset:** [`sepidmnorozy/Spanish_sentiment`](https://huggingface.co/datasets/sepidmnorozy/Spanish_sentiment) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/sepidmnorozy/Spanish_sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | spa | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{mollanorozy-etal-2023-cross,
      address = {Dubrovnik, Croatia},
      author = {Mollanorozy, Sepideh  and
    Tanti, Marc  and
    Nissim, Malvina},
      booktitle = {Proceedings of the 5th Workshop on Research in Computational Linguistic Typology and Multilingual NLP},
      doi = {10.18653/v1/2023.sigtyp-1.9},
      editor = {Beinborn, Lisa  and
    Goswami, Koustava  and
    Murado{\\u{g}}lu, Saliha  and
    Sorokin, Alexey  and
    Kumar, Ritesh  and
    Shcherbakov, Andreas  and
    Ponti, Edoardo M.  and
    Cotterell, Ryan  and
    Vylomova, Ekaterina},
      month = may,
      pages = {89--95},
      publisher = {Association for Computational Linguistics},
      title = {Cross-lingual Transfer Learning with \{P\}ersian},
      url = {https://aclanthology.org/2023.sigtyp-1.9},
      year = {2023},
    }

    ```




#### SpanishSentimentClassification.v2

A Spanish dataset for sentiment classification.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/spanish_sentiment`](https://huggingface.co/datasets/mteb/spanish_sentiment) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/sepidmnorozy/Spanish_sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | spa | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{mollanorozy-etal-2023-cross,
      address = {Dubrovnik, Croatia},
      author = {Mollanorozy, Sepideh  and
    Tanti, Marc  and
    Nissim, Malvina},
      booktitle = {Proceedings of the 5th Workshop on Research in Computational Linguistic Typology and Multilingual NLP},
      doi = {10.18653/v1/2023.sigtyp-1.9},
      editor = {Beinborn, Lisa  and
    Goswami, Koustava  and
    Murado{\\u{g}}lu, Saliha  and
    Sorokin, Alexey  and
    Kumar, Ritesh  and
    Shcherbakov, Andreas  and
    Ponti, Edoardo M.  and
    Cotterell, Ryan  and
    Vylomova, Ekaterina},
      month = may,
      pages = {89--95},
      publisher = {Association for Computational Linguistics},
      title = {Cross-lingual Transfer Learning with \{P\}ersian},
      url = {https://aclanthology.org/2023.sigtyp-1.9},
      year = {2023},
    }

    ```




#### StyleClassification

A dataset containing formal and informal sentences in Persian for style classification.

**Dataset:** [`MCINext/style-classification`](https://huggingface.co/datasets/MCINext/style-classification) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/MCINext/style-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | fas | Spoken | derived | found |



??? quote "Citation"


    ```bibtex

    ```




#### SwahiliNewsClassification

Dataset for Swahili News Classification, categorized with 6 domains (Local News (Kitaifa), International News (Kimataifa), Finance News (Uchumi), Health News (Afya), Sports News (Michezo), and Entertainment News (Burudani)). Building and Optimizing Swahili Language Models: Techniques, Embeddings, and Datasets

**Dataset:** [`Mollel/SwahiliNewsClassification`](https://huggingface.co/datasets/Mollel/SwahiliNewsClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/Mollel/SwahiliNewsClassification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swa | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{davis2020swahili,
      author = {Davis, David},
      doi = {10.5281/zenodo.5514203},
      publisher = {Zenodo},
      title = {Swahili: News Classification Dataset (0.2)},
      url = {https://doi.org/10.5281/zenodo.5514203},
      year = {2020},
    }

    ```




#### SwahiliNewsClassification.v2

Dataset for Swahili News Classification, categorized with 6 domains (Local News (Kitaifa), International News (Kimataifa), Finance News (Uchumi), Health News (Afya), Sports News (Michezo), and Entertainment News (Burudani)). Building and Optimizing Swahili Language Models: Techniques, Embeddings, and Datasets
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/swahili_news`](https://huggingface.co/datasets/mteb/swahili_news) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://huggingface.co/datasets/Mollel/SwahiliNewsClassification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swa | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{davis2020swahili,
      author = {Davis, David},
      doi = {10.5281/zenodo.5514203},
      publisher = {Zenodo},
      title = {Swahili: News Classification Dataset (0.2)},
      url = {https://doi.org/10.5281/zenodo.5514203},
      year = {2020},
    }

    ```




#### SweRecClassification

A Swedish dataset for sentiment classification on review

**Dataset:** [`mteb/swerec_classification`](https://huggingface.co/datasets/mteb/swerec_classification) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swe | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{nielsen-2023-scandeval,
      address = {T{\'o}rshavn, Faroe Islands},
      author = {Nielsen, Dan},
      booktitle = {Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)},
      editor = {Alum{\"a}e, Tanel  and
    Fishel, Mark},
      month = may,
      pages = {185--201},
      publisher = {University of Tartu Library},
      title = {{S}cand{E}val: A Benchmark for {S}candinavian Natural Language Processing},
      url = {https://aclanthology.org/2023.nodalida-1.20},
      year = {2023},
    }

    ```




#### SweRecClassification.v2

A Swedish dataset for sentiment classification on review
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/swe_rec`](https://huggingface.co/datasets/mteb/swe_rec) • **License:** cc-by-sa-4.0 • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | swe | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{nielsen-2023-scandeval,
      address = {T{\'o}rshavn, Faroe Islands},
      author = {Nielsen, Dan},
      booktitle = {Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)},
      editor = {Alum{\"a}e, Tanel  and
    Fishel, Mark},
      month = may,
      pages = {185--201},
      publisher = {University of Tartu Library},
      title = {{S}cand{E}val: A Benchmark for {S}candinavian Natural Language Processing},
      url = {https://aclanthology.org/2023.nodalida-1.20},
      year = {2023},
    }

    ```




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

**Dataset:** [`mteb/SwissJudgementClassification`](https://huggingface.co/datasets/mteb/SwissJudgementClassification) • **License:** cc-by-4.0 • [Learn more →](https://aclanthology.org/2021.nllp-1.3/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu, fra, ita | Legal, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{niklaus2022empirical,
      archiveprefix = {arXiv},
      author = {Joel Niklaus and Matthias Stürmer and Ilias Chalkidis},
      eprint = {2209.12325},
      primaryclass = {cs.CL},
      title = {An Empirical Study on Cross-X Transfer for Legal Judgment Prediction},
      year = {2022},
    }

    ```




#### SynPerChatbotConvSAAnger

Synthetic Persian Chatbot Conversational Sentiment Analysis Anger

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-anger`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-anger) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    ```




#### SynPerChatbotConvSAFear

Synthetic Persian Chatbot Conversational Sentiment Analysis Fear

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-fear`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-fear) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    ```




#### SynPerChatbotConvSAFriendship

Synthetic Persian Chatbot Conversational Sentiment Analysis Friendship

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-friendship`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-friendship) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    ```




#### SynPerChatbotConvSAHappiness

Synthetic Persian Chatbot Conversational Sentiment Analysis Happiness

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-happiness`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-happiness) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    ```




#### SynPerChatbotConvSAJealousy

Synthetic Persian Chatbot Conversational Sentiment Analysis Jealousy

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-jealousy`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-jealousy) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    ```




#### SynPerChatbotConvSALove

Synthetic Persian Chatbot Conversational Sentiment Analysis Love

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-love`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-love) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    ```




#### SynPerChatbotConvSASadness

Synthetic Persian Chatbot Conversational Sentiment Analysis Sadness

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-sadness`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-sadness) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    ```




#### SynPerChatbotConvSASatisfaction

Synthetic Persian Chatbot Conversational Sentiment Analysis Satisfaction

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-satisfaction`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-satisfaction) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    ```




#### SynPerChatbotConvSASurprise

Synthetic Persian Chatbot Conversational Sentiment Analysis Surprise

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-surprise`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-surprise) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    ```




#### SynPerChatbotConvSAToneChatbotClassification

Synthetic Persian Chatbot Conversational Sentiment Analysis Tone Chatbot Classification

**Dataset:** [`MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-tone-chatbot-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-tone-chatbot-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    ```




#### SynPerChatbotConvSAToneUserClassification

Synthetic Persian Chatbot Conversational Sentiment Analysis Tone User

**Dataset:** [`MCINext/chatbot-conversational-sentiment-analysis-tone-user-classification`](https://huggingface.co/datasets/MCINext/chatbot-conversational-sentiment-analysis-tone-user-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    ```




#### SynPerChatbotRAGToneChatbotClassification

Synthetic Persian Chatbot RAG Tone Chatbot Classification

**Dataset:** [`MCINext/synthetic-persian-chatbot-rag-tone-chatbot-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-tone-chatbot-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    ```




#### SynPerChatbotRAGToneUserClassification

Synthetic Persian Chatbot RAG Tone User Classification

**Dataset:** [`MCINext/synthetic-persian-chatbot-rag-tone-user-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-tone-user-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    ```




#### SynPerChatbotSatisfactionLevelClassification

Synthetic Persian Chatbot Satisfaction Level Classification

**Dataset:** [`MCINext/synthetic-persian-chatbot-satisfaction-level-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-satisfaction-level-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    ```




#### SynPerChatbotToneChatbotClassification

Synthetic Persian Chatbot Tone Chatbot Classification

**Dataset:** [`MCINext/synthetic-persian-chatbot-tone-chatbot-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-tone-chatbot-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    ```




#### SynPerChatbotToneUserClassification

Synthetic Persian Chatbot Tone User Classification

**Dataset:** [`MCINext/synthetic-persian-chatbot-tone-user-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-tone-user-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | Spoken | LM-generated | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    ```




#### SynPerTextToneClassification

Persian Text Tone

**Dataset:** [`MCINext/synthetic-persian-text-tone-classification`](https://huggingface.co/datasets/MCINext/synthetic-persian-text-tone-classification) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | not specified | LM-generated | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    ```




#### SynPerTextToneClassification.v2

Persian Text Tone
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/syn_per_text_tone`](https://huggingface.co/datasets/mteb/syn_per_text_tone) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | fas | not specified | LM-generated | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    ```




#### SynPerTextToneClassification.v3

This version of the Persian text tone classification dataset is an improved version of its predecessors.
         It excludes several classes identified as having low-quality data, leading to a more reliable benchmark.

**Dataset:** [`MCINext/synthetic-persian-text-tone-classification-v3`](https://huggingface.co/datasets/MCINext/synthetic-persian-text-tone-classification-v3) • **License:** not specified • [Learn more →](https://mcinext.com/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | accuracy | fas | not specified | LM-generated | LM-generated and verified |



??? quote "Citation"


    ```bibtex

    ```




#### TNews

Short Text Classification for News

**Dataset:** [`C-MTEB/TNews-classification`](https://huggingface.co/datasets/C-MTEB/TNews-classification) • **License:** not specified • [Learn more →](https://www.cluebenchmarks.com/introduce.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @inproceedings{xu-etal-2020-clue,
      address = {Barcelona, Spain (Online)},
      author = {Xu, Liang  and
    Hu, Hai and
    Zhang, Xuanwei and
    Li, Lu and
    Cao, Chenjie and
    Li, Yudong and
    Xu, Yechen and
    Sun, Kai and
    Yu, Dian and
    Yu, Cong and
    Tian, Yin and
    Dong, Qianqian and
    Liu, Weitang and
    Shi, Bo and
    Cui, Yiming and
    Li, Junyi and
    Zeng, Jun and
    Wang, Rongzhao and
    Xie, Weijian and
    Li, Yanting and
    Patterson, Yina and
    Tian, Zuoyu and
    Zhang, Yiwen and
    Zhou, He and
    Liu, Shaoweihua and
    Zhao, Zhe and
    Zhao, Qipeng and
    Yue, Cong and
    Zhang, Xinrui and
    Yang, Zhengliang and
    Richardson, Kyle and
    Lan, Zhenzhong },
      booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
      doi = {10.18653/v1/2020.coling-main.419},
      month = dec,
      pages = {4762--4772},
      publisher = {International Committee on Computational Linguistics},
      title = {{CLUE}: A {C}hinese Language Understanding Evaluation Benchmark},
      url = {https://aclanthology.org/2020.coling-main.419},
      year = {2020},
    }

    ```




#### TNews.v2

Short Text Classification for News
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/t_news`](https://huggingface.co/datasets/mteb/t_news) • **License:** not specified • [Learn more →](https://www.cluebenchmarks.com/introduce.html)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @inproceedings{xu-etal-2020-clue,
      address = {Barcelona, Spain (Online)},
      author = {Xu, Liang  and
    Hu, Hai and
    Zhang, Xuanwei and
    Li, Lu and
    Cao, Chenjie and
    Li, Yudong and
    Xu, Yechen and
    Sun, Kai and
    Yu, Dian and
    Yu, Cong and
    Tian, Yin and
    Dong, Qianqian and
    Liu, Weitang and
    Shi, Bo and
    Cui, Yiming and
    Li, Junyi and
    Zeng, Jun and
    Wang, Rongzhao and
    Xie, Weijian and
    Li, Yanting and
    Patterson, Yina and
    Tian, Zuoyu and
    Zhang, Yiwen and
    Zhou, He and
    Liu, Shaoweihua and
    Zhao, Zhe and
    Zhao, Qipeng and
    Yue, Cong and
    Zhang, Xinrui and
    Yang, Zhengliang and
    Richardson, Kyle and
    Lan, Zhenzhong },
      booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
      doi = {10.18653/v1/2020.coling-main.419},
      month = dec,
      pages = {4762--4772},
      publisher = {International Committee on Computational Linguistics},
      title = {{CLUE}: A {C}hinese Language Understanding Evaluation Benchmark},
      url = {https://aclanthology.org/2020.coling-main.419},
      year = {2020},
    }

    ```




#### TamilNewsClassification

A Tamil dataset for 6-class classification of Tamil news articles

**Dataset:** [`mlexplorer008/tamil_news_classification`](https://huggingface.co/datasets/mlexplorer008/tamil_news_classification) • **License:** mit • [Learn more →](https://github.com/vanangamudi/tamil-news-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | tam | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{kunchukuttan2020indicnlpcorpus,
      author = {Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
      journal = {arXiv preprint arXiv:2005.00085},
      title = {AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
      year = {2020},
    }

    ```




#### TamilNewsClassification.v2

A Tamil dataset for 6-class classification of Tamil news articles
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/tamil_news`](https://huggingface.co/datasets/mteb/tamil_news) • **License:** mit • [Learn more →](https://github.com/vanangamudi/tamil-news-classification)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | tam | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @article{kunchukuttan2020indicnlpcorpus,
      author = {Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
      journal = {arXiv preprint arXiv:2005.00085},
      title = {AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
      year = {2020},
    }

    ```




#### TelemarketingSalesRuleLegalBenchClassification

Determine how 16 C.F.R. § 310.3(a)(1) and 16 C.F.R. § 310.3(a)(2) (governing deceptive practices) apply to different fact patterns. This dataset is designed to test a model’s ability to apply 16 C.F.R. § 310.3(a)(1) and 16 C.F.R. § 310.3(a)(2) of the Telemarketing Sales Rule to a simple fact pattern with a clear outcome. Each fact pattern ends with the question: “Is this a violation of the Telemarketing Sales Rule?” Each fact pattern is paired with the answer “Yes” or the answer “No.” Fact patterns are listed in the column “text,” and answers are listed in the column “label.”

**Dataset:** [`mteb/TelemarketingSalesRuleLegalBenchClassification`](https://huggingface.co/datasets/mteb/TelemarketingSalesRuleLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




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



??? quote "Citation"


    ```bibtex

    @inproceedings{Schabus2017,
      address = {Tokyo, Japan},
      author = {Dietmar Schabus and Marcin Skowron and Martin Trapp},
      booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
      doi = {10.1145/3077136.3080711},
      month = aug,
      pages = {1241--1244},
      title = {One Million Posts: A Data Set of German Online Discussions},
      year = {2017},
    }

    ```




#### TenKGnadClassification.v2

10k German News Articles Dataset (10kGNAD) contains news articles from the online Austrian newspaper website DER Standard with their topic classification (9 classes).
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/ten_k_gnad`](https://huggingface.co/datasets/mteb/ten_k_gnad) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://tblock.github.io/10kGNAD/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | deu | News, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{Schabus2017,
      address = {Tokyo, Japan},
      author = {Dietmar Schabus and Marcin Skowron and Martin Trapp},
      booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
      doi = {10.1145/3077136.3080711},
      month = aug,
      pages = {1241--1244},
      title = {One Million Posts: A Data Set of German Online Discussions},
      year = {2017},
    }

    ```




#### TextualismToolDictionariesLegalBenchClassification

Determine if a paragraph from a judicial opinion is applying a form textualism that relies on the dictionary meaning of terms.

**Dataset:** [`mteb/TextualismToolDictionariesLegalBenchClassification`](https://huggingface.co/datasets/mteb/TextualismToolDictionariesLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




#### TextualismToolPlainLegalBenchClassification

Determine if a paragraph from a judicial opinion is applying a form textualism that relies on the ordinary (“plain”) meaning of terms.

**Dataset:** [`mteb/TextualismToolPlainLegalBenchClassification`](https://huggingface.co/datasets/mteb/TextualismToolPlainLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




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



??? quote "Citation"


    ```bibtex

    @misc{lin2023toxicchat,
      archiveprefix = {arXiv},
      author = {Zi Lin and Zihan Wang and Yongqi Tong and Yangkun Wang and Yuxin Guo and Yujia Wang and Jingbo Shang},
      eprint = {2310.17389},
      primaryclass = {cs.CL},
      title = {ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation},
      year = {2023},
    }

    ```




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



??? quote "Citation"


    ```bibtex

    @misc{lin2023toxicchat,
      archiveprefix = {arXiv},
      author = {Zi Lin and Zihan Wang and Yongqi Tong and Yangkun Wang and Yuxin Guo and Yujia Wang and Jingbo Shang},
      eprint = {2310.17389},
      primaryclass = {cs.CL},
      title = {ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation},
      year = {2023},
    }

    ```




#### ToxicConversationsClassification

Collection of comments from the Civil Comments platform together with annotations if the comment is toxic or not.

**Dataset:** [`mteb/toxic_conversations_50k`](https://huggingface.co/datasets/mteb/toxic_conversations_50k) • **License:** cc-by-4.0 • [Learn more →](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{jigsaw-unintended-bias-in-toxicity-classification,
      author = {cjadams and Daniel Borkan and inversion and Jeffrey Sorensen and Lucas Dixon and Lucy Vasserman and nithum},
      publisher = {Kaggle},
      title = {Jigsaw Unintended Bias in Toxicity Classification},
      url = {https://kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification},
      year = {2019},
    }

    ```




#### ToxicConversationsClassification.v2

Collection of comments from the Civil Comments platform together with annotations if the comment is toxic or not.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/toxic_conversations`](https://huggingface.co/datasets/mteb/toxic_conversations) • **License:** cc-by-4.0 • [Learn more →](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{jigsaw-unintended-bias-in-toxicity-classification,
      author = {cjadams and Daniel Borkan and inversion and Jeffrey Sorensen and Lucas Dixon and Lucy Vasserman and nithum},
      publisher = {Kaggle},
      title = {Jigsaw Unintended Bias in Toxicity Classification},
      url = {https://kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification},
      year = {2019},
    }

    ```




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




#### TswanaNewsClassification

Tswana News Classification Dataset

**Dataset:** [`dsfsi/daily-news-dikgang`](https://huggingface.co/datasets/dsfsi/daily-news-dikgang) • **License:** cc-by-sa-4.0 • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-031-49002-6_17)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tsn | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{marivate2023puoberta,
      author = {Vukosi Marivate and Moseli Mots'Oehli and Valencia Wagner and Richard Lastrucci and Isheanesu Dzingirai},
      booktitle = {SACAIR 2023 (To Appear)},
      dataset_url = {https://github.com/dsfsi/PuoBERTa},
      keywords = {NLP},
      preprint_url = {https://arxiv.org/abs/2310.09141},
      software_url = {https://huggingface.co/dsfsi/PuoBERTa},
      title = {PuoBERTa: Training and evaluation of a curated language model for Setswana},
      year = {2023},
    }

    ```




#### TswanaNewsClassification.v2

Tswana News Classification Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/tswana_news`](https://huggingface.co/datasets/mteb/tswana_news) • **License:** cc-by-sa-4.0 • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-031-49002-6_17)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tsn | News, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{marivate2023puoberta,
      author = {Vukosi Marivate and Moseli Mots'Oehli and Valencia Wagner and Richard Lastrucci and Isheanesu Dzingirai},
      booktitle = {SACAIR 2023 (To Appear)},
      dataset_url = {https://github.com/dsfsi/PuoBERTa},
      keywords = {NLP},
      preprint_url = {https://arxiv.org/abs/2310.09141},
      software_url = {https://huggingface.co/dsfsi/PuoBERTa},
      title = {PuoBERTa: Training and evaluation of a curated language model for Setswana},
      year = {2023},
    }

    ```




#### TurkicClassification

A dataset of news classification in three Turkic languages.

**Dataset:** [`Electrotubbie/classification_Turkic_languages`](https://huggingface.co/datasets/Electrotubbie/classification_Turkic_languages) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/Electrotubbie/classification_Turkic_languages/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | bak, kaz, kir | News, Written | derived | found |



??? quote "Citation"


    ```bibtex


    ```




#### TurkishMovieSentimentClassification

Turkish Movie Review Dataset

**Dataset:** [`asparius/Turkish-Movie-Review`](https://huggingface.co/datasets/asparius/Turkish-Movie-Review) • **License:** not specified • [Learn more →](https://www.win.tue.nl/~mpechen/publications/pubs/MT_WISDOM2013.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tur | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{Demirtas2013CrosslingualPD,
      author = {Erkin Demirtas and Mykola Pechenizkiy},
      booktitle = {wisdom},
      title = {Cross-lingual polarity detection with machine translation},
      url = {https://api.semanticscholar.org/CorpusID:3912960},
      year = {2013},
    }

    ```




#### TurkishMovieSentimentClassification.v2

Turkish Movie Review Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/turkish_movie_sentiment`](https://huggingface.co/datasets/mteb/turkish_movie_sentiment) • **License:** not specified • [Learn more →](https://www.win.tue.nl/~mpechen/publications/pubs/MT_WISDOM2013.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tur | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{Demirtas2013CrosslingualPD,
      author = {Erkin Demirtas and Mykola Pechenizkiy},
      booktitle = {wisdom},
      title = {Cross-lingual polarity detection with machine translation},
      url = {https://api.semanticscholar.org/CorpusID:3912960},
      year = {2013},
    }

    ```




#### TurkishProductSentimentClassification

Turkish Product Review Dataset

**Dataset:** [`asparius/Turkish-Product-Review`](https://huggingface.co/datasets/asparius/Turkish-Product-Review) • **License:** not specified • [Learn more →](https://www.win.tue.nl/~mpechen/publications/pubs/MT_WISDOM2013.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tur | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{Demirtas2013CrosslingualPD,
      author = {Erkin Demirtas and Mykola Pechenizkiy},
      booktitle = {wisdom},
      title = {Cross-lingual polarity detection with machine translation},
      url = {https://api.semanticscholar.org/CorpusID:3912960},
      year = {2013},
    }

    ```




#### TurkishProductSentimentClassification.v2

Turkish Product Review Dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/turkish_product_sentiment`](https://huggingface.co/datasets/mteb/turkish_product_sentiment) • **License:** not specified • [Learn more →](https://www.win.tue.nl/~mpechen/publications/pubs/MT_WISDOM2013.pdf)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tur | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{Demirtas2013CrosslingualPD,
      author = {Erkin Demirtas and Mykola Pechenizkiy},
      booktitle = {wisdom},
      title = {Cross-lingual polarity detection with machine translation},
      url = {https://api.semanticscholar.org/CorpusID:3912960},
      year = {2013},
    }

    ```




#### TweetEmotionClassification

A dataset of 10,000 tweets that was created with the aim of covering the most frequently used emotion categories in Arabic tweets.

**Dataset:** [`mteb/TweetEmotionClassification`](https://huggingface.co/datasets/mteb/TweetEmotionClassification) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-77116-8_8)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{al2018emotional,
      author = {Al-Khatib, Amr and El-Beltagy, Samhaa R},
      booktitle = {Computational Linguistics and Intelligent Text Processing: 18th International Conference, CICLing 2017, Budapest, Hungary, April 17--23, 2017, Revised Selected Papers, Part II 18},
      organization = {Springer},
      pages = {105--114},
      title = {Emotional tone detection in arabic tweets},
      year = {2018},
    }

    ```




#### TweetEmotionClassification.v2

A dataset of 10,012 tweets that was created with the aim of covering the most frequently used emotion categories in Arabic tweets.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/TweetEmotionClassification`](https://huggingface.co/datasets/mteb/TweetEmotionClassification) • **License:** not specified • [Learn more →](https://link.springer.com/chapter/10.1007/978-3-319-77116-8_8)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{al2018emotional,
      author = {Al-Khatib, Amr and El-Beltagy, Samhaa R},
      booktitle = {Computational Linguistics and Intelligent Text Processing: 18th International Conference, CICLing 2017, Budapest, Hungary, April 17--23, 2017, Revised Selected Papers, Part II 18},
      organization = {Springer},
      pages = {105--114},
      title = {Emotional tone detection in arabic tweets},
      year = {2018},
    }

    ```




#### TweetSarcasmClassification

Arabic sarcasm detection dataset, which was created through the reannotation of available Arabic sentiment analysis datasets.

**Dataset:** [`iabufarha/ar_sarcasm`](https://huggingface.co/datasets/iabufarha/ar_sarcasm) • **License:** mit • [Learn more →](https://aclanthology.org/2020.osact-1.5/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{abu-farha-magdy-2020-arabic,
      abstract = {Sarcasm is one of the main challenges for sentiment analysis systems. Its complexity comes from the expression of opinion using implicit indirect phrasing. In this paper, we present ArSarcasm, an Arabic sarcasm detection dataset, which was created through the reannotation of available Arabic sentiment analysis datasets. The dataset contains 10,547 tweets, 16{\%} of which are sarcastic. In addition to sarcasm the data was annotated for sentiment and dialects. Our analysis shows the highly subjective nature of these tasks, which is demonstrated by the shift in sentiment labels based on annotators{'} biases. Experiments show the degradation of state-of-the-art sentiment analysers when faced with sarcastic content. Finally, we train a deep learning model for sarcasm detection using BiLSTM. The model achieves an F1 score of 0.46, which shows the challenging nature of the task, and should act as a basic baseline for future research on our dataset.},
      address = {Marseille, France},
      author = {Abu Farha, Ibrahim  and
    Magdy, Walid},
      booktitle = {Proceedings of the 4th Workshop on Open-Source Arabic Corpora and Processing Tools, with a Shared Task on Offensive Language Detection},
      editor = {Al-Khalifa, Hend  and
    Magdy, Walid  and
    Darwish, Kareem  and
    Elsayed, Tamer  and
    Mubarak, Hamdy},
      isbn = {979-10-95546-51-1},
      language = {English},
      month = may,
      pages = {32--39},
      publisher = {European Language Resource Association},
      title = {From {A}rabic Sentiment Analysis to Sarcasm Detection: The {A}r{S}arcasm Dataset},
      url = {https://aclanthology.org/2020.osact-1.5},
      year = {2020},
    }

    ```




#### TweetSarcasmClassification.v2

Arabic sarcasm detection dataset, which was created through the reannotation of available Arabic sentiment analysis datasets.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2632)

**Dataset:** [`mteb/tweet_sarcasm`](https://huggingface.co/datasets/mteb/tweet_sarcasm) • **License:** mit • [Learn more →](https://aclanthology.org/2020.osact-1.5/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{abu-farha-magdy-2020-arabic,
      abstract = {Sarcasm is one of the main challenges for sentiment analysis systems. Its complexity comes from the expression of opinion using implicit indirect phrasing. In this paper, we present ArSarcasm, an Arabic sarcasm detection dataset, which was created through the reannotation of available Arabic sentiment analysis datasets. The dataset contains 10,547 tweets, 16{\%} of which are sarcastic. In addition to sarcasm the data was annotated for sentiment and dialects. Our analysis shows the highly subjective nature of these tasks, which is demonstrated by the shift in sentiment labels based on annotators{'} biases. Experiments show the degradation of state-of-the-art sentiment analysers when faced with sarcastic content. Finally, we train a deep learning model for sarcasm detection using BiLSTM. The model achieves an F1 score of 0.46, which shows the challenging nature of the task, and should act as a basic baseline for future research on our dataset.},
      address = {Marseille, France},
      author = {Abu Farha, Ibrahim  and
    Magdy, Walid},
      booktitle = {Proceedings of the 4th Workshop on Open-Source Arabic Corpora and Processing Tools, with a Shared Task on Offensive Language Detection},
      editor = {Al-Khalifa, Hend  and
    Magdy, Walid  and
    Darwish, Kareem  and
    Elsayed, Tamer  and
    Mubarak, Hamdy},
      isbn = {979-10-95546-51-1},
      language = {English},
      month = may,
      pages = {32--39},
      publisher = {European Language Resource Association},
      title = {From {A}rabic Sentiment Analysis to Sarcasm Detection: The {A}r{S}arcasm Dataset},
      url = {https://aclanthology.org/2020.osact-1.5},
      year = {2020},
    }

    ```




#### TweetSentimentClassification

A multilingual Sentiment Analysis dataset consisting of tweets in 8 different languages.

**Dataset:** [`mteb/tweet_sentiment_multilingual`](https://huggingface.co/datasets/mteb/tweet_sentiment_multilingual) • **License:** cc-by-3.0 • [Learn more →](https://aclanthology.org/2022.lrec-1.27)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | ara, deu, eng, fra, hin, ... (8) | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{barbieri-etal-2022-xlm,
      abstract = {Language models are ubiquitous in current NLP, and their multilingual capacity has recently attracted considerable attention. However, current analyses have almost exclusively focused on (multilingual variants of) standard benchmarks, and have relied on clean pre-training and task-specific corpora as multilingual signals. In this paper, we introduce XLM-T, a model to train and evaluate multilingual language models in Twitter. In this paper we provide: (1) a new strong multilingual baseline consisting of an XLM-R (Conneau et al. 2020) model pre-trained on millions of tweets in over thirty languages, alongside starter code to subsequently fine-tune on a target task; and (2) a set of unified sentiment analysis Twitter datasets in eight different languages and a XLM-T model trained on this dataset.},
      address = {Marseille, France},
      author = {Barbieri, Francesco  and
    Espinosa Anke, Luis  and
    Camacho-Collados, Jose},
      booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
      month = jun,
      pages = {258--266},
      publisher = {European Language Resources Association},
      title = {{XLM}-{T}: Multilingual Language Models in {T}witter for Sentiment Analysis and Beyond},
      url = {https://aclanthology.org/2022.lrec-1.27},
      year = {2022},
    }

    ```




#### TweetSentimentExtractionClassification



**Dataset:** [`mteb/tweet_sentiment_extraction`](https://huggingface.co/datasets/mteb/tweet_sentiment_extraction) • **License:** not specified • [Learn more →](https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{tweet-sentiment-extraction,
      author = {Maggie, Phil Culliton, Wei Chen},
      publisher = {Kaggle},
      title = {Tweet Sentiment Extraction},
      url = {https://kaggle.com/competitions/tweet-sentiment-extraction},
      year = {2020},
    }

    ```




#### TweetSentimentExtractionClassification.v2


        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/tweet_sentiment_extraction`](https://huggingface.co/datasets/mteb/tweet_sentiment_extraction) • **License:** not specified • [Learn more →](https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @misc{tweet-sentiment-extraction,
      author = {Maggie, Phil Culliton, Wei Chen},
      publisher = {Kaggle},
      title = {Tweet Sentiment Extraction},
      url = {https://kaggle.com/competitions/tweet-sentiment-extraction},
      year = {2020},
    }

    ```




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




#### TweetTopicSingleClassification

Topic classification dataset on Twitter with 6 labels. Each instance of
        TweetTopic comes with a timestamp which distributes from September 2019 to August 2021.
        Tweets were preprocessed before the annotation to normalize some artifacts, converting
        URLs into a special token {{URL}} and non-verified usernames into {{USERNAME}}. For verified
        usernames, we replace its display name (or account name) with symbols {@}.


**Dataset:** [`mteb/TweetTopicSingleClassification`](https://huggingface.co/datasets/mteb/TweetTopicSingleClassification) • **License:** not specified • [Learn more →](https://arxiv.org/abs/2209.09824)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | News, Social, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{dimosthenis-etal-2022-twitter,
      address = {Gyeongju, Republic of Korea},
      author = {Antypas, Dimosthenis  and
    Ushio, Asahi  and
    Camacho-Collados, Jose  and
    Neves, Leonardo  and
    Silva, Vitor  and
    Barbieri, Francesco},
      booktitle = {Proceedings of the 29th International Conference on Computational Linguistics},
      month = oct,
      publisher = {International Committee on Computational Linguistics},
      title = {{T}witter {T}opic {C}lassification},
      year = {2022},
    }

    ```




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



??? quote "Citation"


    ```bibtex

    @inproceedings{dimosthenis-etal-2022-twitter,
      address = {Gyeongju, Republic of Korea},
      author = {Antypas, Dimosthenis  and
    Ushio, Asahi  and
    Camacho-Collados, Jose  and
    Neves, Leonardo  and
    Silva, Vitor  and
    Barbieri, Francesco},
      booktitle = {Proceedings of the 29th International Conference on Computational Linguistics},
      month = oct,
      publisher = {International Committee on Computational Linguistics},
      title = {{T}witter {T}opic {C}lassification},
      year = {2022},
    }

    ```




#### UCCVCommonLawLegalBenchClassification

Determine if a contract is governed by the Uniform Commercial Code (UCC) or the common law of contracts.

**Dataset:** [`mteb/UCCVCommonLawLegalBenchClassification`](https://huggingface.co/datasets/mteb/UCCVCommonLawLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    ```




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



??? quote "Citation"


    ```bibtex

    @inproceedings{rao-tetreault-2018-dear,
      author = {Rao, Sudha  and
    Tetreault, Joel},
      booktitle = {Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
      month = jun,
      publisher = {Association for Computational Linguistics},
      title = {Dear Sir or Madam, May {I} Introduce the {GYAFC} Dataset: Corpus, Benchmarks and Metrics for Formality Style Transfer},
      url = {https://aclanthology.org/N18-1012},
      year = {2018},
    }

    ```




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



??? quote "Citation"


    ```bibtex

    @inproceedings{rao-tetreault-2018-dear,
      author = {Rao, Sudha  and
    Tetreault, Joel},
      booktitle = {Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
      month = jun,
      publisher = {Association for Computational Linguistics},
      title = {Dear Sir or Madam, May {I} Introduce the {GYAFC} Dataset: Corpus, Benchmarks and Metrics for Formality Style Transfer},
      url = {https://aclanthology.org/N18-1012},
      year = {2018},
    }

    ```




#### UnfairTOSLegalBenchClassification

Given a clause from a terms-of-service contract, determine the category the clause belongs to. The purpose of this task is classifying clauses in Terms of Service agreements. Clauses have been annotated by into nine categories: ['Arbitration', 'Unilateral change', 'Content removal', 'Jurisdiction', 'Choice of law', 'Limitation of liability', 'Unilateral termination', 'Contract by using', 'Other']. The first eight categories correspond to clauses that would potentially be deemed potentially unfair. The last category (Other) corresponds to clauses in agreements which don’t fit into these categories.

**Dataset:** [`mteb/UnfairTOSLegalBenchClassification`](https://huggingface.co/datasets/mteb/UnfairTOSLegalBenchClassification) • **License:** cc-by-4.0 • [Learn more →](https://huggingface.co/datasets/nguha/legalbench)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Legal, Written | expert-annotated | found |



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

    @article{lippi2019claudette,
      author = {Lippi, Marco and Pa{\l}ka, Przemys{\l}aw and Contissa, Giuseppe and Lagioia, Francesca and Micklitz, Hans-Wolfgang and Sartor, Giovanni and Torroni, Paolo},
      journal = {Artificial Intelligence and Law},
      pages = {117--139},
      publisher = {Springer},
      title = {CLAUDETTE: an automated detector of potentially unfair clauses in online terms of service},
      volume = {27},
      year = {2019},
    }

    ```




#### UrduRomanSentimentClassification

The Roman Urdu dataset is a data corpus comprising of more than 20000 records tagged for sentiment (Positive, Negative, Neutral)

**Dataset:** [`mteb/UrduRomanSentimentClassification`](https://huggingface.co/datasets/mteb/UrduRomanSentimentClassification) • **License:** mit • [Learn more →](https://archive.ics.uci.edu/dataset/458/roman+urdu+data+set)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | urd | Social, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{misc_roman_urdu_data_set_458,
      author = {Sharf,Zareen},
      howpublished = {UCI Machine Learning Repository},
      note = {{DOI}: https://doi.org/10.24432/C58325},
      title = {{Roman Urdu Data Set}},
      year = {2018},
    }

    ```




#### UrduRomanSentimentClassification.v2

The Roman Urdu dataset is a data corpus comprising of more than 20000 records tagged for sentiment (Positive, Negative, Neutral)
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/urdu_roman_sentiment`](https://huggingface.co/datasets/mteb/urdu_roman_sentiment) • **License:** mit • [Learn more →](https://archive.ics.uci.edu/dataset/458/roman+urdu+data+set)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | urd | Social, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @misc{misc_roman_urdu_data_set_458,
      author = {Sharf,Zareen},
      howpublished = {UCI Machine Learning Repository},
      note = {{DOI}: https://doi.org/10.24432/C58325},
      title = {{Roman Urdu Data Set}},
      year = {2018},
    }

    ```




#### VieStudentFeedbackClassification

A Vietnamese dataset for classification of student feedback

**Dataset:** [`mteb/VieStudentFeedbackClassification`](https://huggingface.co/datasets/mteb/VieStudentFeedbackClassification) • **License:** mit • [Learn more →](https://ieeexplore.ieee.org/document/8573337)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Reviews, Written | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{8573337,
      author = {Nguyen, Kiet Van and Nguyen, Vu Duc and Nguyen, Phu X. V. and Truong, Tham T. H. and Nguyen, Ngan Luu-Thuy},
      booktitle = {2018 10th International Conference on Knowledge and Systems Engineering (KSE)},
      doi = {10.1109/KSE.2018.8573337},
      number = {},
      pages = {19-24},
      title = {UIT-VSFC: Vietnamese Students’ Feedback Corpus for Sentiment Analysis},
      volume = {},
      year = {2018},
    }

    ```




#### VieStudentFeedbackClassification.v2

A Vietnamese dataset for classification of student feedback
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/vie_student_feedback`](https://huggingface.co/datasets/mteb/vie_student_feedback) • **License:** mit • [Learn more →](https://ieeexplore.ieee.org/document/8573337)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | vie | Reviews, Written | human-annotated | created |



??? quote "Citation"


    ```bibtex

    @inproceedings{8573337,
      author = {Nguyen, Kiet Van and Nguyen, Vu Duc and Nguyen, Phu X. V. and Truong, Tham T. H. and Nguyen, Ngan Luu-Thuy},
      booktitle = {2018 10th International Conference on Knowledge and Systems Engineering (KSE)},
      doi = {10.1109/KSE.2018.8573337},
      number = {},
      pages = {19-24},
      title = {UIT-VSFC: Vietnamese Students’ Feedback Corpus for Sentiment Analysis},
      volume = {},
      year = {2018},
    }

    ```




#### WRIMEClassification

A dataset of Japanese social network rated for sentiment

**Dataset:** [`mteb/WRIMEClassification`](https://huggingface.co/datasets/mteb/WRIMEClassification) • **License:** https://huggingface.co/datasets/shunk031/wrime#licensing-information • [Learn more →](https://aclanthology.org/2021.naacl-main.169/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | jpn | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{kajiwara-etal-2021-wrime,
      abstract = {We annotate 17,000 SNS posts with both the writer{'}s subjective emotional intensity and the reader{'}s objective one to construct a Japanese emotion analysis dataset. In this study, we explore the difference between the emotional intensity of the writer and that of the readers with this dataset. We found that the reader cannot fully detect the emotions of the writer, especially anger and trust. In addition, experimental results in estimating the emotional intensity show that it is more difficult to estimate the writer{'}s subjective labels than the readers{'}. The large gap between the subjective and objective emotions imply the complexity of the mapping from a post to the subjective emotion intensities, which also leads to a lower performance with machine learning models.},
      address = {Online},
      author = {Kajiwara, Tomoyuki  and
    Chu, Chenhui  and
    Takemura, Noriko  and
    Nakashima, Yuta  and
    Nagahara, Hajime},
      booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
      doi = {10.18653/v1/2021.naacl-main.169},
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
      pages = {2095--2104},
      publisher = {Association for Computational Linguistics},
      title = {{WRIME}: A New Dataset for Emotional Intensity Estimation with Subjective and Objective Annotations},
      url = {https://aclanthology.org/2021.naacl-main.169},
      year = {2021},
    }

    ```




#### WRIMEClassification.v2

A dataset of Japanese social network rated for sentiment
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wrime`](https://huggingface.co/datasets/mteb/wrime) • **License:** https://huggingface.co/datasets/shunk031/wrime#licensing-information • [Learn more →](https://aclanthology.org/2021.naacl-main.169/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | jpn | Social, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{kajiwara-etal-2021-wrime,
      abstract = {We annotate 17,000 SNS posts with both the writer{'}s subjective emotional intensity and the reader{'}s objective one to construct a Japanese emotion analysis dataset. In this study, we explore the difference between the emotional intensity of the writer and that of the readers with this dataset. We found that the reader cannot fully detect the emotions of the writer, especially anger and trust. In addition, experimental results in estimating the emotional intensity show that it is more difficult to estimate the writer{'}s subjective labels than the readers{'}. The large gap between the subjective and objective emotions imply the complexity of the mapping from a post to the subjective emotion intensities, which also leads to a lower performance with machine learning models.},
      address = {Online},
      author = {Kajiwara, Tomoyuki  and
    Chu, Chenhui  and
    Takemura, Noriko  and
    Nakashima, Yuta  and
    Nagahara, Hajime},
      booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
      doi = {10.18653/v1/2021.naacl-main.169},
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
      pages = {2095--2104},
      publisher = {Association for Computational Linguistics},
      title = {{WRIME}: A New Dataset for Emotional Intensity Estimation with Subjective and Objective Annotations},
      url = {https://aclanthology.org/2021.naacl-main.169},
      year = {2021},
    }

    ```




#### Waimai

Sentiment Analysis of user reviews on takeaway platforms

**Dataset:** [`C-MTEB/waimai-classification`](https://huggingface.co/datasets/C-MTEB/waimai-classification) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @article{xiao2023c,
      author = {Xiao, Shitao and Liu, Zheng and Zhang, Peitian and Muennighof, Niklas},
      journal = {arXiv preprint arXiv:2309.07597},
      title = {C-pack: Packaged resources to advance general chinese embedding},
      year = {2023},
    }

    ```




#### Waimai.v2

Sentiment Analysis of user reviews on takeaway platforms
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/waimai`](https://huggingface.co/datasets/mteb/waimai) • **License:** not specified • [Learn more →](https://aclanthology.org/2023.nodalida-1.20/)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | cmn | not specified | not specified | not specified |



??? quote "Citation"


    ```bibtex

    @article{xiao2023c,
      author = {Xiao, Shitao and Liu, Zheng and Zhang, Peitian and Muennighof, Niklas},
      journal = {arXiv preprint arXiv:2309.07597},
      title = {C-pack: Packaged resources to advance general chinese embedding},
      year = {2023},
    }

    ```




#### WikipediaBioMetChemClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEasy2GeneExpressionVsMetallurgyClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaEasy2GeneExpressionVsMetallurgyClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




#### WikipediaBioMetChemClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wikipedia_bio_met_chem`](https://huggingface.co/datasets/mteb/wikipedia_bio_met_chem) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




#### WikipediaBiolumNeurochemClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaMedium2BioluminescenceVsNeurochemistryClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaMedium2BioluminescenceVsNeurochemistryClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




#### WikipediaChemEngSpecialtiesClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaMedium5Classification`](https://huggingface.co/datasets/BASF-AI/WikipediaMedium5Classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




#### WikipediaChemFieldsClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEZ10Classification`](https://huggingface.co/datasets/BASF-AI/WikipediaEZ10Classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




#### WikipediaChemFieldsClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wikipedia_chem_fields`](https://huggingface.co/datasets/mteb/wikipedia_chem_fields) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




#### WikipediaChemistryTopicsClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEasy10Classification`](https://huggingface.co/datasets/BASF-AI/WikipediaEasy10Classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




#### WikipediaCompChemSpectroscopyClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaMedium2ComputationalVsSpectroscopistsClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaMedium2ComputationalVsSpectroscopistsClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




#### WikipediaCompChemSpectroscopyClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wikipedia_comp_chem_spectroscopy`](https://huggingface.co/datasets/mteb/wikipedia_comp_chem_spectroscopy) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




#### WikipediaCryobiologySeparationClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEasy5Classification`](https://huggingface.co/datasets/BASF-AI/WikipediaEasy5Classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




#### WikipediaCrystallographyAnalyticalClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaMedium2CrystallographyVsChromatographyTitrationpHClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaMedium2CrystallographyVsChromatographyTitrationpHClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




#### WikipediaCrystallographyAnalyticalClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wikipedia_crystallography_analytical`](https://huggingface.co/datasets/mteb/wikipedia_crystallography_analytical) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




#### WikipediaGreenhouseEnantiopureClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEasy2GreenhouseVsEnantiopureClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaEasy2GreenhouseVsEnantiopureClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




#### WikipediaIsotopesFissionClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaHard2IsotopesVsFissionProductsNuclearFissionClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaHard2IsotopesVsFissionProductsNuclearFissionClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




#### WikipediaLuminescenceClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaHard2BioluminescenceVsLuminescenceClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaHard2BioluminescenceVsLuminescenceClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




#### WikipediaOrganicInorganicClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEasy2SpecialClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaEasy2SpecialClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




#### WikipediaSaltsSemiconductorsClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaHard2SaltsVsSemiconductorMaterialsClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaHard2SaltsVsSemiconductorMaterialsClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




#### WikipediaSolidStateColloidalClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEasy2SolidStateVsColloidalClassification`](https://huggingface.co/datasets/BASF-AI/WikipediaEasy2SolidStateVsColloidalClassification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




#### WikipediaTheoreticalAppliedClassification

ChemTEB evaluates the performance of text embedding models on chemical domain data.

**Dataset:** [`BASF-AI/WikipediaEZ2Classification`](https://huggingface.co/datasets/BASF-AI/WikipediaEZ2Classification) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




#### WikipediaTheoreticalAppliedClassification.v2

ChemTEB evaluates the performance of text embedding models on chemical domain data.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wikipedia_theoretical_applied`](https://huggingface.co/datasets/mteb/wikipedia_theoretical_applied) • **License:** cc-by-nc-sa-4.0 • [Learn more →](https://arxiv.org/abs/2412.00532)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Chemistry | derived | created |



??? quote "Citation"


    ```bibtex

    @article{kasmaee2024chemteb,
      author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
      journal = {arXiv preprint arXiv:2412.00532},
      title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
      year = {2024},
    }

    ```




#### WisesightSentimentClassification

Wisesight Sentiment Corpus: Social media messages in Thai language with sentiment label (positive, neutral, negative, question)

**Dataset:** [`mteb/WisesightSentimentClassification`](https://huggingface.co/datasets/mteb/WisesightSentimentClassification) • **License:** cc0-1.0 • [Learn more →](https://github.com/PyThaiNLP/wisesight-sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | tha | News, Social, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @software{bact_2019_3457447,
      author = {Suriyawongkul, Arthit and
    Chuangsuwanich, Ekapol and
    Chormai, Pattarawat and
    Polpanumas, Charin},
      doi = {10.5281/zenodo.3457447},
      month = sep,
      publisher = {Zenodo},
      title = {PyThaiNLP/wisesight-sentiment: First release},
      url = {https://doi.org/10.5281/zenodo.3457447},
      version = {v1.0},
      year = {2019},
    }

    ```




#### WisesightSentimentClassification.v2

Wisesight Sentiment Corpus: Social media messages in Thai language with sentiment label (positive, neutral, negative, question)
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/wisesight_sentiment`](https://huggingface.co/datasets/mteb/wisesight_sentiment) • **License:** cc0-1.0 • [Learn more →](https://github.com/PyThaiNLP/wisesight-sentiment)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | f1 | tha | News, Social, Written | expert-annotated | found |



??? quote "Citation"


    ```bibtex

    @software{bact_2019_3457447,
      author = {Suriyawongkul, Arthit and
    Chuangsuwanich, Ekapol and
    Chormai, Pattarawat and
    Polpanumas, Charin},
      doi = {10.5281/zenodo.3457447},
      month = sep,
      publisher = {Zenodo},
      title = {PyThaiNLP/wisesight-sentiment: First release},
      url = {https://doi.org/10.5281/zenodo.3457447},
      version = {v1.0},
      year = {2019},
    }

    ```




#### WongnaiReviewsClassification

Wongnai features over 200,000 restaurants, beauty salons, and spas across Thailand on its platform, with detailed information about each merchant and user reviews. In this dataset there are 5 classes corressponding each star rating

**Dataset:** [`Wongnai/wongnai_reviews`](https://huggingface.co/datasets/Wongnai/wongnai_reviews) • **License:** lgpl-3.0 • [Learn more →](https://github.com/wongnai/wongnai-corpus)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | tha | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @software{cstorm125_2020_3852912,
      author = {cstorm125 and lukkiddd},
      doi = {10.5281/zenodo.3852912},
      month = may,
      publisher = {Zenodo},
      title = {PyThaiNLP/classification-benchmarks: v0.1-alpha},
      url = {https://doi.org/10.5281/zenodo.3852912},
      version = {v0.1-alpha},
      year = {2020},
    }

    ```




#### YahooAnswersTopicsClassification

Dataset composed of questions and answers from Yahoo Answers, categorized into topics.

**Dataset:** [`mteb/YahooAnswersTopicsClassification`](https://huggingface.co/datasets/mteb/YahooAnswersTopicsClassification) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/yahoo_answers_topics)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Web, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{NIPS2015_250cf8b5,
      author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
      booktitle = {Advances in Neural Information Processing Systems},
      editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
      pages = {},
      publisher = {Curran Associates, Inc.},
      title = {Character-level Convolutional Networks for Text Classification},
      url = {https://proceedings.neurips.cc/paper_files/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf},
      volume = {28},
      year = {2015},
    }

    ```




#### YahooAnswersTopicsClassification.v2

Dataset composed of questions and answers from Yahoo Answers, categorized into topics.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/yahoo_answers_topics`](https://huggingface.co/datasets/mteb/yahoo_answers_topics) • **License:** not specified • [Learn more →](https://huggingface.co/datasets/yahoo_answers_topics)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Web, Written | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{NIPS2015_250cf8b5,
      author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
      booktitle = {Advances in Neural Information Processing Systems},
      editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
      pages = {},
      publisher = {Curran Associates, Inc.},
      title = {Character-level Convolutional Networks for Text Classification},
      url = {https://proceedings.neurips.cc/paper_files/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf},
      volume = {28},
      year = {2015},
    }

    ```




#### YelpReviewFullClassification

Yelp Review Full is a dataset for sentiment analysis, containing 5 classes corresponding to ratings 1-5.

**Dataset:** [`Yelp/yelp_review_full`](https://huggingface.co/datasets/Yelp/yelp_review_full) • **License:** https://huggingface.co/datasets/Yelp/yelp_review_full#licensing-information • [Learn more →](https://arxiv.org/abs/1509.01626)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{NIPS2015_250cf8b5,
      author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
      booktitle = {Advances in Neural Information Processing Systems},
      editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
      pages = {},
      publisher = {Curran Associates, Inc.},
      title = {Character-level Convolutional Networks for Text Classification},
      url = {https://proceedings.neurips.cc/paper_files/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf},
      volume = {28},
      year = {2015},
    }

    ```




#### YelpReviewFullClassification.v2

Yelp Review Full is a dataset for sentiment analysis, containing 5 classes corresponding to ratings 1-5.
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/yelp_review_full`](https://huggingface.co/datasets/mteb/yelp_review_full) • **License:** https://huggingface.co/datasets/Yelp/yelp_review_full#licensing-information • [Learn more →](https://arxiv.org/abs/1509.01626)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | eng | Reviews, Written | derived | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{NIPS2015_250cf8b5,
      author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
      booktitle = {Advances in Neural Information Processing Systems},
      editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
      pages = {},
      publisher = {Curran Associates, Inc.},
      title = {Character-level Convolutional Networks for Text Classification},
      url = {https://proceedings.neurips.cc/paper_files/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf},
      volume = {28},
      year = {2015},
    }

    ```




#### YueOpenriceReviewClassification

A Cantonese dataset for review classification

**Dataset:** [`izhx/yue-openrice-review`](https://huggingface.co/datasets/izhx/yue-openrice-review) • **License:** not specified • [Learn more →](https://github.com/Christainx/Dataset_Cantonese_Openrice)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | yue | Reviews, Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{xiang2019sentiment,
      author = {Xiang, Rong and Jiao, Ying and Lu, Qin},
      booktitle = {Proceedings of the 8th KDD Workshop on Issues of Sentiment Discovery and Opinion Mining (WISDOM)},
      organization = {KDD WISDOM},
      pages = {1--9},
      title = {Sentiment Augmented Attention Network for Cantonese Restaurant Review Analysis},
      year = {2019},
    }

    ```




#### YueOpenriceReviewClassification.v2

A Cantonese dataset for review classification
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)

**Dataset:** [`mteb/yue_openrice_review`](https://huggingface.co/datasets/mteb/yue_openrice_review) • **License:** not specified • [Learn more →](https://github.com/Christainx/Dataset_Cantonese_Openrice)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to category (t2c) | accuracy | yue | Reviews, Spoken | human-annotated | found |



??? quote "Citation"


    ```bibtex

    @inproceedings{xiang2019sentiment,
      author = {Xiang, Rong and Jiao, Ying and Lu, Qin},
      booktitle = {Proceedings of the 8th KDD Workshop on Issues of Sentiment Discovery and Opinion Mining (WISDOM)},
      organization = {KDD WISDOM},
      pages = {1--9},
      title = {Sentiment Augmented Attention Network for Cantonese Restaurant Review Analysis},
      year = {2019},
    }

    ```
