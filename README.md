# Massive Text Embedding Benchmark

Massive Text Embedding Benchmark 

## Installation

```bash
pip install mteb
```

## Minimal use

* Using a python script:

````python
from mteb import MTEB
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
model_name = "average_word_embeddings_komninos"

model = SentenceTransformer(model_name)
evaluation = MTEB(tasks=["Banking77Classification"])
evaluation.run(model, output_folder=f"results/{model_name}")


````

* Using CLI

```bash
mteb --available_tasks

mteb -m average_word_embeddings_komninos \
    -t Banking77Classification  \
    --output_folder results/average_word_embeddings_komninos \
    --verbosity 3
```

## Advanced usage

### Tasks selection

Tasks can be selected by providing the list of tasks that needs to be run, but also

* by their types (e.g. "Clustering" or "Classification")

````python
evaluation = MTEB(task_types=['Clustering', 'Retrieval']) # Only select clustering and retrieval tasks
````

* by their categories e.g. "S2S" (sentence to sentence) or "P2P" (paragraph to paragraph)

````python
evaluation = MTEB(task_categories=['S2S']) # Only select sentence2sentence tasks
````

* by their languages

````python
evaluation = MTEB(task_langs=["en", "de"]) # Only select tasks which support "en", "de" or "en-de"
````

You can also specify which languages to load for multilingual/crosslingual tasks like this:

````python
from mteb.tasks.BitextMining import BUCCBitextMining

evaluation = MTEB(tasks=[
        BUCCBitextMining(langs=["de-en"]), # Only load "de-en" and fr-en" subsets of BUCC
        AmazonReviewsClassification(langs=["en", "fr"]) # Only load "en" and "fr" subsets of Amazon Reviews
])
````

### Evaluation split
We can choose to evaluate only on `test` splits of all tasks by doing the following:

````python
evaluation.run(model, eval_splits=["test"])
````

### Using a custom model

Models should implement the following interface, implementing an `encode` function taking as inputs a list of sentences, and returning a list of embeddings (embeddings can be `np.array`, `torch.tensor`, etc.).

```python
class MyModel():
    def encode(self, sentences, batch_size=32):
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        pass

model = MyModel()
evaluation = MTEB(tasks=["Banking77Classification"])
evaluation.run(model)
```

### Evaluating on a custom task

To add a new task, you need to implement a new class that inherits from the `AbsTask` associated with the task type (e.g. `AbsTaskReranking` for reranking tasks). You can find the supported task types in [here](https://github.com/embeddings-benchmark/mteb-draft/tree/main/mteb/abstasks).

```python
from mteb import MTEB
from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from sentence_transformers import SentenceTransformer


class MindSmallReranking(AbsTaskReranking):
    @property
    def description(self):
        return {
            "name": "MindSmallReranking",
            "hf_hub_name": "mteb/mind_small",
            "description": "Microsoft News Dataset: A Large-Scale English Dataset for News Recommendation Research",
            "reference": "https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf",
            "type": "Reranking",
            "category": "s2s",
            "eval_splits": ["validation"],
            "eval_langs": ["en"],
            "main_score": "map",
        }

model = SentenceTransformer("average_word_embeddings_komninos")
evaluation = MTEB(tasks=[MindSmallReranking()])
evaluation.run(model)
```

> **Note:** for multilingual tasks, make sure your class also inherits from the `MultilingualTask` class like in [this](https://github.com/embeddings-benchmark/mteb-draft/blob/main/mteb/tasks/Classification/MTOPIntentClassification.py) example.

## Available tasks

| Name                                                                                                                                                                  | Hub URL                                          | Description                                                                                                                                                                                                      | Type               | Category   |   N° Languages |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------|:-----------|---------------:|
| [BUCC](https://comparable.limsi.fr/bucc2018/bucc2018-task.html)                                                                                                       | mteb/bucc-bitext-mining                          | BUCC bitext mining dataset                                                                                                                                                                                       | BitextMining       | s2s        |              4 |
| [Tatoeba](https://github.com/facebookresearch/LASER/tree/main/data/tatoeba/v1)                                                                                        | mteb/tatoeba-bitext-mining                       | 1,000 English-aligned sentence pairs for each language based on the Tatoeba corpus                                                                                                                               | BitextMining       | s2s        |            112 |
| [AmazonCounterfactualClassification](https://arxiv.org/abs/2104.06893)                                                                                                | mteb/amazon_counterfactual                       | A collection of Amazon customer reviews annotated for counterfactual detection pair classification.                                                                                                              | Classification     | s2s        |              4 |
| [AmazonPolarityClassification](https://dl.acm.org/doi/10.1145/2507157.2507163)                                                                                        | mteb/amazon_polarity                             | Amazon Polarity Classification Dataset.                                                                                                                                                                          | Classification     | s2s        |              1 |
| [AmazonReviewsClassification](https://arxiv.org/abs/2010.02573)                                                                                                       | mteb/amazon_reviews_multi                        | A collection of Amazon reviews specifically designed to aid research in multilingual text classification.                                                                                                        | Classification     | s2s        |              6 |
| [Banking77Classification](https://arxiv.org/abs/2003.04807)                                                                                                           | mteb/banking77                                   | Dataset composed of online banking queries annotated with their corresponding intents.                                                                                                                           | Classification     | s2s        |              1 |
| [EmotionClassification](https://www.aclweb.org/anthology/D18-1404)                                                                                                    | mteb/emotion                                     | Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise. For more detailed information please refer to the paper.                                | Classification     | s2s        |              1 |
| [ImdbClassification](http://www.aclweb.org/anthology/P11-1015)                                                                                                        | mteb/imdb                                        | Large Movie Review Dataset                                                                                                                                                                                       | Classification     | p2p        |              1 |
| [MassiveIntentClassification](https://arxiv.org/abs/2204.08582#:~:text=MASSIVE%20contains%201M%20realistic%2C%20parallel,diverse%20languages%20from%2029%20genera.)   | mteb/amazon_massive_intent                       | MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages                                                                                                | Classification     | s2s        |             51 |
| [MassiveScenarioClassification](https://arxiv.org/abs/2204.08582#:~:text=MASSIVE%20contains%201M%20realistic%2C%20parallel,diverse%20languages%20from%2029%20genera.) | mteb/amazon_massive_scenario                     | MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages                                                                                                | Classification     | s2s        |             51 |
| [MTOPDomainClassification](https://arxiv.org/pdf/2008.09335.pdf)                                                                                                      | mteb/mtop_domain                                 | MTOP: Multilingual Task-Oriented Semantic Parsing                                                                                                                                                                | Classification     | s2s        |              6 |
| [MTOPIntentClassification](https://arxiv.org/pdf/2008.09335.pdf)                                                                                                      | mteb/mtop_intent                                 | MTOP: Multilingual Task-Oriented Semantic Parsing                                                                                                                                                                | Classification     | s2s        |              6 |
| [ToxicConversationsClassification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview)                                    | mteb/toxic_conversations_50k                     | Collection of comments from the Civil Comments platform together with annotations if the comment is toxic or not.                                                                                                | Classification     | s2s        |              1 |
| [TweetSentimentExtractionClassification](https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview)                                                     | mteb/tweet_sentiment_extraction                  |                                                                                                                                                                                                                  | Classification     | s2s        |              1 |
| [ArxivClusteringP2P](https://www.kaggle.com/Cornell-University/arxiv)                                                                                                 | mteb/arxiv-clustering-p2p                        | Clustering of titles+abstract from arxiv. Clustering of 30 sets, either on the main or secondary category                                                                                                        | Clustering         | p2p        |              1 |
| [ArxivClusteringS2S](https://www.kaggle.com/Cornell-University/arxiv)                                                                                                 | mteb/arxiv-clustering-s2s                        | Clustering of titles from arxiv. Clustering of 30 sets, either on the main or secondary category                                                                                                                 | Clustering         | s2s        |              1 |
| [BiorxivClusteringP2P](https://api.biorxiv.org/)                                                                                                                      | mteb/biorxiv-clustering-p2p                      | Clustering of titles+abstract from biorxiv. Clustering of 10 sets, based on the main category.                                                                                                                   | Clustering         | p2p        |              1 |
| [BiorxivClusteringS2S](https://api.biorxiv.org/)                                                                                                                      | mteb/biorxiv-clustering-s2s                      | Clustering of titles from biorxiv. Clustering of 10 sets, based on the main category.                                                                                                                            | Clustering         | s2s        |              1 |
| [MedrxivClusteringP2P](https://api.biorxiv.org/)                                                                                                                      | mteb/medrxiv-clustering-p2p                      | Clustering of titles+abstract from medrxiv. Clustering of 10 sets, based on the main category.                                                                                                                   | Clustering         | p2p        |              1 |
| [MedrxivClusteringS2S](https://api.biorxiv.org/)                                                                                                                      | mteb/medrxiv-clustering-s2s                      | Clustering of titles from medrxiv. Clustering of 10 sets, based on the main category.                                                                                                                            | Clustering         | s2s        |              1 |
| [RedditClustering](https://arxiv.org/abs/2104.07081)                                                                                                                  | mteb/reddit-clustering                           | Clustering of titles from 199 subreddits. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.                                                                              | Clustering         | s2s        |              1 |
| [RedditClusteringP2P](https://huggingface.co/datasets/sentence-transformers/reddit-title-body)                                                                        | mteb/reddit-clustering-p2p                       | Clustering of title+posts from reddit. Clustering of 10 sets of 50k paragraphs and 40 sets of 10k paragraphs.                                                                                                    | Clustering         | p2p        |              1 |
| [StackExchangeClustering](https://arxiv.org/abs/2104.07081)                                                                                                           | mteb/stackexchange-clustering                    | Clustering of titles from 121 stackexchanges. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.                                                                          | Clustering         | s2s        |              1 |
| [StackExchangeClusteringP2P](https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_title_body_jsonl)                                                 | mteb/stackexchange-clustering-p2p                | Clustering of title+body from stackexchange. Clustering of 5 sets of 10k paragraphs and 5 sets of 5k paragraphs.                                                                                                 | Clustering         | p2p        |              1 |
| [TwentyNewsgroupsClustering](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)                                                                           | mteb/twentynewsgroups-clustering                 | Clustering of the 20 Newsgroups dataset (subject only).                                                                                                                                                          | Clustering         | s2s        |              1 |
| [SprintDuplicateQuestions](https://www.aclweb.org/anthology/D18-1131/)                                                                                                | mteb/sprintduplicatequestions-pairclassification | Duplicate questions from the Sprint community.                                                                                                                                                                   | PairClassification | s2s        |              1 |
| [TwitterSemEval2015](https://alt.qcri.org/semeval2015/task1/)                                                                                                         | mteb/twittersemeval2015-pairclassification       | Paraphrase-Pairs of Tweets from the SemEval 2015 workshop.                                                                                                                                                       | PairClassification | s2s        |              1 |
| [TwitterURLCorpus](https://languagenet.github.io/)                                                                                                                    | mteb/twitterurlcorpus-pairclassification         | Paraphrase-Pairs of Tweets.                                                                                                                                                                                      | PairClassification | s2s        |              1 |
| [AskUbuntuDupQuestions](https://github.com/taolei87/askubuntu)                                                                                                        | mteb/askubuntudupquestions-reranking             | AskUbuntu Question Dataset - Questions from AskUbuntu with manual annotations marking pairs of questions as similar or non-similar                                                                               | Reranking          | s2s        |              1 |
| [MindSmallReranking](https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf)                                                                 | mteb/mind_small                                  | Microsoft News Dataset: A Large-Scale English Dataset for News Recommendation Research                                                                                                                           | Reranking          | s2s        |              1 |
| [SciDocs](https://allenai.org/data/scidocs)                                                                                                                           | mteb/scidocs-reranking                           | Ranking of related scientific papers based on their title.                                                                                                                                                       | Reranking          | s2s        |              1 |
| [StackOverflowDupQuestions](https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf)                                                          | mteb/stackoverflowdupquestions-reranking         | Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python                                                                                                                  | Reranking          | s2s        |              1 |
| [ArguAna](http://argumentation.bplaced.net/arguana/data)                                                                                                              | nan                                              | NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval                                                                                                                                 | Retrieval          | s2s        |              1 |
| [ClimateFEVER](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)                                                                                  | nan                                              | CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change.                                                                                     | Retrieval          | s2s        |              1 |
| [CQADupstackRetrieval](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)                                                                                          | nan                                              | CQADupStack: A Benchmark Data Set for Community Question-Answering Research                                                                                                                                      | Retrieval          | s2s        |              1 |
| [DBPedia](https://github.com/iai-group/DBpedia-Entity/)                                                                                                               | nan                                              | DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base                                                                                                                   | Retrieval          | s2s        |              1 |
| [FEVER](https://fever.ai/)                                                                                                                                            | nan                                              | FEVER (Fact Extraction and VERification) consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. | Retrieval          | s2s        |              1 |
| [FiQA2018](https://sites.google.com/view/fiqa/)                                                                                                                       | nan                                              | Financial Opinion Mining and Question Answering                                                                                                                                                                  | Retrieval          | s2s        |              1 |
| [HotpotQA](https://hotpotqa.github.io/)                                                                                                                               | nan                                              | HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.                             | Retrieval          | s2s        |              1 |
| [MSMARCO](https://microsoft.github.io/msmarco/)                                                                                                                       | nan                                              | MS MARCO is a collection of datasets focused on deep learning in search                                                                                                                                          | Retrieval          | s2s        |              1 |
| [MSMARCOv2](https://microsoft.github.io/msmarco/TREC-Deep-Learning.html)                                                                                              | nan                                              | nan                                                                                                                                                                                                              | Retrieval          | s2s        |              1 |
| [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)                                                                                                   | nan                                              | NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval                                                                                                                                 | Retrieval          | s2s        |              1 |
| [NQ](https://ai.google.com/research/NaturalQuestions/)                                                                                                                | nan                                              | NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval                                                                                                                                 | Retrieval          | s2s        |              1 |
| [QuoraRetrieval](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)                                                                              | nan                                              | QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.                                                                    | Retrieval          | s2s        |              1 |
| [SCIDOCS](https://allenai.org/data/scidocs)                                                                                                                           | nan                                              | SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation.                                                    | Retrieval          | s2s        |              1 |
| [SciFact](https://github.com/allenai/scifact)                                                                                                                         | nan                                              | nan                                                                                                                                                                                                              | Retrieval          | s2s        |              1 |
| [Touche2020](https://webis.de/events/touche-20/shared-task-1.html)                                                                                                    | nan                                              | Touché Task 1: Argument Retrieval for Controversial Questions                                                                                                                                                    | Retrieval          | s2s        |              1 |
| [TRECCOVID](https://ir.nist.gov/covidSubmit/index.html)                                                                                                               | nan                                              | nan                                                                                                                                                                                                              | Retrieval          | s2s        |              1 |
| [BIOSSES](https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html)                                                                                                      | mteb/biosses-sts                                 | Biomedical Semantic Similarity Estimation.                                                                                                                                                                       | STS                | s2s        |              1 |
| [SICK-R](https://www.aclweb.org/anthology/S14-2001.pdf)                                                                                                               | mteb/biosses-sts                                 | Semantic Textual Similarity SICK-R dataset as described here:                                                                                                                                                    | STS                | s2s        |              1 |
| [STS12](https://www.aclweb.org/anthology/S12-1051.pdf)                                                                                                                | mteb/sts12-sts                                   | SemEval STS 2012 dataset.                                                                                                                                                                                        | STS                | s2s        |              1 |
| [STS13](https://www.aclweb.org/anthology/S13-1004/)                                                                                                                   | mteb/sts13-sts                                   | SemEval STS 2013 dataset.                                                                                                                                                                                        | STS                | s2s        |              1 |
| [STS14](http://alt.qcri.org/semeval2014/task10/)                                                                                                                      | mteb/sts14-sts                                   | SemEval STS 2014 dataset. Currently only the English dataset                                                                                                                                                     | STS                | s2s        |              1 |
| [STS15](http://alt.qcri.org/semeval2015/task2/)                                                                                                                       | mteb/sts15-sts                                   | SemEval STS 2015 dataset                                                                                                                                                                                         | STS                | s2s        |              1 |
| [STS16](http://alt.qcri.org/semeval2016/task1/)                                                                                                                       | mteb/sts16-sts                                   | SemEval STS 2016 dataset                                                                                                                                                                                         | STS                | s2s        |              1 |
| [STS17](http://alt.qcri.org/semeval2016/task1/)                                                                                                                       | mteb/sts17-crosslingual-sts                      | STS 2017 dataset                                                                                                                                                                                                 | STS                | s2s        |             11 |
| [STS22](https://competitions.codalab.org/competitions/33835)                                                                                                          | mteb/sts22-crosslingual-sts                      | SemEval 2022 Task 8: Multilingual News Article Similarity                                                                                                                                                        | STS                | s2s        |             18 |
| [STSBenchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)                                                                                                  | mteb/stsbenchmark-sts                            | Semantic Textual Similarity Benchmark (STSbenchmark) dataset.                                                                                                                                                    | STS                | s2s        |              1 |
| [SummEval](https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html)                                                                                                     | mteb/summeval                                    | Biomedical Semantic Similarity Estimation.                                                                                                                                                                       | Summarization      | s2s        |              1 |
