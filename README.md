<h1 align="center">Massive Text Embedding Benchmark</h1>

<p align="center">
    <a href="https://github.com/mbeddings-benchmark/mteb/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/embeddings-benchmark/mteb.svg">
    </a>
    <a href="https://www.python.org/">
            <img alt="Build" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple">
    </a>
    <a href="https://github.com/embeddings-benchmark/mteb/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/embeddings-benchmark/mteb.svg?color=green">
    </a>
    <a href="https://pepy.tech/project/mteb">
        <img alt="Downloads" src="https://static.pepy.tech/personalized-badge/mteb?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="https://arxiv.org/abs/2210.07316">Paper</a> |
        <a href="https://huggingface.co/spaces/mteb/leaderboard">Leaderboard</a> |
        <a href="#installation">Installation</a> |
        <a href="#usage">Usage</a> |
        <a href="#available-tasks">Tasks</a> |
        <a href="https://huggingface.co/mteb">Hugging Face</a>
    <p>
</h4>

<!-- > The development of MTEB is supported by: -->


<h3 align="center">
    <a href="https://huggingface.co/"><img style="float: middle; padding: 10px 10px 10px 10px;" width="50" height="50" src="./images/hf_logo.png" /></a>
</h3>


## Installation

```bash
pip install mteb
```

## Usage

* Using a python script:

```python
from mteb import MTEB
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
model_name = "average_word_embeddings_komninos"

model = SentenceTransformer(model_name)
evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(model, output_folder=f"results/{model_name}")
```

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
from mteb.tasks import AmazonReviewsClassification, BUCCBitextMining

evaluation = MTEB(tasks=[
        AmazonReviewsClassification(langs=["en", "fr"]) # Only load "en" and "fr" subsets of Amazon Reviews
        BUCCBitextMining(langs=["de-en"]), # Only load "de-en" subset of BUCC
])
````

### Evaluation split
We can choose to evaluate only on `test` splits of all tasks by doing the following:

````python
evaluation.run(model, eval_splits=["test"])
````

### Using a custom model

Models should implement the following interface, implementing an `encode` function taking as inputs a list of sentences, and returning a list of embeddings (embeddings can be `np.array`, `torch.tensor`, etc.). For inspiration, you can look at the [mtebscripts repo](https://github.com/embeddings-benchmark/mtebscripts) used for running diverse models via SLURM scripts for the paper.

```python
class MyModel():
    def encode(self, sentences, batch_size=32, **kwargs):
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

If you'd like to use different encoding functions for query and corpus when evaluating a Dense Retrieval Exact Search (DRES) model on retrieval tasks from BeIR, you can make your model DRES compatible. If compatible like the below example, it will be used for BeIR upon evaluation.

```python
from mteb import AbsTaskRetrieval, DRESModel

class MyModel(DRESModel):
    # Refer to the code of DRESModel for the methods to overwrite
    pass

assert AbsTaskRetrieval.is_dres_compatible(MyModel)
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

## Leaderboard

The MTEB Leaderboard is available [here](https://huggingface.co/spaces/mteb/leaderboard). To submit:
1. Run your model on MTEB
2. Format the json files into metadata using the script at `scripts/mteb_meta.py`. For example
`python scripts/mteb_meta.py path_to_results_folder`, which will create a `mteb_metadata.md` file. If you ran CQADupstack retrieval, make sure to merge the results first with `python scripts/merge_cqadupstack.py path_to_results_folder`.
3. Copy the content of the `mteb_metadata.md` file to the top of a `README.md` file of your model on the Hub. See [here](https://huggingface.co/Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit/blob/main/README.md) for an example.
4. Refresh the leaderboard and you should see your scores 🥇
5. To have the scores appear without refreshing, you can open an issue on the [Community Tab of the LB](https://huggingface.co/spaces/mteb/leaderboard/discussions).

## Available tasks

| Name | Hub URL | Description | Type | Category | #Languages | Train #Samples | Dev #Samples | Test #Samples | Avg. chars / train | Avg. chars / dev | Avg. chars / test
|:-----|:-----|:-----|:-----|:-----|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| [BUCC](https://comparable.limsi.fr/bucc2018/bucc2018-task.html) | [mteb/bucc-bitext-mining](https://huggingface.co/datasets/mteb/bucc-bitext-mining) | BUCC bitext mining dataset | BitextMining | s2s | 4 | 0 | 0 | 641684 | 0 | 0 | 101.3 |
| [Tatoeba](https://github.com/facebookresearch/LASER/tree/main/data/tatoeba/v1) | [mteb/tatoeba-bitext-mining](https://huggingface.co/datasets/mteb/tatoeba-bitext-mining) | 1,000 English-aligned sentence pairs for each language based on the Tatoeba corpus | BitextMining | s2s | 112 | 0 | 0 | 2000 | 0 | 0 | 39.4 |
| [AmazonCounterfactualClassification](https://arxiv.org/abs/2104.06893) | [mteb/amazon_counterfactual](https://huggingface.co/datasets/mteb/amazon_counterfactual) | A collection of Amazon customer reviews annotated for counterfactual detection pair classification. | Classification | s2s | 4 | 4018 | 335 | 670 | 107.3 | 109.2 | 106.1 |
| [AmazonPolarityClassification](https://dl.acm.org/doi/10.1145/2507157.2507163) | [mteb/amazon_polarity](https://huggingface.co/datasets/mteb/amazon_polarity) | Amazon Polarity Classification Dataset. | Classification | s2s | 1 | 3600000 | 0 | 400000 | 431.6 | 0 | 431.4 |
| [AmazonReviewsClassification](https://arxiv.org/abs/2010.02573) | [mteb/amazon_reviews_multi](https://huggingface.co/datasets/mteb/amazon_reviews_multi) | A collection of Amazon reviews specifically designed to aid research in multilingual text classification. | Classification | s2s | 6 | 1200000 | 30000 | 30000 | 160.5 | 159.2 | 160.4 |
| [Banking77Classification](https://arxiv.org/abs/2003.04807) | [mteb/banking77](https://huggingface.co/datasets/mteb/banking77) | Dataset composed of online banking queries annotated with their corresponding intents. | Classification | s2s | 1 | 10003 | 0 | 3080 | 59.5 | 0 | 54.2 |
| [EmotionClassification](https://www.aclweb.org/anthology/D18-1404) | [mteb/emotion](https://huggingface.co/datasets/mteb/emotion) | Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise. For more detailed information please refer to the paper. | Classification | s2s | 1 | 16000 | 2000 | 2000 | 96.8 | 95.3 | 96.6 |
| [ImdbClassification](http://www.aclweb.org/anthology/P11-1015) | [mteb/imdb](https://huggingface.co/datasets/mteb/imdb) | Large Movie Review Dataset | Classification | p2p | 1 | 25000 | 0 | 25000 | 1325.1 | 0 | 1293.8 |
| [MassiveIntentClassification](https://arxiv.org/abs/2204.08582#:~:text=MASSIVE%20contains%201M%20realistic%2C%20parallel,diverse%20languages%20from%2029%20genera.) | [mteb/amazon_massive_intent](https://huggingface.co/datasets/mteb/amazon_massive_intent) | MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages | Classification | s2s | 51 | 11514 | 2033 | 2974 | 35.0 | 34.8 | 34.6 |
| [MassiveScenarioClassification](https://arxiv.org/abs/2204.08582#:~:text=MASSIVE%20contains%201M%20realistic%2C%20parallel,diverse%20languages%20from%2029%20genera.) | [mteb/amazon_massive_scenario](https://huggingface.co/datasets/mteb/amazon_massive_scenario) | MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages | Classification | s2s | 51 | 11514 | 2033 | 2974 | 35.0 | 34.8 | 34.6 |
| [MTOPDomainClassification](https://arxiv.org/pdf/2008.09335.pdf) | [mteb/mtop_domain](https://huggingface.co/datasets/mteb/mtop_domain) | MTOP: Multilingual Task-Oriented Semantic Parsing | Classification | s2s | 6 | 15667 | 2235 | 4386 | 36.6 | 36.5 | 36.8 |
| [MTOPIntentClassification](https://arxiv.org/pdf/2008.09335.pdf) | [mteb/mtop_intent](https://huggingface.co/datasets/mteb/mtop_intent) | MTOP: Multilingual Task-Oriented Semantic Parsing | Classification | s2s | 6 | 15667 | 2235 | 4386 | 36.6 | 36.5 | 36.8 |
| [ToxicConversationsClassification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview) | [mteb/toxic_conversations_50k](https://huggingface.co/datasets/mteb/toxic_conversations_50k) | Collection of comments from the Civil Comments platform together with annotations if the comment is toxic or not. | Classification | s2s | 1 | 50000 | 0 | 50000 | 298.8 | 0 | 296.6 |
| [TweetSentimentExtractionClassification](https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview) | [mteb/tweet_sentiment_extraction](https://huggingface.co/datasets/mteb/tweet_sentiment_extraction) |  | Classification | s2s | 1 | 27481 | 0 | 3534 | 68.3 | 0 | 67.8 |
| [ArxivClusteringP2P](https://www.kaggle.com/Cornell-University/arxiv) | [mteb/arxiv-clustering-p2p](https://huggingface.co/datasets/mteb/arxiv-clustering-p2p) | Clustering of titles+abstract from arxiv. Clustering of 30 sets, either on the main or secondary category | Clustering | p2p | 1 | 0 | 0 | 732723 | 0 | 0 | 1009.9 |
| [ArxivClusteringS2S](https://www.kaggle.com/Cornell-University/arxiv) | [mteb/arxiv-clustering-s2s](https://huggingface.co/datasets/mteb/arxiv-clustering-s2s) | Clustering of titles from arxiv. Clustering of 30 sets, either on the main or secondary category | Clustering | s2s | 1 | 0 | 0 | 732723 | 0 | 0 | 74.0 |
| [BiorxivClusteringP2P](https://api.biorxiv.org/) | [mteb/biorxiv-clustering-p2p](https://huggingface.co/datasets/mteb/biorxiv-clustering-p2p) | Clustering of titles+abstract from biorxiv. Clustering of 10 sets, based on the main category. | Clustering | p2p | 1 | 0 | 0 | 75000 | 0 | 0 | 1666.2 |
| [BiorxivClusteringS2S](https://api.biorxiv.org/) | [mteb/biorxiv-clustering-s2s](https://huggingface.co/datasets/mteb/biorxiv-clustering-s2s) | Clustering of titles from biorxiv. Clustering of 10 sets, based on the main category. | Clustering | s2s | 1 | 0 | 0 | 75000 | 0 | 0 | 101.6 |
| [MedrxivClusteringP2P](https://api.biorxiv.org/) | [mteb/medrxiv-clustering-p2p](https://huggingface.co/datasets/mteb/medrxiv-clustering-p2p) | Clustering of titles+abstract from medrxiv. Clustering of 10 sets, based on the main category. | Clustering | p2p | 1 | 0 | 0 | 37500 | 0 | 0 | 1981.2 |
| [MedrxivClusteringS2S](https://api.biorxiv.org/) | [mteb/medrxiv-clustering-s2s](https://huggingface.co/datasets/mteb/medrxiv-clustering-s2s) | Clustering of titles from medrxiv. Clustering of 10 sets, based on the main category. | Clustering | s2s | 1 | 0 | 0 | 37500 | 0 | 0 | 114.7 |
| [RedditClustering](https://arxiv.org/abs/2104.07081) | [mteb/reddit-clustering](https://huggingface.co/datasets/mteb/reddit-clustering) | Clustering of titles from 199 subreddits. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences. | Clustering | s2s | 1 | 0 | 0 | 420464 | 0 | 0 | 64.7 |
| [RedditClusteringP2P](https://huggingface.co/datasets/sentence-transformers/reddit-title-body) | [mteb/reddit-clustering-p2p](https://huggingface.co/datasets/mteb/reddit-clustering-p2p) | Clustering of title+posts from reddit. Clustering of 10 sets of 50k paragraphs and 40 sets of 10k paragraphs. | Clustering | p2p | 1 | 0 | 0 | 459399 | 0 | 0 | 727.7 |
| [StackExchangeClustering](https://arxiv.org/abs/2104.07081) | [mteb/stackexchange-clustering](https://huggingface.co/datasets/mteb/stackexchange-clustering) | Clustering of titles from 121 stackexchanges. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences. | Clustering | s2s | 1 | 0 | 417060 | 373850 | 0 | 56.8 | 57.0 |
| [StackExchangeClusteringP2P](https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_title_body_jsonl) | [mteb/stackexchange-clustering-p2p](https://huggingface.co/datasets/mteb/stackexchange-clustering-p2p) | Clustering of title+body from stackexchange. Clustering of 5 sets of 10k paragraphs and 5 sets of 5k paragraphs. | Clustering | p2p | 1 | 0 | 0 | 75000 | 0 | 0 | 1090.7 |
| [TwentyNewsgroupsClustering](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) | [mteb/twentynewsgroups-clustering](https://huggingface.co/datasets/mteb/twentynewsgroups-clustering) | Clustering of the 20 Newsgroups dataset (subject only). | Clustering | s2s | 1 | 0 | 0 | 59545 | 0 | 0 | 32.0 |
| [SprintDuplicateQuestions](https://www.aclweb.org/anthology/D18-1131/) | [mteb/sprintduplicatequestions-pairclassification](https://huggingface.co/datasets/mteb/sprintduplicatequestions-pairclassification) | Duplicate questions from the Sprint community. | PairClassification | s2s | 1 | 0 | 101000 | 101000 | 0 | 65.2 | 67.9 |
| [TwitterSemEval2015](https://alt.qcri.org/semeval2015/task1/) | [mteb/twittersemeval2015-pairclassification](https://huggingface.co/datasets/mteb/twittersemeval2015-pairclassification) | Paraphrase-Pairs of Tweets from the SemEval 2015 workshop. | PairClassification | s2s | 1 | 0 | 0 | 16777 | 0 | 0 | 38.3 |
| [TwitterURLCorpus](https://languagenet.github.io/) | [mteb/twitterurlcorpus-pairclassification](https://huggingface.co/datasets/mteb/twitterurlcorpus-pairclassification) | Paraphrase-Pairs of Tweets. | PairClassification | s2s | 1 | 0 | 0 | 51534 | 0 | 0 | 79.5 |
| [AskUbuntuDupQuestions](https://github.com/taolei87/askubuntu) | [mteb/askubuntudupquestions-reranking](https://huggingface.co/datasets/mteb/askubuntudupquestions-reranking) | AskUbuntu Question Dataset - Questions from AskUbuntu with manual annotations marking pairs of questions as similar or non-similar | Reranking | s2s | 1 | 0 | 0 | 2255 | 0 | 0 | 52.5 |
| [MindSmallReranking](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf) | [mteb/mind_small](https://huggingface.co/datasets/mteb/mind_small) | Microsoft News Dataset: A Large-Scale English Dataset for News Recommendation Research | Reranking | s2s | 1 | 231530 | 0 | 107968 | 69.0 | 0 | 70.9 |
| [SciDocsRR](https://allenai.org/data/scidocs) | [mteb/scidocs-reranking](https://huggingface.co/datasets/mteb/scidocs-reranking) | Ranking of related scientific papers based on their title. | Reranking | s2s | 1 | 0 | 19594 | 19599 | 0 | 69.4 | 69.0 |
| [StackOverflowDupQuestions](https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf) | [mteb/stackoverflowdupquestions-reranking](https://huggingface.co/datasets/mteb/stackoverflowdupquestions-reranking) | Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python | Reranking | s2s | 1 | 23018 | 0 | 3467 | 49.6 | 0 | 49.8 |
| [ArguAna](http://argumentation.bplaced.net/arguana/data) | [BeIR/arguana](https://huggingface.co/datasets/BeIR/arguana) | NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval | Retrieval | p2p | 1 | 0 | 0 | 10080 | 0 | 0 | 1052.9 |
| [ClimateFEVER](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html) | [BeIR/climate-fever](https://huggingface.co/datasets/BeIR/climate-fever) | CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change. | Retrieval | s2p | 1 | 0 | 0 | 5418128 | 0 | 0 | 539.1 |
| [CQADupstackAndroidRetrieval](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/) | [BeIR/cqadupstack/android](https://huggingface.co/datasets/BeIR/cqadupstack-qrels) | CQADupStack: A Benchmark Data Set for Community Question-Answering Research | Retrieval | s2p | 1 | 0 | 0 | 23697 | 0 | 0 | 578.7 |
| [CQADupstackEnglishRetrieval](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/) | [BeIR/cqadupstack/english](https://huggingface.co/datasets/BeIR/cqadupstack-qrels) | CQADupStack: A Benchmark Data Set for Community Question-Answering Research | Retrieval | s2p | 1 | 0 | 0 | 41791 | 0 | 0 | 467.1 |
| [CQADupstackGamingRetrieval](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/) | [BeIR/cqadupstack/gaming](https://huggingface.co/datasets/BeIR/cqadupstack-qrels) | CQADupStack: A Benchmark Data Set for Community Question-Answering Research | Retrieval | s2p | 1 | 0 | 0 | 46896 | 0 | 0 | 474.7 |
| [CQADupstackGisRetrieval](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/) | [BeIR/cqadupstack/gis](https://huggingface.co/datasets/BeIR/cqadupstack-qrels) | CQADupStack: A Benchmark Data Set for Community Question-Answering Research | Retrieval | s2p | 1 | 0 | 0 | 38522 | 0 | 0 | 991.1 |
| [CQADupstackMathematicaRetrieval](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/) | [BeIR/cqadupstack/mathematica](https://huggingface.co/datasets/BeIR/cqadupstack-qrels) | CQADupStack: A Benchmark Data Set for Community Question-Answering Research | Retrieval | s2p | 1 | 0 | 0 | 17509 | 0 | 0 | 1103.7 |
| [CQADupstackPhysicsRetrieval](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/) | [BeIR/cqadupstack/physics](https://huggingface.co/datasets/BeIR/cqadupstack-qrels) | CQADupStack: A Benchmark Data Set for Community Question-Answering Research | Retrieval | s2p | 1 | 0 | 0 | 39355 | 0 | 0 | 799.4 |
| [CQADupstackProgrammersRetrieval](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/) | [BeIR/cqadupstack/programmers](https://huggingface.co/datasets/BeIR/cqadupstack-qrels) | CQADupStack: A Benchmark Data Set for Community Question-Answering Research | Retrieval | s2p | 1 | 0 | 0 | 33052 | 0 | 0 | 1030.2 |
| [CQADupstackStatsRetrieval](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/) | [BeIR/cqadupstack/stats](https://huggingface.co/datasets/BeIR/cqadupstack-qrels) | CQADupStack: A Benchmark Data Set for Community Question-Answering Research | Retrieval | s2p | 1 | 0 | 0 | 42921 | 0 | 0 | 1041.0 |
| [CQADupstackTexRetrieval](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/) | [BeIR/cqadupstack/tex](https://huggingface.co/datasets/BeIR/cqadupstack-qrels) | CQADupStack: A Benchmark Data Set for Community Question-Answering Research | Retrieval | s2p | 1 | 0 | 0 | 71090 | 0 | 0 | 1246.9 |
| [CQADupstackUnixRetrieval](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/) | [BeIR/cqadupstack/unix](https://huggingface.co/datasets/BeIR/cqadupstack-qrels) | CQADupStack: A Benchmark Data Set for Community Question-Answering Research | Retrieval | s2p | 1 | 0 | 0 | 48454 | 0 | 0 | 984.7 |
| [CQADupstackWebmastersRetrieval](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/) | [BeIR/cqadupstack/webmasters](https://huggingface.co/datasets/BeIR/cqadupstack-qrels) | CQADupStack: A Benchmark Data Set for Community Question-Answering Research | Retrieval | s2p | 1 | 0 | 0 | 17911 | 0 | 0 | 689.8 |
| [CQADupstackWordpressRetrieval](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/) | [BeIR/cqadupstack/wordpress](https://huggingface.co/datasets/BeIR/cqadupstack-qrels) | CQADupStack: A Benchmark Data Set for Community Question-Answering Research | Retrieval | s2p | 1 | 0 | 0 | 49146 | 0 | 0 | 1111.9 |
| [DBPedia](https://github.com/iai-group/DBpedia-Entity/) | [BeIR/dbpedia-entity](https://huggingface.co/datasets/BeIR/dbpedia-entity) | DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base | Retrieval | s2p | 1 | 0 | 4635989 | 4636322 | 0 | 310.2 | 310.1 |
| [FEVER](https://fever.ai/) | [BeIR/fever](https://huggingface.co/datasets/BeIR/fever) | FEVER (Fact Extraction and VERification) consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. | Retrieval | s2p | 1 | 0 | 0 | 5423234 | 0 | 0 | 538.6 |
| [FiQA2018](https://sites.google.com/view/fiqa/) | [BeIR/fiqa](https://huggingface.co/datasets/BeIR/fiqa) | Financial Opinion Mining and Question Answering | Retrieval | s2p | 1 | 0 | 0 | 58286 | 0 | 0 | 760.4 |
| [HotpotQA](https://hotpotqa.github.io/) | [BeIR/hotpotqa](https://huggingface.co/datasets/BeIR/hotpotqa) | HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems. | Retrieval | s2p | 1 | 0 | 0 | 5240734 | 0 | 0 | 288.6 |
| [MSMARCO](https://microsoft.github.io/msmarco/) | [BeIR/msmarco](https://huggingface.co/datasets/BeIR/msmarco) | MS MARCO is a collection of datasets focused on deep learning in search. Note that the dev set is used for the leaderboard. | Retrieval | s2p | 1 | 0 | 8848803 | 8841866 | 0 | 336.6 | 336.8 |
| [MSMARCOv2](https://microsoft.github.io/msmarco/TREC-Deep-Learning.html) | [BeIR/msmarco-v2](https://huggingface.co/datasets/BeIR/msmarco-v2) | MS MARCO is a collection of datasets focused on deep learning in search | Retrieval | s2p | 1 | 138641342 | 138368101 | 0 | 341.4 | 342.0 | 0 |
| [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) | [BeIR/nfcorpus](https://huggingface.co/datasets/BeIR/nfcorpus) | NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval | Retrieval | s2p | 1 | 0 | 0 | 3956 | 0 | 0 | 1462.7 |
| [NQ](https://ai.google.com/research/NaturalQuestions/) | [BeIR/nq](https://huggingface.co/datasets/BeIR/nq) | NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval | Retrieval | s2p | 1 | 0 | 0 | 2684920 | 0 | 0 | 492.7 |
| [QuoraRetrieval](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) | [BeIR/quora](https://huggingface.co/datasets/BeIR/quora) | QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions. | Retrieval | s2s | 1 | 0 | 0 | 532931 | 0 | 0 | 62.9 |
| [SCIDOCS](https://allenai.org/data/scidocs) | [BeIR/scidocs](https://huggingface.co/datasets/BeIR/scidocs) | SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation. | Retrieval | s2p | 1 | 0 | 0 | 26657 | 0 | 0 | 1161.9 |
| [SciFact](https://github.com/allenai/scifact) | [BeIR/scifact](https://huggingface.co/datasets/BeIR/scifact) | SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts. | Retrieval | s2p | 1 | 0 | 0 | 5483 | 0 | 0 | 1422.3 |
| [Touche2020](https://webis.de/events/touche-20/shared-task-1.html) | [BeIR/webis-touche2020](https://huggingface.co/datasets/BeIR/webis-touche2020) | Touché Task 1: Argument Retrieval for Controversial Questions | Retrieval | s2p | 1 | 0 | 0 | 382594 | 0 | 0 | 1720.1 |
| [TRECCOVID](https://ir.nist.gov/covidSubmit/index.html) | [BeIR/trec-covid](https://huggingface.co/datasets/BeIR/trec-covid) | TRECCOVID is an ad-hoc search challenge based on the CORD-19 dataset containing scientific articles related to the COVID-19 pandemic | Retrieval | s2p | 1 | 0 | 0 | 171382 | 0 | 0 | 1117.4 |
| [BIOSSES](https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html) | [mteb/biosses-sts](https://huggingface.co/datasets/mteb/biosses-sts) | Biomedical Semantic Similarity Estimation. | STS | s2s | 1 | 0 | 0 | 200 | 0 | 0 | 156.6 |
| [SICK-R](https://www.aclweb.org/anthology/S14-2001.pdf) | [mteb/sickr-sts](https://huggingface.co/datasets/mteb/sickr-sts) | Semantic Textual Similarity SICK-R dataset as described here: | STS | s2s | 1 | 0 | 0 | 19854 | 0 | 0 | 46.1 |
| [STS12](https://www.aclweb.org/anthology/S12-1051.pdf) | [mteb/sts12-sts](https://huggingface.co/datasets/mteb/sts12-sts) | SemEval STS 2012 dataset. | STS | s2s | 1 | 4468 | 0 | 6216 | 100.7 | 0 | 64.7 |
| [STS13](https://www.aclweb.org/anthology/S13-1004/) | [mteb/sts13-sts](https://huggingface.co/datasets/mteb/sts13-sts) | SemEval STS 2013 dataset. | STS | s2s | 1 | 0 | 0 | 3000 | 0 | 0 | 54.0 |
| [STS14](http://alt.qcri.org/semeval2014/task10/) | [mteb/sts14-sts](https://huggingface.co/datasets/mteb/sts14-sts) | SemEval STS 2014 dataset. Currently only the English dataset | STS | s2s | 1 | 0 | 0 | 7500 | 0 | 0 | 54.3 |
| [STS15](http://alt.qcri.org/semeval2015/task2/) | [mteb/sts15-sts](https://huggingface.co/datasets/mteb/sts15-sts) | SemEval STS 2015 dataset | STS | s2s | 1 | 0 | 0 | 6000 | 0 | 0 | 57.7 |
| [STS16](http://alt.qcri.org/semeval2016/task1/) | [mteb/sts16-sts](https://huggingface.co/datasets/mteb/sts16-sts) | SemEval STS 2016 dataset | STS | s2s | 1 | 0 | 0 | 2372 | 0 | 0 | 65.3 |
| [STS17](http://alt.qcri.org/semeval2016/task1/) | [mteb/sts17-crosslingual-sts](https://huggingface.co/datasets/mteb/sts17-crosslingual-sts) | STS 2017 dataset | STS | s2s | 11 | 0 | 0 | 500 | 0 | 0 | 43.3 |
| [STS22](https://competitions.codalab.org/competitions/33835) | [mteb/sts22-crosslingual-sts](https://huggingface.co/datasets/mteb/sts22-crosslingual-sts) | SemEval 2022 Task 8: Multilingual News Article Similarity | STS | s2s | 18 | 0 | 0 | 8060 | 0 | 0 | 1992.8 |
| [STSBenchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) | [mteb/stsbenchmark-sts](https://huggingface.co/datasets/mteb/stsbenchmark-sts) | Semantic Textual Similarity Benchmark (STSbenchmark) dataset. | STS | s2s | 1 | 11498 | 3000 | 2758 | 57.6 | 64.0 | 53.6 |
| [SummEval](https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html) | [mteb/summeval](https://huggingface.co/datasets/mteb/summeval) | Biomedical Semantic Similarity Estimation. | Summarization | s2s | 1 | 0 | 0 | 2800 | 0 | 0 | 359.8 |


## Citation

If you find MTEB useful, feel free to cite our publication [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316):

```bibtex
@article{muennighoff2022mteb,
  doi = {10.48550/ARXIV.2210.07316},
  url = {https://arxiv.org/abs/2210.07316},
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Lo{\"\i}c and Reimers, Nils},
  title = {MTEB: Massive Text Embedding Benchmark},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2210.07316},  
  year = {2022}
}
```
