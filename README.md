# Massive Text Embedding Benchmark

Massive Text Embedding Benchmark - Internal Development Git

## Installation

```bash
pip install git+https://github.com/embeddings-benchmark/mteb-draft.git
```

## Minimal use

* Using a python script:

````python
from mteb import MTEB
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("average_word_embeddings_komninos")
evaluation = MTEB(tasks=["Banking77Classification"])
evaluation.run(model)
````

* Using CLI

```bash
mteb --available_tasks

mteb -m average_word_embeddings_komninos \
    -t Banking77Classification NFCorpus \
    --output_folder results \
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

You can also specify which languages to load for multilingual/crosslingual tasks like this:

````python
from mteb.tasks.BitextMining import BUCCBitextMining

evaluation = MTEB(tasks=[
        BUCCBitextMining(langs=["de-en"]), # Only load "de-en" and fr-en" subsets of BUCC
        AmazonReviewsClassification(langs=["en", "fr"]) # Only load "en" and "fr" subsets of Amazon Reviews
])
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

| Name                                   | Task                | Category   | Language                 |
|:---------------------------------------|:--------------------|:-----------|:-------------------------|
| ArxivClusterings2s                     | Clustering          | s2s        | English                  |
| ArxivClusteringp2p                     | Clustering          | p2p        | English                  |
| BiorxivClusterings2s                   | Clustering          | s2s        | English                  |
| BiorxivClusteringp2p                   | Clustering          | p2p        | English                  |
| MedrxivClusterings2s                   | Clustering          | s2s        | English                  |
| MedrxivClusteringp2p                   | Clustering          | p2p        | English                  |
| StackExchangeClustering                | Clustering          | s2s        | English                  |
| StackExchangeClusteringp2p             | Clustering          | p2p        | English                  |
| RedditClustering                       | Clustering          | s2s        | English                  |
| RedditClusteringp2p                    | Clustering          | p2p        | English                  |
| TwentyNewsgroupsClustering             | Clustering          | s2s        | English                  |
| AskUbuntuDupQuestions                  | Reranking           | s2s        | English                  |
| SciDocsReranking                       | Reranking           | s2s        | English                  |
| StackOverflowDupQuestions              | Reranking           | s2s        | English                  |
| MindSmallReranking                     | Reranking           | s2s        | English                  |
| BiossesSTS                             | STS                 | s2s        | English                  |
| SickrSTS                               | STS                 | s2s        | English                  |
| STS12STS                               | STS                 | s2s        | English                  |
| STS13STS                               | STS                 | s2s        | English                  |
| STS14STS                               | STS                 | s2s        | English                  |
| STS15STS                               | STS                 | s2s        | English                  |
| STS16STS                               | STS                 | s2s        | English                  |
| STS17CrossLingualSTS                   | STS                 | s2s        | Crosslingual (10 pairs)  |
| STS22CrossLingualSTS                   | STS                 | s2s        | Crosslingual (18 pairs)  |
| STSBenchmarkSTS                        | STS                 | s2s        | English                  |
| BUCCBitextMining                       | BitextMining        | s2s        | Crosslingual (4 pairs)   |
| TatoebaBitextMining                    | BitextMining        | s2s        | Crosslingual (112 pairs) |
| SprintDuplicateQuestionsBC             | Pair Classification | s2s        | English                  |
| TwitterSemEval2015BC                   | Pair Classification | s2s        | English                  |
| TwitterURLCorpusBC                     | Pair Classification | s2s        | English                  |
| AmazonCounterfactualClassification     | Classification      | s2s        | Multilingual (4 langs)   |
| AmazonPolarityClassification           | Classification      | s2s        | English                  |
| AmazonReviewsClassification            | Classification      | s2s        | Multilingual (6 langs)   |
| Banking77Classification                | Classification      | s2s        | English                  |
| EmotionClassification                  | Classification      | s2s        | English                  |
| ImdbClassification                     | Classification      | p2p        | English                  |
| MassiveIntentClassification            | Classification      | s2s        | Multilingual (51 langs)  |
| MassiveScenarioClassification          | Classification      | s2s        | Multilingual (51 langs)  |
| MTOPDomainClassification               | Classification      | s2s        | Multilingual (6 langs)   |
| MTOPIntentClassification               | Classification      | s2s        | Multilingual (6 langs)   |
| ToxicConversationsClassification       | Classification      | s2s        | English                  |
| TweetSentimentExtractionClassification | Classification      | s2s        | English                  |
| ArguAna                                | Retrieval           | s2s        | English                  |
| ClimateFEVER                           | Retrieval           | s2s        | English                  |
| CQADupstackRetrieval                   | Retrieval           | s2s        | English                  |
| DBPedia                                | Retrieval           | s2s        | English                  |
| FEVER                                  | Retrieval           | s2s        | English                  |
| FiQA2018                               | Retrieval           | s2s        | English                  |
| HotpotQA                               | Retrieval           | s2s        | English                  |
| MSMARCO                                | Retrieval           | s2s        | English                  |
| MSMARCOv2                              | Retrieval           | s2s        | English                  |
| NFCorpus                               | Retrieval           | s2s        | English                  |
| NQ                                     | Retrieval           | s2s        | English                  |
| QuoraRetrieval                         | Retrieval           | s2s        | English                  |
| SCIDOCS                                | Retrieval           | s2s        | English                  |
| SciFact                                | Retrieval           | s2s        | English                  |
| Touche2020                             | Retrieval           | s2s        | English                  |
| TRECCOVID                              | Retrieval           | s2s        | English                  |
| SummEval                               | Summarization       | s2s        | English                  |
