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
