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
    --output_folder mteb_output \
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
