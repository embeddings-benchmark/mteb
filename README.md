# Massive Text Embedding Benchmark
Massive Text Embedding Benchmark - Internal Development Git

## Minimal use

````python
from mteb import MTEB

model = ...
eval = MTEB()
eval.run(model)
````

## Details

### Tasks selection

Tasks can be selected by their types, their categories, or directly by providing the list of tasks that needs to be run.

````python
eval = MTEB(task_types=['Reranking', 'Clustering']) # Clustering and Reranking tasks
````

````python
eval = MTEB(task_categories=['S2S']) # Only select sentence2sentence tasks
````

````python
eval = MTEB(task_list=['RedditClustering', 'StackExchangeClustering', 'TwitterSemEval2015BC])
````

The list of available Tasks / Tasks types / Tasks categories can be found with:

````python
eval = MTEB()
print(eval.available_tasks)
print(eval.available_task_types)
print(eval.available_task_categories)
````

### Models

Models should implement the following interface, implementing an `encode` function taking as inputs a list of sentences, and returning a list of embeddings (embeddings can be `np.array`, `torch.tensor`, etc.).

````python
class Model(ABC):
    @abstractmethod
    def encode(self, sentences: List[str]) -> List[List[float]]:
        pass
````
