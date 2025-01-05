<h1 align="center">Gene-MTEB Benchmark</h1>

## Installation

```bash
pip install mteb
```

## Example Usage

* Using a Python script:

```python
import mteb
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
model_name = "average_word_embeddings_komninos"

model = mteb.get_model(model_name) # if the model is not implemented in MTEB it will be eq. to SentenceTransformer(model_name)
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")
```
