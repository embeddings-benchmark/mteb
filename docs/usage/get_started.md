# Get Started

This usage documentation first introduces a simple example of how to evaluate a model using `mteb`.
Then introduces model detailed section of [defining model](defining_the_model.md), [selecting tasks](selecting_tasks.md) and [running the evaluation](./running_the_evaluation.md). Each section contains subsections pertaining to these.


## Evaluating a Model

Evaluating a model on MTEB follows a three step approach, 1) defining model, 2) selecting the tasks and 3) running the evaluation

```python
import mteb

# Specify the model that we want to evaluate
model = ...

# specify what you want to evaluate it on
tasks = mteb.get_tasks(tasks=["{task1}", "{task1}"])

# run the evaluation
results = mteb.evaluate(model, tasks=tasks)
```

For instance if we want to run [`"sentence-transformers/all-MiniLM-L6-v2"`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) on
`"Banking77Classification"` we can do this using the following code:

```python
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# load the model using MTEB
model = mteb.get_model(model_name) # will default to SentenceTransformers(model_name) if not implemented in MTEB
# or using SentenceTransformers
model = SentenceTransformers(model_name)

# select the desired tasks and evaluate
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
results = mteb.evaluate(model, tasks=tasks)
```

!!! Note
    While `mteb.evaluate` supports `SentenceTransformers` we do recommend that the user use `mteb.get_model` to fetch the model as this prioritizes the
    implementation in `mteb`, which might not match 1-1 its `SentenceTransformers` implementation. For leaderboards results we see the `mteb`
    implementation as the reference implementation.


## Evaluating on Different Modalities

MTEB is not only text evaluating, but also allow you to evaluate image and image-text embeddings.

!!! Note
    Running MTEB on images requires you to install the optional dependencies using `pip install mteb[image]`

To evaluate image embeddings you can follow the same approach for any other task in `mteb`. Simply ensuring that the task contains the modality "image":

```python
tasks = mteb.get_tasks(modalities=["image"]) # Only select tasks with image modalities
task = task[0]

print(task.metadata.modalites)
# ['text', 'image']
```

However, we recommend starting with one of the predefined benchmarks:

```python
import mteb
benchmark = mteb.get_benchmark("MIEB(eng)")
model = mteb.get_model("openai/clip-vit-base-patch16") # example model

results = mteb.evaluate(model, tasks=benchmark)
```

You can also specify exclusive modality filtering to only get tasks with exactly the requested modalities (default behavior with `exclusive_modality_filter=False`):

```python
# Get tasks with image modality, this will also include tasks having both text and image modalities
tasks = mteb.get_tasks(modalities=["image"], exclusive_modality_filter=False)

# Get tasks that have ONLY image modality
tasks = mteb.get_tasks(modalities=["image"], exclusive_modality_filter=True)
```
