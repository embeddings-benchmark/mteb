# Welcome to MIEB! 👋

The Massive Image Embedding Benchmark (MIEB) is an image extension of [MTEB](https://arxiv.org/abs/2210.07316) to cover embedding tasks for image-text tasks. 

## 🌱 Background

MIEB intends to extend MTEB and MMTEB to cover image representation learning and image-text alignment tasks. 

## 🪴 Contributing to MIEB

The FIRST step is to _always_ create an issue in the MTEB repo (this one), and add the `mieb` label. PRs without issues will not be accepted. 

There are a few ways for anyone to contribute to MIEB:

  1. Add a dataset as an existing task type. This means that the `AbsTask` already exists, e.g. `AbsTaskImageClassification`, and the effort is solely in adding an instance of it. 
  2.  Add a model. This could mean either: a) The model wrapper, e.g. `OpenCLIPWrapper`, already exists, and the effort is solely in adding a filled out `ModelMeta` object, and/or b) Add a new model wrapper.
  3. Add a new task type. This means that the existing task types do not cover this new task. An accompanying evaluator should also be implemented.

Let's go through an example.

## Example

Here is an example implementing a zero-shot image classification from scratch. Let's say we wish to implement CIFAR10 as a task and evaluate an OpenCLIP model on it. 

To solve this task, we need to encode the `images`, encode the `class label candidates with prompts` (e.g. "this is a dog pic", "this is a cat pic"), and compare them by calculating similarity, and then argmax out the class prediction for each image. We begin by implementing a model wrapper. 

### Model Wrapper
See the [`ImageEncoder` class](https://github.com/embeddings-benchmark/mteb/blob/mieb/mteb/encoder_interface.py) for more details. The model class implements `get_text_embeddings`, `get_image_embeddings`, and `calculate_probs` methods. 
As an example,  [`OpenCLIPWrapper`](https://github.com/embeddings-benchmark/mteb/blob/mieb/mteb/models/openclip_models.py) is first implemented, with metadata defined below.
```python
class OpenCLIPWrapper:
    ...
```
See also [adding a model](adding_a_model.md) for reference.

### X Evaluator
With the model, [ZeroshotClassificationEvaluator](https://github.com/embeddings-benchmark/mteb/blob/mieb/mteb/evaluation/evaluators/Image/ZeroshotClassificationEvaluator.py) is implemented here. This defines how the model are used to do zero-shot classification and get back results on desired metrics.
```python
class ZeroshotClassificationEvaluator(Evaluator):
    def __init__(self, ...):
        ...
    def __call__(self, model: Encoder, *, encode_kwargs: dict[str, Any] = {}):
        """Get embeddings and calculate scores."""
        ...
```

### AbsTask X
With the evaluator, [AbsTaskZeroshotClassification](https://github.com/embeddings-benchmark/mteb/blob/mieb/mteb/abstasks/Image/AbsTaskZeroshotClassification.py) is defined, operating on the dataset, calling the defined Evaluator, and gives out results.
```python
class AbsTaskZeroshotClassification(AbsTask):
    ...
```


### Dataset class
With all these, we can then define the dataset. [CIFAR10](https://github.com/embeddings-benchmark/mteb/blob/mieb/mteb/tasks/Image/ZeroshotClassification/eng/CIFAR.py) is implemented like this, subclassing `AbsTaskZeroshotClassification`, and overwrite the `get_candidate_labels` function, which gives `["a photo of {label_name}"]` to be used in the evaluator.
```python
class CIFAR10ZeroShotClassification(AbsTaskZeroshotClassification):
    metadata = TaskMetadata(...)

    def get_candidate_labels(self) -> list[str]:
        ...
```
See also [adding a dataset](adding_a_dataset.md) for reference.

### Putting them all together
With all these, we can then 
```python
import mteb

model_name = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
model = mteb.get_model(model_name=model_name)

tasks = mteb.get_tasks(tasks=["CIFAR10ZeroShot"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model)
```

By default, results will be under `results/laion__CLIP-ViT-L-14-laion2B-s32B-b82K/REVISION/CIFAR10ZeroShot.json`. Sometimes metrics can be a bit different than what the original paper claimed. This might be due to the resolution/layout difference of images in the remake of the dataset.


## Specific Model running Instructions

Some models require some specific steps before running. Those are collected here.

<details>
    <summary> Vista </summary>

    ## set up VISTA 

    ```
    git clone https://github.com/FlagOpen/FlagEmbedding.git
    cd FlagEmbedding/research/visual_bge
    pip install -e .
    pip install torchvision timm einops ftfy
    ```
    back to the root folder of mteb; download the vision tower for bge-base
    ```
    cd ..
    wget https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_base_en_v1.5.pth?download=true
    ```
    rename it to `visualized_base_en_V1.5.pth`
    ```
    mv Visualized_base_en_v1.5.pth?download=true visualized_base_en_V1.5.pth
    ```
    download the vision tower for bge-m3
    ```
    wget https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_m3.pth?download=true
    ```
    rename it to `visualized_m3.pth`
    ```
    mv Visualized_m3.pth?download=true visualized_m3.pth
    ```


</details>