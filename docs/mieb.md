# Welcome to MIEB! 👋

The Massive Image Embedding Benchmark (MIEB) is an image extension of [MTEB](https://arxiv.org/abs/2210.07316) to cover embedding tasks for image-text tasks. 

## Background

MIEB intends to extend MTEB and MMTEB to cover image representation learning and image-text alignment tasks. 

## Contributing to MIEB

To make a MIEB task `X` fully runable from scratch for dataset `Y` with model `Z`, we need to implemenet an `AbsTaskX` class for it, which inherits from `AbsTask`. A task-specific `XEvaluator`, which will be called in `AbsTaskX`; a dataset-specific (e.g., Dataset `Y`) class `class Y(AbsTaskX)` inheriting from the corresponding `AbsTaskX`; and some model class `ZModelWrapper` that has needed functions.

## Example

Here is an example implementing zero-shot image classification from scratch.

To solve this task, we basically need to encode the `images`, encode the `class label candidates with prompts` (things like "this is a dog pic", "this is a cat pic"), and similarity-compare them, to argmax out the class prediction for each image.

#### Model Wrapper
See the [`ImageEncoder` class](https://github.com/embeddings-benchmark/mteb/blob/mieb/mteb/encoder_interface.py). The model class implements `get_text_embeddings`, `get_image_embeddings`, and `calculate_probs` methods. 
As an example,  [`CLIPModelWrapper`](https://github.com/embeddings-benchmark/mteb/blob/mieb/mteb/models/clip_models.py) is first implemented, with metadata defined.


#### X Evaluator
With the model, [ZeroshotClassificationEvaluator](https://github.com/embeddings-benchmark/mteb/blob/mieb/mteb/evaluation/evaluators/Image/ZeroshotClassificationEvaluator.py) is implemented here, basically the pipeline of using the defined models to do zero-shot classification.

```python
class ZeroshotClassificationEvaluator(Evaluator):
    def __init__(self, ...):
        pass
    def __call__(self, model: Encoder, *, encode_kwargs: dict[str, Any] = {}):
        """Get embeddings and calculate scores."""
```

#### AbsTask X
With the evaluator, [AbsTaskZeroshotClassification](https://github.com/embeddings-benchmark/mteb/blob/mieb/mteb/abstasks/Image/AbsTaskZeroshotClassification.py) is defined, operating on the dataset, calling the defined Evaluator, and gives out results.


#### Dataset class
With all these, we can then define the dataset. Here I choose Rendered SST2 as an example, which is to classify SST2 movie reviews, with reviews rendered into images. [RenderedSST2](https://github.com/embeddings-benchmark/mteb/blob/mieb/mteb/tasks/Image/ZeroshotClassification/eng/RenderedSST2.py) is implemented like this, subclassing `AbsTaskZeroshotClassification`, and overwrite the `get_candidate_labels` function, which gives `["a negative review of a movie", "a positive review of a movie"]` to be used in the evaluator.

With all these, we can then 
```python

import mteb

model_name = "openai/clip-vit-large-patch14"
model = mteb.get_model(model_name = model_name)

tasks = mteb.get_tasks(tasks=["RenderedSST2"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results-mieb/{model_name}")
```
And yeah, the results will be under [`results-mieb/openai/clip-vit-large-patch14`](https://github.com/embeddings-benchmark/mteb/blob/mieb/results-mieb/openai__clip-vit-large-patch14/32bd64288804d66eefd0ccbe215aa642df71cc41/RenderedSST2.json) and look legit with an `"accuracy": 0.6979681493684788,`, a bit higher than the original CLIP paper but might be resolution/layout difference of images in the remake of the dataset by the CLIP benchmark team.


## Specific Model running Instructions

Some models require some specific steps before running. Those are collected here.

<details>

    <summary> Vista </summary>

    ## set up VISTA 

    the latest FlagEmbedding repo doesn't support VISTA anymore so we use a old version.
    ```
    git clone --no-checkout https://github.com/FlagOpen/FlagEmbedding.git
    cd FlagEmbedding
    git checkout 5c9260277977f8f8e256e56a8e12387552693af9
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