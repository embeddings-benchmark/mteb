To make a MIEB task `X` fully runable from scratch for dataset `Y` with model `Z`, we need to implemenet an `AbsTaskX` class for it, subclassing `AbsTask`; a task-specific `XEvaluator`, which will be called in `AbsTaskX`; a dataset-specific (e.g., Dataset `Y`) class `class Y(AbsTaskX)` subclassing the corresponding `AbsTaskX`, which is itself the subclass of `AbsTask`; and some model class `ZModelWrapper` that has needed functions.

## Example

Here is an example implementing zero-shot image classification from scratch.

To solve this task, we basically need to encode the `images`, encode the `class label candidates with prompts` (things like "this is a dog pic", "this is a cat pic"), and similarity-compare them, to argmax out the class prediction for each image.

#### ModelWrapper
Since we don't have an established class like `SentenceTransformer` or `DRES` anymore now, we first decide for this task so far, we need the model class to have `get_text_embeddings`, `get_image_embeddings`, and `calculate_probs`. As an example,  [CLIPModelWrapper](https://github.com/embeddings-benchmark/mteb/blob/mieb/mteb/models/clip_models.py) is first implemented, with MetaData defined.

#### X Evaluator
With the model, [ZeroshotClassificationEvaluator](https://github.com/embeddings-benchmark/mteb/blob/mieb/mteb/evaluation/evaluators/Image/ZeroshotClassificationEvaluator.py) is implemented here, basically the pipeline of using the defined models to do zero-shot classification.

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

