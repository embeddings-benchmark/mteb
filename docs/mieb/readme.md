# Welcome to MIEB! 👋

The [Massive Image Embedding Benchmark (MIEB)](https://arxiv.org/abs/2504.10471) is an image extension of [MTEB](https://arxiv.org/abs/2210.07316) to cover embedding tasks for image-text tasks.

## 🌱 Background

MIEB intends to extend MTEB and MMTEB to cover image representation learning and image-text alignment tasks. At the time of publishing, MIEB offers 130 tasks over 8 task categories. 3 benchmarks are offered:
1. `MIEB(Multilingual)`
2. `MIEB(eng)`
3. `MIEB(lite)`

## 🚀 Running MIEB

If you’re already familiar with how MTEB works, then run any benchmark, task, and model the same way!


### Run MIEB in 2 lines via CLI
First, install the `mieb` dependencies:
```sh
pip install mteb[image]
```

Then, run the multilingual benchmark with a selected model, e.g. CLIP:
```sh
mteb run -b ‘MIEB(Multilingual)’ -m openai/clip-vit-base-patch16
```

### Run MIEB in Python

Similarly, running the benchmark can be done in Python in 3 main steps: 1) Select the tasks, load the model, and run the evaluation.

1. Select the whole benchmark
```python
import mteb

tasks = mteb.get_benchmarks("MIEB(Multilingual)")
```

Alternatively, select a single task:
```python
tasks = mteb.get_tasks(tasks=["CIFAR10ZeroShot"])
```

Or select tasks by categories:
```python
tasks = mteb.get_tasks(task_types=["Compositionality"])
```

2. Load a Model:

```python
model_name = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
model = mteb.get_model(model_name=model_name)
```

3. Run the Evaluation:

```python
results = mteb.evaluate(model, tasks=tasks)
```


## 🪴 Contributing to MIEB

The FIRST step is to _always_ create an issue in the MTEB repo (this one), and add the `mieb` label. PRs without issues will not be accepted.

There are a few ways for anyone to contribute to MIEB:

  1. Add a dataset as an existing task type. This means that the `AbsTask` already exists, e.g. `AbsTaskImageClassification`, and the effort is solely in adding an instance of it.
  2.  Add a model. This could mean either: a) The model wrapper, e.g. `OpenCLIPWrapper`, already exists, and the effort is solely in adding a filled out `ModelMeta` object, and/or b) Add a new model wrapper.
  3. Add a new task type. This means that the existing task types do not cover this new task. An accompanying evaluator should also be implemented.

Let's go through an example.

<details>
  <summary> Contribution Example (click to unfold) </summary>

### Example

Here is an example implementing a zero-shot image classification from scratch. Let's say we wish to implement CIFAR10 as a task and evaluate an OpenCLIP model on it.

To solve this task, we need to encode the `images`, encode the `class label candidates with prompts` (e.g. "this is a dog pic", "this is a cat pic"), and compare them by calculating similarity, and then argmax out the class prediction for each image. We begin by implementing a model wrapper.

#### Model Wrapper
See the [`EncoderProtocol`](https://embeddings-benchmark.github.io/mteb/api/model/?h=encoder+protocol#mteb.models.EncoderProtocol) for more details. The model class implements `encode()` and `similarity()` methods for text and image embeddings.
As an example, [`OpenCLIPWrapper`](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/model_implementations/openclip_models.py) is implemented with the required interface.
```python
class OpenCLIPWrapper:
    ...
```
See also [adding a model](../contributing/adding_a_model.md) for reference.

#### X Evaluator
With the model, [ZeroShotClassificationEvaluator](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/_evaluators/zeroshot_classification_evaluator.py) is implemented. This defines how the model is used to do zero-shot classification and get back results on desired metrics.
```python
class ZeroShotClassificationEvaluator(Evaluator):
    def __init__(
        self,
        dataset: Dataset,
        input_column_name: str,
        candidate_labels: list[str],
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        **kwargs,
    ) -> None:
        ...

    def __call__(self, model: Encoder, *, encode_kwargs: dict[str, Any] = {}):
        """Get embeddings and calculate scores."""
        ...
```

#### AbsTask X
With the evaluator, [AbsTaskZeroShotClassification](https://embeddings-benchmark.github.io/mteb/api/task/#mteb.abstasks.zeroshot_classification.AbsTaskZeroShotClassification) is defined. In MTEB v2, the dataset is loaded via `self.load_data()` and stored in `self.dataset` with the structure `self.dataset[hf_subset][split]` where `hf_subset` is the language/subset (e.g., "eng") and `split` is "train"/"test"/"validation".
```python
class AbsTaskZeroShotClassification(AbsTask):
    # Default column names - override if your dataset uses different names
    input_column_name: str = "image"
    label_column_name: str = "label"

    def _evaluate_subset(
        self,
        model: EncoderProtocol,
        data_split: Dataset,
        *,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
        **kwargs,
    ) -> ZeroShotClassificationMetrics:
        # Select relevant columns and pass to evaluator
        data_split = data_split.select_columns(
            [self.input_column_name, self.label_column_name]
        )
        evaluator = ZeroShotClassificationEvaluator(
            data_split,
            self.input_column_name,
            self.get_candidate_labels(),
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            **kwargs,
        )
        probs = evaluator(model, encode_kwargs=encode_kwargs)
        return self._calculate_scores(
            data_split[self.label_column_name],
            torch.tensor(probs).argmax(dim=1).tolist(),
        )
    ...
```


#### Dataset class
With all these, we can then define the dataset. [CIFAR10](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/zeroshot_classification/eng/cifar.py) is implemented by subclassing `AbsTaskZeroShotClassification` and overriding:
- `metadata`: TaskMetadata with `modalities=["text", "image"]` and `category="i2t"` (image-to-text)
- `input_column_name`: Name of the column containing images (default: "image")
- `get_candidate_labels()`: Returns text prompts for each class label
```python
class CIFAR10ZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="CIFAR10ZeroShot",
        type="ZeroShotClassification",
        category="i2t",  # image-to-text classification
        modalities=["text", "image"],  # Required for MIEB tasks
        dataset={"path": "uoft-cs/cifar10", "revision": "..."},
        ...
    )
    input_column_name: str = "img"  # CIFAR10 uses "img" column

    def get_candidate_labels(self) -> list[str]:
        # Access labels via self.dataset after load_data() is called
        # self.label_column_name defaults to "label"
        return [
            f"a photo of a {name}."
            for name in self.dataset["test"].features[self.label_column_name].names
        ]
```
See also [adding a dataset](../contributing/adding_a_dataset.md) for reference.

#### Putting them all together
With all these, we can then
```python
import mteb

model_name = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
model = mteb.get_model(model_name=model_name)

tasks = mteb.get_tasks(tasks=["CIFAR10ZeroShot"])
results = mteb.evaluate(model, tasks=tasks)
```

By default, results will be under `results/laion__CLIP-ViT-L-14-laion2B-s32B-b82K/REVISION/CIFAR10ZeroShot.json`. Sometimes metrics can be a bit different than what the original paper claimed. This might be due to the resolution/layout difference of images in the remake of the dataset.

</details>

## Citing

When using `mieb`, we recommend you use the following citation:

```bibtex
@article{xiao2025mieb,
  author = {Chenghao Xiao and Isaac Chung and Imene Kerboua and Jamie Stirling and Xin Zhang and Márton Kardos and Roman Solomatin and Noura Al Moubayed and Kenneth Enevoldsen and Niklas Muennighoff},
  title = {MIEB: Massive Image Embedding Benchmark},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2504.10471},
  year = {2025},
  url = {https://arxiv.org/abs/2504.10471},
  doi = {10.48550/ARXIV.2504.10471},
}
```
