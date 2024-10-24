<h1 align="center">Massive Text Embedding Benchmark</h1>

<p align="center">
    <a href="https://github.com/embeddings-benchmark/mteb/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/embeddings-benchmark/mteb.svg">
    </a>
    <a href="https://arxiv.org/abs/2210.07316">
        <img alt="GitHub release" src="https://img.shields.io/badge/arXiv-2305.14251-b31b1b.svg">
    </a>
    <a href="https://github.com/embeddings-benchmark/mteb/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/embeddings-benchmark/mteb.svg?color=green">
    </a>
    <a href="https://pepy.tech/project/mteb">
        <img alt="Downloads" src="https://static.pepy.tech/personalized-badge/mteb?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="#installation">Installation</a> |
        <a href="#usage-documentation">Usage</a> |
        <a href="https://huggingface.co/spaces/mteb/leaderboard">Leaderboard</a> |
        <a href="#documentation">Documentation</a> |
        <a href="#citing">Citing</a>
    <p>
</h4>

<h3 align="center">
    <a href="https://huggingface.co/spaces/mteb/leaderboard"><img style="float: middle; padding: 10px 10px 10px 10px;" width="60" height="55" src="./docs/images/hf_logo.png" /></a>
</h3>


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
# or directly from huggingface:
# model_name = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(model_name)
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")
```

<details>
  <summary> Running SentneceTransformermer model with prompts </summary>

Prompts can be passed to the SentenceTransformer model using the `prompts` parameter. The following code shows how to use prompts with SentenceTransformer:

```python
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("average_word_embeddings_komninos", prompts={"query": "Query:", "passage": "Passage:"})
evaluation = mteb.MTEB(tasks=tasks)
```

In prompts the key can be:
1. Prompt types (`passage`, `query`) - they will be used in reranking and retrieval tasks 
2. Task type - these prompts will be used in all tasks of the given type
   1. `BitextMining`
   2. `Classification`
   3. `MultilabelClassification`
   4. `Clustering`
   5. `PairClassification`
   6. `Reranking`
   7. `Retrieval`
   8. `STS`
   9. `Summarization`
   10. `InstructionRetrieval`
3. Pair of task type and prompt type like `Retrival-query` - these prompts will be used in all classification tasks
4. Task name - these prompts will be used in the specific task
5. Pair of task name and prompt type like `NFCorpus-query` - these prompts will be used in the specific task
</details>

* Using CLI

```bash
mteb available_tasks

mteb run -m sentence-transformers/all-MiniLM-L6-v2 \
    -t Banking77Classification  \
    --verbosity 3

# if nothing is specified default to saving the results in the results/{model_name} folder
```

* Using multiple GPUs in parallel can be done by just having a custom encode function that distributes the inputs to multiple GPUs like e.g. [here](https://github.com/microsoft/unilm/blob/b60c741f746877293bb85eed6806736fc8fa0ffd/e5/mteb_eval.py#L60) or [here](https://github.com/ContextualAI/gritlm/blob/09d8630f0c95ac6a456354bcb6f964d7b9b6a609/gritlm/gritlm.py#L75).



## Usage Documentation
Click on each section below to see the details.

<br /> 

<details>
  <summary>  Task selection </summary>

### Task selection

Tasks can be selected by providing the list of datasets, but also

* by their task (e.g. "Clustering" or "Classification")

```python
tasks = mteb.get_tasks(task_types=["Clustering", "Retrieval"]) # Only select clustering and retrieval tasks
```

* by their categories e.g. "s2s" (sentence to sentence) or "p2p" (paragraph to paragraph)

```python
tasks = mteb.get_tasks(categories=["s2s", "p2p"]) # Only select sentence2sentence and paragraph2paragraph datasets
```

* by their languages

```python
tasks = mteb.get_tasks(languages=["eng", "deu"]) # Only select datasets which contain "eng" or "deu" (iso 639-3 codes)
```

You can also specify which languages to load for multilingual/cross-lingual tasks like below:

```python
import mteb

tasks = [
    mteb.get_task("AmazonReviewsClassification", languages = ["eng", "fra"]),
    mteb.get_task("BUCCBitextMining", languages = ["deu"]), # all subsets containing "deu"
]

# or you can select specific huggingface subsets like this:
from mteb.tasks import AmazonReviewsClassification, BUCCBitextMining

evaluation = mteb.MTEB(tasks=[
        AmazonReviewsClassification(hf_subsets=["en", "fr"]) # Only load "en" and "fr" subsets of Amazon Reviews
        BUCCBitextMining(hf_subsets=["de-en"]), # Only load "de-en" subset of BUCC
])
# for an example of a HF subset see "Subset" in the dataset viewer at: https://huggingface.co/datasets/mteb/bucc-bitext-mining
```

</details>

<details>
  <summary>  Running a benchmark </summary>

### Running a Benchmark

`mteb` comes with a set of predefined benchmarks. These can be fetched using `get_benchmark` and run in a similar fashion to other sets of tasks. 
For instance to select the 56 English datasets that form the "Overall MTEB English leaderboard":

```python
import mteb
benchmark = mteb.get_benchmark("MTEB(eng)")
evaluation = mteb.MTEB(tasks=benchmark)
```

The benchmark specified not only a list of tasks, but also what splits and language to run on. To get an overview of all available benchmarks simply run:

```python
import mteb
benchmarks = mteb.get_benchmarks()
```

Generally we use the naming scheme for benchmarks `MTEB(*)`, where the "*" denotes the target of the benchmark. In the case of a language, we use the three-letter language code. For large groups of languages, we use the group notation, e.g., `MTEB(Scandinavian)` for Scandinavian languages. External benchmarks implemented in MTEB like `CoIR` use their original name. When using a benchmark from MTEB please cite `mteb` along with the citations of the benchmark which you can access using:

```python
benchmark.citation
```

</details>

<details>
  <summary>  Passing in `encode` arguments </summary>


### Passing in `encode` arguments

To pass in arguments to the model's `encode` function, you can use the encode keyword arguments (`encode_kwargs`):

```python
evaluation.run(model, encode_kwargs={"batch_size": 32})
```
</details>


<details>
  <summary>  Selecting evaluation split </summary>

### Selecting evaluation split
You can evaluate only on `test` splits of all tasks by doing the following:

```python
evaluation.run(model, eval_splits=["test"])
```

Note that the public leaderboard uses the test splits for all datasets except MSMARCO, where the "dev" split is used.

</details>

<details>
  <summary>  Using a custom model </summary>


### Using a custom model

Models should implement the following interface, implementing an `encode` function taking as inputs a list of sentences, and returning a list of embeddings (embeddings can be `np.array`, `torch.tensor`, etc.). For inspiration, you can look at the [mteb/mtebscripts repo](https://github.com/embeddings-benchmark/mtebscripts) used for running diverse models via SLURM scripts for the paper.

```python
from mteb.encoder_interface import PromptType

class CustomModel:
    def encode(
        self,
        sentences: list[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.
        
        Args:
            sentences: The sentences to encode.
            task_name: The name of the task.
            prompt_type: The prompt type to use.
            **kwargs: Additional arguments to pass to the encoder.
            
        Returns:
            The encoded sentences.
        """
        pass

model = CustomModel()
tasks = mteb.get_task("Banking77Classification")
evaluation = MTEB(tasks=tasks)
evaluation.run(model)
```

</details>

<details>
  <summary>  Evaluating on a custom dataset </summary>


### Evaluating on a custom dataset

To evaluate on a custom task, you can run the following code on your custom task. See [how to add a new task](docs/adding_a_dataset.md), for how to create a new task in MTEB.

```python
from mteb import MTEB
from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from sentence_transformers import SentenceTransformer


class MyCustomTask(AbsTaskReranking):
    ...

model = SentenceTransformer("average_word_embeddings_komninos")
evaluation = MTEB(tasks=[MyCustomTask()])
evaluation.run(model)
```

</details>

<details>
  <summary>  Using a cross encoder for reranking</summary>


### Using a cross encoder for reranking

To use a cross encoder for reranking, you can directly use a CrossEncoder from SentenceTransformers. The following code shows a two-stage run with the second stage reading results saved from the first stage. 

```python
from mteb import MTEB
import mteb
from sentence_transformers import CrossEncoder, SentenceTransformer

cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
dual_encoder = SentenceTransformer("all-MiniLM-L6-v2")

tasks = mteb.get_tasks(tasks=["NFCorpus"], languages=["eng"])

subset = "default" # subset name used in the NFCorpus dataset
eval_splits = ["test"]

evaluation = MTEB(tasks=tasks)
evaluation.run(
    dual_encoder,
    eval_splits=eval_splits,
    save_predictions=True,
    output_folder="results/stage1",
)
evaluation.run(
    cross_encoder,
    eval_splits=eval_splits,
    top_k=5,
    save_predictions=True,
    output_folder="results/stage2",
    previous_results=f"results/stage1/NFCorpus_{subset}_predictions.json",
)
```

</details>

<details>
  <summary>  Saving retrieval task predictions </summary>

### Saving retrieval task predictions

To save the predictions from a retrieval task, add the `--save_predictions` flag in the CLI or set `save_predictions=True` in the run method. The filename will be in the "{task_name}_{subset}_predictions.json" format.

Python:
```python
from mteb import MTEB
import mteb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

tasks = mteb.get_tasks(tasks=["NFCorpus"], languages=["eng"])

evaluation = MTEB(tasks=tasks)
evaluation.run(
    model,
    eval_splits=["test"],
    save_predictions=True,
    output_folder="results",
)
```

CLI:
```bash
mteb run -t NFCorpus -m all-MiniLM-L6-v2 --output_folder results --save_predictions
```

</details>

<details>
  <summary> Fetching result from the results repository </summary>

### Fetching results from the results repository

Multiple models have already been run on tasks available within MTEB. These results are available results [repository](https://github.com/embeddings-benchmark/results).

To make the results more easily accessible, we have designed custom functionality for retrieving from the repository. For instance, if you are selecting the best model for your French and English retrieval task on legal documents you could fetch the relevant tasks and create a dataframe of the results using the following code:

```python
import mteb
from mteb.task_selection import results_to_dataframe

tasks = mteb.get_tasks(
    task_types=["Retrieval"], languages=["eng", "fra"], domains=["Legal"]
)

model_names = [
    "GritLM/GritLM-7B",
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
]
models = [mteb.get_model_meta(name) for name in model_names]

results = mteb.load_results(models=models, tasks=tasks)

df = results_to_dataframe(results)
```

</details>

<details>
  <summary>  Caching Embeddings To Re-Use Them </summary>


### Caching Embeddings To Re-Use Them

There are times you may want to cache the embeddings so you can re-use them. This may be true if you have multiple query sets for the same corpus (e.g. Wikipedia) or are doing some optimization over the queries (e.g. prompting, other experiments). You can setup a cache by using a simple wrapper, which will save the cache per task in the `cache_embeddings/{task_name}` folder:

```python
# define your task and model above as normal
...
# wrap the model with the cache wrapper
from mteb.models.cache_wrapper import CachedEmbeddingWrapper
model_with_cached_emb = CachedEmbeddingWrapper(model, cache_path='path_to_cache_dir')
# run as normal
evaluation.run(model, ...) 
```

</details>

<br /> 



## Documentation

| Documentation                  |                        |
| ------------------------------ | ---------------------- |
| 📋 [Tasks] | Overview of available tasks |
| 📐 [Benchmarks] | Overview of available benchmarks |
| 📈 [Leaderboard] | The interactive leaderboard of the benchmark |
| 🤖 [Adding a model] | Information related to how to submit a model to the leaderboard |
| 👩‍🔬 [Reproducible workflows] | Information related to how to reproduce and create reproducible workflows with MTEB |
| 👩‍💻 [Adding a dataset] | How to add a new task/dataset to MTEB | 
| 👩‍💻 [Adding a leaderboard tab] | How to add a new leaderboard tab to MTEB | 
| 🤝 [Contributing] | How to contribute to MTEB and set it up for development |
| 🌐 [MMTEB] | An open-source effort to extend MTEB to cover a broad set of languages |  

[Tasks]: docs/tasks.md
[Benchmarks]: docs/benchmarks.md
[Contributing]: CONTRIBUTING.md
[Adding a model]: docs/adding_a_model.md
[Adding a dataset]: docs/adding_a_dataset.md
[Adding a leaderboard tab]: docs/adding_a_leaderboard_tab.md
[Leaderboard]: https://huggingface.co/spaces/mteb/leaderboard
[MMTEB]: docs/mmteb/readme.md
[Reproducible workflows]: docs/reproducible_workflow.md

## Citing

MTEB was introduced in "[MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316)", feel free to cite:

```bibtex
@article{muennighoff2022mteb,
  doi = {10.48550/ARXIV.2210.07316},
  url = {https://arxiv.org/abs/2210.07316},
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Lo{\"\i}c and Reimers, Nils},
  title = {MTEB: Massive Text Embedding Benchmark},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2210.07316},  
  year = {2022}
}
```

You may also want to read and cite the amazing work that has extended MTEB & integrated new datasets:
- Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff. "[C-Pack: Packaged Resources To Advance General Chinese Embedding](https://arxiv.org/abs/2309.07597)" arXiv 2023
- Michael Günther, Jackmin Ong, Isabelle Mohr, Alaeddine Abdessalem, Tanguy Abel, Mohammad Kalim Akram, Susana Guzman, Georgios Mastrapas, Saba Sturua, Bo Wang, Maximilian Werk, Nan Wang, Han Xiao. "[Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents](https://arxiv.org/abs/2310.19923)" arXiv 2023
- Silvan Wehrli, Bert Arnrich, Christopher Irrgang. "[German Text Embedding Clustering Benchmark](https://arxiv.org/abs/2401.02709)" arXiv 2024
- Orion Weller, Benjamin Chang, Sean MacAvaney, Kyle Lo, Arman Cohan, Benjamin Van Durme, Dawn Lawrie, Luca Soldaini. "[FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions](https://arxiv.org/abs/2403.15246)" arXiv 2024
- Dawei Zhu, Liang Wang, Nan Yang, Yifan Song, Wenhao Wu, Furu Wei, Sujian Li. "[LongEmbed: Extending Embedding Models for Long Context Retrieval](https://arxiv.org/abs/2404.12096)" arXiv 2024
- Kenneth Enevoldsen, Márton Kardos, Niklas Muennighoff, Kristoffer Laigaard Nielbo. "[The Scandinavian Embedding Benchmarks: Comprehensive Assessment of Multilingual and Monolingual Text Embedding](https://arxiv.org/abs/2406.02396)" arXiv 2024

For works that have used MTEB for benchmarking, you can find them on the [leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
