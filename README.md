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


### Using a script

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

### Using the CLI

```bash
mteb available_tasks # list _all_ available tasks

mteb run -m sentence-transformers/all-MiniLM-L6-v2 \
    -t Banking77Classification  \
    --verbosity 3

# if nothing is specified default to saving the results in the results/{model_name} folder
```

Note that using multiple GPUs in parallel can be done by just having a custom encode function that distributes the inputs to multiple GPUs like e.g. [here](https://github.com/microsoft/unilm/blob/b60c741f746877293bb85eed6806736fc8fa0ffd/e5/mteb_eval.py#L60) or [here](https://github.com/ContextualAI/gritlm/blob/09d8630f0c95ac6a456354bcb6f964d7b9b6a609/gritlm/gritlm.py#L75). See [custom models](docs/usage/usage.md#using-a-custom-model) for more information.


## Usage Documentation
The following links to the main sections in the usage documentation.

| Section | |
| ------- |- |
| **General** | |
| [Evaluating a Model](docs/usage/usage.md#evaluating-a-model) | How to evaluate a model |
| [Evaluating on different Modalities](docs/usage/usage.md#evaluating-on-different-modalities) | How to evaluate image and image-text tasks |
| **Selecting Tasks** | |
| [Selecting a benchmark](docs/usage/usage.md#selecting-a-benchmark) | How to select and filter tasks |
| [Task selection](docs/usage/usage.md#task-selection) | How to select and filter tasks |
| [Selecting Split and Subsets](docs/usage/usage.md#selecting-evaluation-split-or-subsets) | How to select evaluation splits or subsets |
| [Using a Custom Task](docs/usage/usage.md#using-a-custom-task) | How to evaluate on a custom task |
| **Selecting a Model** | |
| [Using a Pre-defined Model](docs/usage/usage.md#using-a-pre-defined-model) | How to run a pre-defined model |
| [Using a SentenceTransformer Model](docs/usage/usage.md#using-a-sentence-transformer-model) | How to run a model loaded using sentence-transformers |
| [Using a Custom Model](docs/usage/usage.md#using-a-custom-model) | How to run and implement a custom model |
| **Running Evaluation** | |
| [Passing Arguments to the model](docs/usage/usage.md#passing-in-encode-arguments) | How to pass `encode` arguments to the model |
| [Running Cross Encoders](docs/usage/usage.md#running-cross-encoders-on-reranking) | How to run cross encoders for reranking |
| [Running Late Interaction (ColBERT)](docs/usage/usage.md#using-late-interaction-models) | How to run late interaction models |
| [Saving Retrieval Predictions](docs/usage/usage.md#saving-retrieval-task-predictions) | How to save prediction for later analysis |
| [Caching Embeddings](docs/usage/usage.md#caching-embeddings-to-re-use-them) | How to cache and re-use embeddings |
| **Leaderboard** | |
| [Running the Leaderboard Locally](docs/usage/usage.md#running-the-leaderboard-locally) | How to run the leaderboard locally |
| [Report Data Contamination](docs/usage/usage.md#annotate-contamination) | How to report data contamination for a model |
| [Fetching Result from the Leaderboard](docs/usage/usage.md#fetching-results-from-the-leaderboard) | How to fetch the raw results from the leaderboard |


## Overview

| Overview                       |                                                                                     |
|--------------------------------|-------------------------------------------------------------------------------------|
| 📈 [Leaderboard]               | The interactive leaderboard of the benchmark                                        |
| 📋 [Tasks]                     | Overview of available tasks                                                         |
| 📐 [Benchmarks]                | Overview of available benchmarks                                                    |
| **Contributing**               |                                                                                     |
| 🤖 [Adding a model]            | Information related to how to submit a model to MTEB and to the leaderboard         |
| 👩‍🔬 [Reproducible workflows]    | Information related to how to create reproducible workflows with MTEB               |
| 👩‍💻 [Adding a dataset]          | How to add a new task/dataset to MTEB                                               |
| 👩‍💻 [Adding a benchmark]        | How to add a new benchmark to MTEB and to the leaderboard                           |
| 🤝 [Contributing]              | How to contribute to MTEB and set it up for development                             |

[Tasks]: docs/tasks.md
[Benchmarks]: docs/benchmarks.md
[Contributing]: CONTRIBUTING.md
[Adding a model]: docs/adding_a_model.md
[Adding a dataset]: docs/adding_a_dataset.md
[Adding a benchmark]: docs/adding_a_benchmark.md
[Leaderboard]: https://huggingface.co/spaces/mteb/leaderboard
[Reproducible workflows]: docs/reproducible_workflow.md

## Citing

MTEB was introduced in "[MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316)", and heavily expanded in "[MMTEB: Massive Multilingual Text Embedding Benchmark](https://arxiv.org/abs/2502.13595)". When using `mteb`, we recommend that you cite both articles.

<details>
  <summary> Bibtex Citation (click to unfold) </summary>


```bibtex
@article{muennighoff2022mteb,
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Lo{\"\i}c and Reimers, Nils},
  title = {MTEB: Massive Text Embedding Benchmark},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2210.07316},
  year = {2022}
  url = {https://arxiv.org/abs/2210.07316},
  doi = {10.48550/ARXIV.2210.07316},
}

@article{enevoldsen2025mmtebmassivemultilingualtext,
  title={MMTEB: Massive Multilingual Text Embedding Benchmark},
  author={Kenneth Enevoldsen and Isaac Chung and Imene Kerboua and Márton Kardos and Ashwin Mathur and David Stap and Jay Gala and Wissam Siblini and Dominik Krzemiński and Genta Indra Winata and Saba Sturua and Saiteja Utpala and Mathieu Ciancone and Marion Schaeffer and Gabriel Sequeira and Diganta Misra and Shreeya Dhakal and Jonathan Rystrøm and Roman Solomatin and Ömer Çağatan and Akash Kundu and Martin Bernstorff and Shitao Xiao and Akshita Sukhlecha and Bhavish Pahwa and Rafał Poświata and Kranthi Kiran GV and Shawon Ashraf and Daniel Auras and Björn Plüster and Jan Philipp Harries and Loïc Magne and Isabelle Mohr and Mariya Hendriksen and Dawei Zhu and Hippolyte Gisserot-Boukhlef and Tom Aarsen and Jan Kostkan and Konrad Wojtasik and Taemin Lee and Marek Šuppa and Crystina Zhang and Roberta Rocca and Mohammed Hamdy and Andrianos Michail and John Yang and Manuel Faysse and Aleksei Vatolin and Nandan Thakur and Manan Dey and Dipam Vasani and Pranjal Chitale and Simone Tedeschi and Nguyen Tai and Artem Snegirev and Michael Günther and Mengzhou Xia and Weijia Shi and Xing Han Lù and Jordan Clive and Gayatri Krishnakumar and Anna Maksimova and Silvan Wehrli and Maria Tikhonova and Henil Panchal and Aleksandr Abramov and Malte Ostendorff and Zheng Liu and Simon Clematide and Lester James Miranda and Alena Fenogenova and Guangyu Song and Ruqiya Bin Safi and Wen-Ding Li and Alessia Borghini and Federico Cassano and Hongjin Su and Jimmy Lin and Howard Yen and Lasse Hansen and Sara Hooker and Chenghao Xiao and Vaibhav Adlakha and Orion Weller and Siva Reddy and Niklas Muennighoff},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2502.13595},
  year={2025},
  url={https://arxiv.org/abs/2502.13595},
  doi = {10.48550/arXiv.2502.13595},
}
```
</details>


If you use any of the specific benchmarks, we also recommend that you cite the authors.

```py
benchmark = mteb.get_benchmark("MTEB(eng, v2)")
benchmark.citation # get citation for a specific benchmark

# you can also create a table of the task for the appendix using:
benchmark.tasks.to_latex()
```

Some of these amazing publications include (ordered chronologically):
- Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff. "[C-Pack: Packaged Resources To Advance General Chinese Embedding](https://arxiv.org/abs/2309.07597)" arXiv 2023
- Michael Günther, Jackmin Ong, Isabelle Mohr, Alaeddine Abdessalem, Tanguy Abel, Mohammad Kalim Akram, Susana Guzman, Georgios Mastrapas, Saba Sturua, Bo Wang, Maximilian Werk, Nan Wang, Han Xiao. "[Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents](https://arxiv.org/abs/2310.19923)" arXiv 2023
- Silvan Wehrli, Bert Arnrich, Christopher Irrgang. "[German Text Embedding Clustering Benchmark](https://arxiv.org/abs/2401.02709)" arXiv 2024
- Orion Weller, Benjamin Chang, Sean MacAvaney, Kyle Lo, Arman Cohan, Benjamin Van Durme, Dawn Lawrie, Luca Soldaini. "[FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions](https://arxiv.org/abs/2403.15246)" arXiv 2024
- Dawei Zhu, Liang Wang, Nan Yang, Yifan Song, Wenhao Wu, Furu Wei, Sujian Li. "[LongEmbed: Extending Embedding Models for Long Context Retrieval](https://arxiv.org/abs/2404.12096)" arXiv 2024
- Kenneth Enevoldsen, Márton Kardos, Niklas Muennighoff, Kristoffer Laigaard Nielbo. "[The Scandinavian Embedding Benchmarks: Comprehensive Assessment of Multilingual and Monolingual Text Embedding](https://arxiv.org/abs/2406.02396)" arXiv 2024
- Ali Shiraee Kasmaee, Mohammad Khodadad, Mohammad Arshi Saloot, Nick Sherck, Stephen Dokas, Hamidreza Mahyar, Soheila Samiee. "[ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance & Efficiency on a Specific Domain](https://arxiv.org/abs/2412.00532)" arXiv 2024
