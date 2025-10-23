<h1 align="center">
  <img src="docs/images/logos/mteb_logo/dots-icon.png" alt="MTEB" width="28" style="vertical-align: middle; margin-right: 10px;"/> MTEB
</h1>

<h3 align="center" style="border-bottom: none;">Multimodal toolbox for evaluating embeddings and retrieval systems</h3>

<p align="center">
    <a href="https://github.com/embeddings-benchmark/mteb/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/embeddings-benchmark/mteb.svg">
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
        <a href="https://embeddings-benchmark.github.io/mteb/installation/">Installation</a> |
        <a href="https://embeddings-benchmark.github.io/mteb/">Usage</a> |
        <a href="https://huggingface.co/spaces/mteb/leaderboard">Leaderboard</a> |
        <a href="https://embeddings-benchmark.github.io/mteb/">Documentation</a> |
        <a href="#citing">Citing</a>
    </p>
</h4>


<h3 align="center">
    <a href="https://huggingface.co/spaces/mteb/leaderboard"><img style="float: middle; padding: 10px 10px 10px 10px;" width="60" height="55" src="./docs/images/logos/hf_logo.png" /></a>
</h3>


## Installation

You can install mteb simply using pip. For more on installation please see the [documentation](https://embeddings-benchmark.github.io/mteb/installation/).

```bash
pip install mteb
```


## Example Usage

Below we present a simple use-case example. For more information, see the [documentation](https://embeddings-benchmark.github.io/mteb/).

```python
import mteb
from sentence_transformers import SentenceTransformer

# Select model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = mteb.get_model(model_name) # if the model is not implemented in MTEB it will be eq. to SentenceTransformer(model_name)

# Select tasks
tasks = mteb.get_tasks(tasks=["Banking77Classification.v2"])

# evaluate
results = mteb.evaluate(model, tasks=tasks)
```

You can also run it using the CLI:

```bash
mteb run \
    -m sentence-transformers/all-MiniLM-L6-v2 \
    -t "Banking77Classification.v2" \
    --output-folder results
```

For more on how to use the CLI check out the [related documentation](https://embeddings-benchmark.github.io/mteb/usage/cli/).

## Overview

| Overview                       |                                                                                      |
|--------------------------------|--------------------------------------------------------------------------------------|
| 📈 [Leaderboard]               | The interactive leaderboard of the benchmark                                         |
| **Get Started**.               |                                                                                      |
| 🏃 [Get Started]               | Overview of how to use mteb                                                          |
| 🤖 [Defining Models]           | How to use existing model and define custom ones                                     |
| 📋 [Selecting tasks]           | How to select tasks, benchmarks, splits etc.                                         |
| 🏭 [Running Evaluation]        | How to run the evaluations, including cache management, speeding up evaluations etc. |
| 📊 [Loading Results]           | How to load and work with existing model results                                     |
| **Overview**.                  |                                                                                      |
| 📋 [Tasks]                     | Overview of available tasks                                                          |
| 📐 [Benchmarks]                | Overview of available benchmarks                                                     |
| 🤖 [Models]                    | Overview of available Models                                                         |
| **Contributing**               |                                                                                      |
| 🤖 [Adding a model]            | How to submit a model to MTEB and to the leaderboard                                 |
| 👩‍💻 [Adding a dataset]          | How to add a new task/dataset to MTEB                                                |
| 👩‍💻 [Adding a benchmark]        | How to add a new benchmark to MTEB and to the leaderboard                            |
| 🤝 [Contributing]              | How to contribute to MTEB and set it up for development                              |

[Get Started]: https://embeddings-benchmark.github.io/mteb/usage/get_started/
[Defining Models]: https://embeddings-benchmark.github.io/mteb/usage/defining_the_model/
[Selecting tasks]: https://embeddings-benchmark.github.io/mteb/usage/selecting_tasks/
[Running Evaluation]: https://embeddings-benchmark.github.io/mteb/usage/running_the_evaluation/
[Loading Results]: https://embeddings-benchmark.github.io/mteb/usage/loading_results/
[Tasks]: https://embeddings-benchmark.github.io/mteb/overview/available_tasks/any2anymultilingualretrieval/
[Benchmarks]: https://embeddings-benchmark.github.io/mteb/overview/available_benchmarks/
[Models]: https://embeddings-benchmark.github.io/mteb/overview/available_models/text/
[Contributing]: https://embeddings-benchmark.github.io/mteb/CONTRIBUTING/
[Adding a model]: https://embeddings-benchmark.github.io/mteb/contributing/adding_a_model/
[Adding a dataset]: https://embeddings-benchmark.github.io/mteb/contributing/adding_a_dataset/
[Adding a benchmark]: https://embeddings-benchmark.github.io/mteb/contributing/adding_a_benchmark/
[Leaderboard]: https://huggingface.co/spaces/mteb/leaderboard

## Citing

MTEB was introduced in "[MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316)", and heavily expanded in "[MMTEB: Massive Multilingual Text Embedding Benchmark](https://arxiv.org/abs/2502.13595)". When using `mteb`, we recommend that you cite both articles.

<details>
  <summary> Bibtex Citation (click to unfold) </summary>


```bibtex
@article{muennighoff2022mteb,
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Loïc and Reimers, Nils},
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


If you use any of the specific benchmarks, we also recommend that you cite the authors of both the benchmark and its tasks:

```py
benchmark = mteb.get_benchmark("MTEB(eng, v2)")
benchmark.citation # get citation for a specific benchmark

# you can also create a table of the task for the appendix using:
benchmark.tasks.to_latex()
```
