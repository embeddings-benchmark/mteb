# MTEB Documentation

!!! info
    We recently released `mteb` version 2.0.0, to see what is new check of [what is new](./whats_new.md#new-in-v20) and see [how to upgrade](./whats_new.md#upgrading-from-v1) your existing code.

Welcome documentation of MTEB. `mteb` a package for benchmark and evaluating the quality of embeddings.

MTEB is the go-to documentation for evaluating embeddings models across a variety of tasks, modalities and domains. MTEB covers more than a 1000 different tasks from covering a diverse set of tasks from historic Swedish patent classification to documentation retrieval for Python. These tasks spread across more than 1000 languages and cover both image and text tasks.

This package was initially introduced as a package for evaluating text embeddings predominantly for English[@mteb_2023], but have since been extended for broad languages coverage[@mmteb_2025] and to support multiple modalities[@mieb_2025].


## Installation

Installation is as simple as:

=== "pip"
    ```bash
    pip install mteb
    ```

=== "uv"
    ```bash
    uv add mteb
    ```

To see more check out the [installation guide](./installation.md).

## Quickstart


=== "Using Script"

    To evaluating a model simply select a model, select tasks and evaluate:

    ```python
    import mteb
    from sentence_transformers import SentenceTransformer

    # Select model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = mteb.get_model(model_name) # if the model is not implemented in MTEB it will be eq. to SentenceTransformer(model_name)

    # Select tasks
    tasks = mteb.get_tasks(tasks=["Banking77Classification"])

    # evaluate
    results = mteb.evaluate(model, tasks=tasks)
    ```

    To see more check out the [usage documentation](./usage/get_started.md)

=== "Using the CLI"

    To run a model from the cli simply specify the `--model/-m` and the `--tasks/-t`
    ```bash
    mteb run \
        -m sentence-transformers/all-MiniLM-L6-v2 \
        -t Banking77Classification \
        --output-folder results
    ```

    To read more about what you can do with the command line interface check out its [documentation](./usage/cli.md)

---

## Citing


MTEB was introduced in the paper "MTEB: Massive Text Embedding Benchmark"[@mteb_2023], and heavily expanded in "MMTEB: Massive Multilingual Text Embedding Benchmark"[@mmteb_2025]. When using `mteb`, we recommend that you cite both articles.


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

If you use any of the specific benchmarks, we also recommend that you cite the paper, which you can obtain using:

```python
benchmark = mteb.get_benchmark("MTEB(eng, v2)")
benchmark.citation # get citation for a specific benchmark

# you can also create a table of the task for the appendix using:
benchmark.tasks.to_latex()
```
