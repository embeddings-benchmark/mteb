# Adding a Dataset

To add a new dataset to MTEB, you need to do three things:

1) Implement a task with the desired dataset, by subclassing an abstract task
2) Add metadata to the task
3) Calculate statistics of the task (run [`task.calculate_descriptive_statistics()`][mteb.AbsTask.calculate_descriptive_statistics])
4) Submit the edits to the [MTEB](https://github.com/embeddings-benchmark/mteb/blob/main) repository

If you have any questions regarding this process feel free to open a discussion [thread](https://github.com/embeddings-benchmark/mteb/discussions).

!!! Note
    When we mention adding a dataset we refer to a subclass of one of the [abstasks](../api/task.md#multimodal-tasks).

## Creating a new subclass

### A Simple Example

To add a new task, you need to implement a new class that inherits from the [`AbsTask`][mteb.AbsTask] associated with the task type (e.g. [`AbsTaskRetrieval`][mteb.abstasks.retrieval.AbsTaskRetrieval] for retrieval tasks). You can find the supported task types in [here](../api/task.md#multimodal-tasks).

??? example "SciDocs Reranking Task"
    ```python
    from mteb.abstasks.retrieval import AbsTaskRetrieval
    from mteb.abstasks.task_metadata import TaskMetadata

    class SciDocsReranking(AbsTaskRetrieval):
        metadata = TaskMetadata(
            name="SciDocsRR",
            description="Ranking of related scientific papers based on their title.",
            reference="https://allenai.org/data/scidocs",
            type="Reranking",
            category="t2t",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map",
            dataset={
                "path": "mteb/scidocs-reranking",
                "revision": "d3c5e1fc0b855ab6097bf1cda04dd73947d7caab",
            },
            date=("2000-01-01", "2020-12-31"), # best guess
            domains=["Academic", "Non-fiction", "Domains"],
            task_subtypes=["Scientific Reranking"],
            license="cc-by-4.0",
            annotations_creators="derived",
            dialect=[],
            sample_creation="found",
            bibtex_citation="""
    @inproceedings{cohan-etal-2020-specter,
        title = "{SPECTER}: Document-level Representation Learning using Citation-informed Transformers",
        author = "Cohan, Arman  and
          Feldman, Sergey  and
          Beltagy, Iz  and
          Downey, Doug  and
          Weld, Daniel",
        editor = "Jurafsky, Dan  and
          Chai, Joyce  and
          Schluter, Natalie  and
          Tetreault, Joel",
        booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
        month = jul,
        year = "2020",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2020.acl-main.207",
        doi = "10.18653/v1/2020.acl-main.207",
        pages = "2270--2282",
    }
    """,
    )

    # testing the task with a model:
    model = mteb.get_model("intfloat/multilingual-e5-small")
    results = mteb.evaluate(model, tasks=[SciDocsReranking()])
    ```

!!! Note
    For multilingual/crosslingual tasks, make sure you've specified [`eval_langs`][mteb.TaskMetadata] as a dictionary, as shown in [this example](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/classification/multilingual/mtop_intent_classification.py).



### A Detailed Example
Often the dataset from HuggingFace is not in the format expected by MTEB. To resolve this you can either change the format on Hugging Face or add a [`dataset_transform`][mteb.AbsTask.dataset_transform] method to your dataset to transform it into the right format on the fly. Here is an example along with some design considerations:

??? example "DBpediaClassificationV2 Task"

    ```python
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.abstasks.classification import AbsTaskClassification

    class DBpediaClassificationV2(AbsTaskClassification):
        metadata = TaskMetadata(
            ... # fill in metadata as shown in the simple example above
        )

        def load_dataset(self):
            self.dataset = load_dataset(
                **self.metadata.dataset,
            )
            ... # some processing
            self.data_loaded = True

        # dataset transform will be called if `load_dataset` is not overridden
        def dataset_transform(self):
            self.dataset = self.stratified_subsampling(
                self.dataset, seed=self.seed, splits=["train", "test"]
            )
    ```

## Creating the metadata object
Along with the task MTEB requires metadata regarding the task. If the metadata isn't available please provide your best guess or leave the field as `None`.

To get an overview of the fields in the metadata object, you can look at the [TaskMetadata][mteb.TaskMetadata] class.


!!! Note
    That these fields can be left blank if the information is not available and can be extended if necessary. We do not include any machine-translated (without verification) datasets in the benchmark.

## Submit a PR

Once you are finished create a PR to the [MTEB](https://github.com/embeddings-benchmark/mteb) repository. If you haven't created a PR before please refer to the [GitHub documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/)

The PR will be reviewed by one of the organizers or contributors who might ask you to change things. Once the PR is approved the dataset will be added into the main repository.


Before you commit, here is a checklist you should complete before submitting:

```markdown
- [ ] I have outlined why this dataset is filling an existing gap in `mteb`
- [ ] I have tested that the dataset runs with the `mteb` package.
- [ ] I have run the following models on the task (adding the results to the pr). These can be run using the `mteb run -m {model_name} -t {task_name}` command.
  - [ ] `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - [ ] `intfloat/multilingual-e5-small`
- [ ] I have checked that the performance is neither trivial (both models gain close to perfect scores) nor random (both models gain close to random scores).
- [ ] I have considered the size of the dataset and reduced it if it is too big (2048 examples is typically large enough for most tasks)
```

An easy way to test it is using:

=== "Python"
    ```python
    import mteb
    # sample model:
    model = mteb.get_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    task = mteb.get_task("{name of your task}")

    results = mteb.evaluate(model, task)
    ```
=== "CLI"
    ```bash
    mteb run -m sentence-transformers/paraphrase-multilingual-MiniLM -t {name of your task}
    ```
