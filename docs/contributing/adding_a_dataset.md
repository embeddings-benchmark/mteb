# Adding a Dataset

To add a new dataset to MTEB, you need to do three things:

1. **Create a new subclass**: Implement a task with the desired dataset, by subclassing an abstract task
2. **Add metedata** describing the task, main scores, languages etc.
3. **Calculate descriptive statistics**, which is used to detect duplicates, few number of samples or very short documents
4. **Submit A PR** of the the edits to the [MTEB](https://github.com/embeddings-benchmark/mteb/blob/main) repository

We go through these steps below, but if you have any questions regarding this process feel free to open a discussion [thread](https://github.com/embeddings-benchmark/mteb/discussions). It is also reasonable to look at [tasks already implemented](https://github.com/embeddings-benchmark/mteb/tree/main/mteb/tasks) to get an idea about the structure.


## Creating a new subclass

### Overview of the task

Implementing a task in `mteb`, typically has the following structure:

```python
from datasets import load_dataset

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.classification import AbsTaskClassification

class MyNewTask(AbsTaskClassification):
    # metada contains information such as title, description, metrics etc.
    metadata = TaskMetadata(...)

    # task specific setting e.g. specifying the column names
    label_column_name = "label"
    input_column_name = "text"

    # This is the function that loads the dataset, this is typically untouched
    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        self.dataset = load_dataset(
            **self.metadata.dataset,
        )
        self.dataset_transform() # optional processing
        self.data_loaded = True

    # dataset transform, which allow you to process the dataset
    # including downsampling, filtering etc.
    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        # some processing
        ...
```

### Select an appropriate task

To add a dataset you first need to figure out which type of task is the best suited for the dataset. Below we will give you an overview of the most common,
but do see [abstasks](../api/task.md#multimodal-tasks) for an overview of all the tasks available.

| Task                     | Abstask                                                                       | Description                                                                                                                                                                                                                                                                                                                                                                                                                                  | Default Metric                                                                                      |
|--------------------------|-------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Classification           | [`AbsTaskClassification`][mteb.abstasks.classification.AbsTaskClassification] | Fits a classifier on the embeddings derived from the model. The goal is the predict the labels correctly. This does not change the weights of the model itself. See also classes for [multi label classificaiton][mteb.abstasks.multilabel_classification.AbsTaskMultilabelClassification], [regression][mteb.abstasks.regression.AbsTaskRegression] and [pair classification][mteb.abstasks.pair_classification.AbsTaskPairClassification]. | [Accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision)                                    |
| Clustering               | [`AbsTaskClustering`][mteb.abstasks.clustering.AbsTaskClustering]             | Cluster documents based on their embeddings. The goal is to cluster documents according to predefined categories. Support clustering in multiple hiarchies.                                                                                                                                                                                                                                                                                  | [V-Measure](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html) |
| Retrieval                | [`AbsTaskRetrieval`][mteb.abstasks.retrieval.AbsTaskRetrieval]                | Retrieval tasks include a corpus from which you retreive from using a query. The goal is to retrieve relevant documents. See also [`convert_to_reranking`][mteb.abstasks.retrieval.AbsTaskRetrieval.convert_to_reranking] for how to convert a retrieval task into a reranking task.                                                                                                                                                         | [NDCG_at_10](https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG)               |
| Semantic Text Similarity | [`AbsTaskSTS`][mteb.abstasks.sts.AbsTaskSTS]                                  | Compares the (semantic) similarity of pairs of documents. The goal is to embed documents such that semanticly similar statements appear close.                                                                                                                                                                                                                                                                                               | [Spearman](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)                 |

!!! Note
    While each task has a default score we compute multiple and it is possible to select any of these are the default metric for your task.


### Implementing the task

Once we have decided on task, we can implement them as follows:

=== "Classification"

    For our classification task we use the [poem sentiment](https://huggingface.co/datasets/mteb/poem_sentiment) dataset, which consist of verses
    with four labels 0 (negative), 1 (positive), 2 (no_impact) and 3 (mixed). Let us say that we just want to look at the label 1 and 2.

    We can then implement the task as follows:

    ```py
    import mteb
    from mteb.abstasks import AbsTaskClassification


    class MyClassificationtask(AbsTaskClassification):
        metadata = mteb.TaskMetadata(  # minimal metadata
            name="MyClassificationTask",
            description="A dummy classification task based on poems sentiment classification.",
            main_score="accuracy",
            eval_langs=["eng-Latn"],
            eval_splits=["test"],
            type="Classification",
            dataset={
                "path": "mteb/poem_sentiment",
                "revision": "9fdc57b89ccc09a8d9256f376112d626878e51a7",
            },
            prompt="Classify poem verses as positive or negtive"
        )

        label_column_name = "label"
        input_column_name = "text"

        def dataset_transform(self, num_proc=None, **kwargs) -> None:
            # filter to only include label 1 positive and label 0 negative
            self.dataset = self.dataset.filter(
                lambda x: x[self.label_column_name] in [0, 1], num_proc=num_proc
            )
    ```

    Once we have the task we can then test to make sure that everything works as intended:

    ```py
    # ensure that the dataset can be loaded and transformed properly
    task = MyClassificationtask()
    task.load_data()

    print(task.dataset["test"][0]) # check one of the samples:
    # {'id': 1, 'text': 'shall yet be glad for him, and he shall bless', 'label': 1}

    # ensure that we can evaluate the model on the task
    mdl = mteb.get_model("baseline/random-encoder-baseline")
    results = mteb.evaluate(mdl, task)
    print(results[0].get_score())  # print the accuracy score of the random baseline
    # 0.49428571428571433
    ```

=== "Clustering"

    MISSING

=== "Retrieval"

    MISSING

=== "Semantic Similarity"

    MISSING


??? example "Overwriting `load_data`"

    While we do not recommend overwriting `load_data` it can often be useful, when developing tasks and can also be used in conjuction with
    [`push_dataset_to_hub`][mteb.AbsTask.push_dataset_to_hub] to push datasets to the hub in the correct format.

    ```python
    import mteb
    from mteb.abstasks import AbsTaskClassification


    class MyClassificationtask(AbsTaskClassification):
        metadata = mteb.TaskMetadata(  # minimal metadata
            name="MyClassificationTask",
            description="A dummy classification just for testing",
            main_score="accuracy",
            eval_langs=["eng-Latn"],
            eval_splits=["test"],
            type="Classification",
            dataset={"path": "na", "revision": "na"},
        )

        label_column_name = "label"
        input_column_name = "text"

        def load_data(self, num_proc=None, **kwargs) -> None:
            self.dataset = {
                "test": [
                    {"text": "sample text 1", "label": 1},
                    {"text": "sample text 2", "label": 0},
                ]
            }
            self.data_loaded = True
    ```

    which can then be run as follows:

    ```py
    task = MyClassificationtask()
    mdl = mteb.get_model("baseline/random-encoder-baseline")
    results = mteb.evaluate(mdl, task)
    ```


### Filling out the TaskMetadata

MISSING
<!--
Along with the task MTEB requires metadata regarding the task. If the metadata isn't available please provide your best guess or leave the field as `None`.

To get an overview of the fields in the metadata object, you can look at the [TaskMetadata][mteb.TaskMetadata] class.


!!! Note
    That these fields can be left blank if the information is not available and can be extended if necessary. We do not include any machine-translated (without verification) datasets in the benchmark.
-->


### Pushing the dataset to the hub

MISSING


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
