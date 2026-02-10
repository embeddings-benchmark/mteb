# Adding a Dataset

To add a new dataset to MTEB, you need to do three things:

- 1) **Implement a new task**: Implement a task with the desired dataset, by subclassing an abstract task
- 2) **Fill out the metadata** describing the task, main scores, languages etc.
- 3) **Calculate descriptive statistics**, which is used to detect duplicates, few number of samples or very short documents
- 4) **Submit a Pull Request** of the the edits to the [MTEB](https://github.com/embeddings-benchmark/mteb/blob/main) repository

We go through these steps below, but if you have any questions regarding this process feel free to open a discussion [thread](https://github.com/embeddings-benchmark/mteb/discussions). It is also reasonable to look at [tasks already implemented](https://github.com/embeddings-benchmark/mteb/tree/main/mteb/tasks) to get an idea about the structure.


## Implement a new task

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

| Task | Abstask | Description | Common Metric |
| ---- | ------- | ----------- | -------------- |
| Classification | [`AbsTaskClassification`][mteb.abstasks.classification.AbsTaskClassification]    | Fits a classifier on the embeddings derived from the model. The goal is the predict the labels correctly. This does not change the weights of the model itself. See also classes for [multi label classificaiton][mteb.abstasks.multilabel_classification.AbsTaskMultilabelClassification], [regression][mteb.abstasks.regression.AbsTaskRegression] and [pair classification][mteb.abstasks.pair_classification.AbsTaskPairClassification]. | [Accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision) |
| Clustering | [`AbsTaskClustering`][mteb.abstasks.clustering.AbsTaskClustering]    | Cluster documents based on their embeddings. The goal is to cluster documents according to predefined categories. Support clustering in multiple hiarchies. | [V-Measure](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html) |
| Retrieval | [`AbsTaskRetrieval`][mteb.abstasks.retrieval.AbsTaskRetrieval]    | Retrieval tasks include a corpus from which you retreive from using a query. The goal is to retrieve relevant documents. See also [`convert_to_reranking`](../api/task/?h=convert_to_reranking#mteb.abstasks.retrieval.AbsTaskRetrieval.convert_to_reranking) for how to convert a retrieval task into a reranking task. | [NDCG@10](https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG) |
| Semantic Text Similarity | [`AbsTaskSTS`][mteb.abstasks.sts.AbsTaskSTS]    | Compares the (semantic) similarity of pairs of documents. The goal is to embed documents such that semanticly similar statements appear close. | [Spearman](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) |

!!! Note
    While each task has a main score we compute multiple and it is possible to select any of these are the main metric for your task.


### Implementing the task

Once we have decided on task, we can implement them as follows:

=== "Classification"

    For our classification task we use the [poem sentiment](https://huggingface.co/datasets/mteb/poem_sentiment) dataset, which consist of verses
    with four labels 0 (negative), 1 (positive), 2 (no_impact) and 3 (mixed). Let us say that we just want to look at the label 1 and 2.

    We can then implement the task as follows:

    ```python
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

    ```python
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

    For our classification task we use the [swedn](https://huggingface.co/datasets/mteb/SwednClusteringP2P) dataset of Swedish newspapers
    along four main categories. It contains three splits "headlines", "articles" and "summaries", here we will look at the headlines only.

    The clustering task performs multiple experiments to get a more consistent estimate and reports the avarage. Here we set it 3 experiments
    using 1000 samples, but typically you would set it higher. Using the default is also a reasonable idea.
    
    We can then implement the task as follows:

    ```python
    class MyClusteringTask(AbsTaskClustering):
    metadata = mteb.TaskMetadata(  # minimal metadata
        name="MyClusteringTask",
        description="A dummy clustering task on swedish news articles",
        main_score="v_measure",
        eval_langs=["swe-Latn"],
        eval_splits=["headlines"],  # just use the headlines split for evaluation
        type="Clustering",
        dataset={
            "path": "mteb/SwednClusteringP2P",
            "revision": "f8dbf10ec231cc25e9f63454d5cd2d90af95e5f8",
        },
    )

    label_column_name = "labels"
    input_column_name = "sentences"
    n_clusters = 10  # number of clustering experiments to run
    max_documents_per_cluster = (
        1000  # numberof documents to sample per each clustering experiment
    )
    ```

    Once we have the task we can then test to make sure that everything works as intended:

    ```python
    # ensure that the dataset can be loaded and transformed properly
    task = MyClusteringTask()
    task.load_data()

    print(task.dataset["headlines"][0])  # check one of the samples:
    # {'sentences': 'Ryssland kritiserar utökade Irak-sanktioner', 'labels': 'domestic news'}

    # ensure that we can evaluate the model on the task
    mdl = mteb.get_model("baseline/random-encoder-baseline")
    results = mteb.evaluate(mdl, task)
    print(results[0].get_score())  # print the v_measure score of the random baseline
    # 0.006444026234557669
    ```


=== "Retrieval"

    For our retrieval dataset we use the a dataset consisting of [android-related questions and answer](https://huggingface.co/datasets/mteb/CQADupstackAndroidRetrieval). The dataset consist of a corpus of questions and a set of queries, where each query has a set of relevant documents in the corpus. The goal is to retrieve the relevant documents for each query.

    ```python
    import mteb
    from mteb.abstasks import AbsTaskRetrieval


    class MyRetrievalTask(AbsTaskRetrieval):
        metadata = mteb.TaskMetadata(  # minimal metadata
            name="MyRetrievalTask",
            description="A dummy retrieval task on a community question-answering dataset on android-related questions.",
            main_score="ndcg_at_10",
            eval_langs=["eng-Latn"],
            eval_splits=["test"],
            type="Retrieval",
            dataset={
                "path": "mteb/CQADupstackAndroidRetrieval",
                "revision": "9be4c0e46342e8e3aff577a89b9a1ec9bc6b4af3",
            },
            prompt="Given a question, retrieve the most relevant question from the corpus.",
        )
    ```

    Once we have the task we can then test to make sure that everything works as intended:

    ```python
    # ensure that the dataset can be loaded without errors
    task = MyRetrievalTask()
    task.load_data()

    test_set = task.dataset["default"]["test"]  # default unless there are multiple subsets
    print(test_set["corpus"][0])  # print the first item in the corpus
    # {'id': '51829', 'title': 'How can show android tablet as a external storage to PC?', 'text': "I want to send files to android tablet ..."}
    print(test_set["queries"][0])  # print the first query
    # {'id': '11546', 'text': 'Android chroot ubuntu - is it possible to get ubuntu to recognise usb devices'}
    test_set["relevant_docs"]["11546"]  # print the relevant documents for the first query
    # {'18572': 1}

    # ensure that we can evaluate the model on the task
    mdl = mteb.get_model("baseline/random-encoder-baseline")
    results = mteb.evaluate(mdl, task)
    print(results[0].get_score())  # print the spearman score of the random baseline
    # 0.021194685839832323
    ```


=== "Multilingual Semantic Similarity"

    We kick up the notch here an do a multilingual example using a [Indic Cross-lingual semantic similarity corpus](https://huggingface.co/datasets/mteb/IndicCrosslingualSTS). We will use just three of the languages as an example, but we can easily add more. The subsets on huggingface we will
    be using is "en-as", "en-bn", and "en-gu". The dataset consist of pairs of sentences between English and an Indic language which are rated based
    on their similarity on a score from 0-5.

    We will implement this as follows:

    ```python
    import mteb
    from mteb.abstasks import AbsTaskSTS


    class MySTSTask(AbsTaskSTS):
        metadata = mteb.TaskMetadata(  # minimal metadata
            name="MySTSTask",
            description="A dummy STS task on a cross-lingual dataset between English and 3 indic languages",
            main_score="v_measure",
            # for multilingual tasks, we need to specify the eval_langs as a mapping between the huggingface subset (e.g. "en-as")
            # and their respective language codes in the dataset.
            eval_langs={
                "en-as": ["eng-Latn", "asm-Beng"],
                "en-bn": ["eng-Latn", "ben-Beng"],
                "en-gu": ["eng-Latn", "guj-Gujr"],
            },
            eval_splits=["test"], 
            type="STS",
            dataset={
                "path": "mteb/IndicCrosslingualSTS",
                "revision": "217bb120770a619b091d77aa06421a770821b22b",
            },
        )

        column_names = ("sentence1", "sentence2")
        min_score = 0
        max_score = 5
    ```
    
    Once we have the task we can then test to make sure that everything works as intended:

    ```python
    # ensure that the dataset can be loaded without errors
    task = MySTSTask()
    task.load_data()

    print(task.dataset["en-gu"]["test"][0])  # check one of the samples:
    # {'sentence1': 'Akrund is a small village in Dhansura Taluka of Aravalli district of northern Gujarat in western India.', 'sentence2': 'રાહતલાવ ભારત દેશના પશ્ચિમ ભાગમાં આવેલા ગુજરાત રાજ્યના ચરોતર પ્રદેશમાં આવેલા આણંદ જિલ્લામાં આવેલા આણંદ તાલુકામાં આવેલું એક ગામ છે.', 'score': 1.0}

    # ensure that we can evaluate the model on the task
    mdl = mteb.get_model("baseline/random-encoder-baseline")
    results = mteb.evaluate(mdl, task)
    print(results[0].get_score())  # print the spearman score of the random baseline
    # 0.021194685839832323
    ```



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

To run a task locally you do not necesarily need to fill out all the fields in the `TaskMetadata`, but if you want to include the dataset in
`mteb` we require that all fields are filled out. This is to ensure that we have enough information about the dataset to be able to include it in the benchmark and to make it easier for users to understand the dataset and its characteristics.

If you are making a PR, feel free to leave fields and `None` if you are unsure about how to fill. You can always ask about it during the PR.

Here is an example of how to fill out the `TaskMetadata` for the poem sentiment classification dataset, we implemented above and which is also
available within `mteb` as `PoemSentimentClassification.v2`, for more details about the fields see the [TaskMetadata documentation](../api/task.md#metadata).

```python
TaskMetadata(
        name="PoemSentimentClassification.v2",
        # description should 
        description="Poem Sentiment consist of poem verses from Project Gutenberg annotated for sentiment using the labels negative (0), positive (1), no_impact (2) and mixed (3). This version was corrected as a part of [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900) to fix common issues on the original datasets, including removing duplicates, and train-test leakage.",
        reference="https://arxiv.org/abs/2011.02686",
        dataset={
            "path": "mteb/poem_sentiment",
            "revision": "9fdc57b89ccc09a8d9256f376112d626878e51a7",
        },
        type="Classification",
        category="t2c", # text-2-class
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1700-01-01", "1900-01-01"), # a very rough guess of the date range of the poems, we do not have exact dates for all poems
        domains=["Written", "Fiction", "Poetry"],
        task_subtypes=["Sentiment/Hate speech"], # if no subtypes match then just use []
        license="cc-by-4.0",
        annotations_creators="human-annotated", 
        dialect=["eng-Latn-US", "en-Latn-GB"], # dialects is often unknown if so just use []
        sample_creation="found", # the text was not created for the purpose of the dataset, but rather found and annotated
        adapted_from=["PoemSentimentClassification"], # Previous version of the dataset, can be None
        bibtex_citation=r"""
@misc{sheng2020investigating,
  archiveprefix = {arXiv},
  author = {Emily Sheng and David Uthus},
  eprint = {2011.02686},
  primaryclass = {cs.CL},
  title = {Investigating Societal Biases in a Poetry Composition System},
  year = {2020},
}
""",
    )
```


### Pushing the dataset to the hub

Once a dataset is added to the repository, it can be pushed to the hub using the [`push_dataset_to_hub`](../api/task/?h=push_data#mteb.AbsTask.push_dataset_to_hub) method. This will reupload the dataset to the hub in the correct format and with all the processing applied. This means that
if you have done some processing in the `dataset_transform` method, this will be reflected in the dataset pushed to the hub.
You can then update class to utilize the new dataset on the hub and remove the `dataset_transform` method. This avoid the need to do the processing every time the dataset is loaded, which can be time consuming for large datasets.

```python
import mteb

class MyTask(...):
    ...

task = MyTask()
repo_name = f"myorg/{task.metadata.name}"
# Push the dataset to the Hub
task.push_dataset_to_hub(repo_name)
```


## Create a Pull Request

Once you have your task you can create a pull request (PR) to the main repository. To do so place task inside the [`mteb/tasks`](https://github.com/embeddings-benchmark/mteb/tree/main/mteb/tasks) directory, and make sure to import the task in the `__init__.py` file in the same directory.


## Calculate descriptive statistics

Before creating a pull request, it is important to calculate some descriptive statistics about the dataset. This is to ensure that the dataset is not too small, does not contain duplicates and that the documents are not too short. This is important to ensure that the dataset is of high quality and that it can be used to evaluate models in a meaningful way.

To calculate the descriptive statistics, you can simply run [`task.calculate_descriptive_statistics()`][mteb.AbsTask.calculate_descriptive_statistics].

## Submit a Pull Request

Once added, here is a checklist to ensure that everything works before you submit the PR:

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


Once submitted the PR will be reviewed by one of the organizers or contributors who might ask you to change things. Once the PR is approved the dataset will be added into the main repository.