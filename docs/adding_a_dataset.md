
# Adding a Dataset

To add a new dataset to MTEB, you need to do three things:

1) Implement a task with the desired dataset, by subclassing an abstract task
2) Add metadata to the task
3) Submit the edits to the [MTEB](https://github.com/embeddings-benchmark/mteb) repository

If you have any questions regarding this process feel free to open a discussion [thread](https://github.com/embeddings-benchmark/mteb/discussions).

> Note: When we mention adding a dataset we refer to a subclass of one of the abstasks.

## 1) Creating a new subclass

### A Simple Example

To add a new task, you need to implement a new class that inherits from the `AbsTask` associated with the task type (e.g. `AbsTaskReranking` for reranking tasks). You can find the supported task types in [here](https://github.com/embeddings-benchmark/mteb-draft/tree/main/mteb/abstasks).

```python
from mteb import MTEB
from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from sentence_transformers import SentenceTransformer
from mteb.abstasks.TaskMetadata import TaskMetadata

class SciDocsReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="SciDocsRR",
        description="Ranking of related scientific papers based on their title.",
        reference="https://allenai.org/data/scidocs",
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="map",
        dataset={
            "path": "mteb/scidocs-reranking",
            "revision": "d3c5e1fc0b855ab6097bf1cda04dd73947d7caab",
        }
        date=None,
        form="written",
        domains=["Academic", "Non-fiction"],
        task_subtypes=["Scientific Reranking"],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators=None,
        dialect=None,
        text_creation="found",
        bibtex_citation= ... # removed for brevity
        n_samples={"test": 19599},
        avg_character_length={"test": 69.0},
)

# testing the task with a model:
model = SentenceTransformer("average_word_embeddings_komninos")
evaluation = MTEB(tasks=[MindSmallReranking()])
evaluation.run(model)
```

> **Note:** for multilingual tasks, make sure your class also inherits from the `MultilingualTask` class like in [this](https://github.com/embeddings-benchmark/mteb-draft/blob/main/mteb/tasks/Classification/MTOPIntentClassification.py) example.
> For cross-lingual tasks, make sure your class also inherits from the `CrosslingualTask` class like in [this](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/BitextMining/TatoebaBitextMining.py).



### A Detailed Example
Often the dataset from HuggingFace is not in the format expected by MTEB. To resolve this you can either change the format on Hugging Face or add a `dataset_transform` method to your dataset to transform it into the right format on the fly. Here is an example along with some design considerations:

```python
class VGClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="VGClustering",
        description="Articles and their classes (e.g. sports) from VG news articles extracted from Norsk Aviskorpus.",
        reference="https://huggingface.co/datasets/navjordj/VG_summarization",
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["nb"],
        main_score="v_measure",
        dataset={
            "path": "navjordj/VG_summarization",
            "revision": "d4c5a8ba10ae71224752c727094ac4c46947fa29",
        },
        date=("2012-01-01", "2020-01-01"),
        form="written",
        domains=["Academic", "Non-fiction"],
        task_subtypes=["Scientific Reranking"],
        license="cc-by-nc",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation= ... # removed for brevity
)

    def dataset_transform(self):
        splits = self.description["eval_splits"]

        documents: list = []
        labels: list = []
        label_col = "classes"

        ds = {}
        for split in splits:
            ds_split = self.dataset[split]

            _label = self.normalize_labels(ds_split[label_col])
            documents.extend(ds_split["title"])
            labels.extend(_label)

            documents.extend(ds_split["ingress"])
            labels.extend(_label)

            documents.extend(ds_split["article"])
            labels.extend(_label)

            assert len(documents) == len(labels)

            rng = random.Random(1111)  # local only seed
            pairs = list(zip(documents, labels))
            rng.shuffle(pairs)
            documents, labels = [list(collection) for collection in zip(*pairs)]

            # To get a more robust estimate we create batches of size 512, this decision can vary depending on dataset
            documents_batched = list(batched(documents, 512))
            labels_batched = list(batched(labels, 512))

            # reduce the size of the dataset as we see that we obtain a consistent scores (if we change the seed) even
            # with only 512x4 samples.
            documents_batched = documents_batched[:4]
            labels_batched = labels_batched[:4]


            ds[split] = datasets.Dataset.from_dict(
                {
                    "sentences": documents_batched,
                    "labels": labels_batched,
                }
            )

        self.dataset = datasets.DatasetDict(ds)
```

</details>


## 2) Creating the metadata object
Along with the task MTEB requires metadata regarding the task. If the metadata isn't available please provide your best guess or leave the field as `None`.

To get an overview of the fields in the metadata object, you can look at the [TaskMetadata](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/abstasks/TaskMetadata.py) class.


Note that these fields can be left blank if the information is not available and can be extended if necessary. We do not include any machine-translated (without verification) datasets in the benchmark.

<details closed>
<summary>Domains</summary>
<br>

The domains follow the categories used in the [Universal Dependencies project](https://universaldependencies.org), though we updated them where deemed appropriate. These do not have to be mutually exclusive.

| **Domain**    | **Description**                                                  |
| ------------- | ---------------------------------------------------------------- |
| Academic      | Academic writing                                                 |
| Religious     | Religious text e.g. bibles                                       |
| Blog          | [Blogpost, weblog etc.](https://en.wikipedia.org/wiki/Blog)      |
| Fiction       | Works of [fiction](https://en.wikipedia.org/wiki/Fiction)        |
| Government    | Governmental communication, websites or similar                  |
| Legal         | Legal documents, laws etc.                                       |
| Medical       | doctors notes, medical procedures or similar                     |
| News          | News articles, tabloids etc.                                     |
| Reviews       | Reviews e.g. user reviews of products                            |
| Non-fiction   | [non-fiction](https://en.wikipedia.org/wiki/Non-fiction) writing |
| Poetry        | Poems, Epics etc.                                                |
| Social        | social media content                                             |
| Spoken        | Spoken dialogues                                                 |
| Encyclopaedic | E.g. Wikipedias                                                  |
| Web           | Web content                                                      |


</details>


<br>
<details closed>
<summary>Task Subtypes</summary>
<br>

These domains subtypes were introduced in the [Scandinavian Embedding Benchmark](https://openreview.net/pdf/f5f1953a9c798ec61bb050e62bc7a94037fd4fab.pdf) and are intended to be extended as needed.



| Formalization           | Task                     | Description                                                                                                     |
| ----------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------- |
| **Retrieval**           |                          | Retrieval focuses on locating and providing relevant information or documents based on a query.                 |
|                         | Question answering       | Finding answers to queries in a dataset, focusing on exact answers or relevant passages.                        |
|                         | Article retrieval        | Identifying and retrieving full articles that are relevant to a given query.                                    |
| **Bitext Mining**       |                          | Bitext mining involves identifying parallel texts across languages or dialects for translation or analysis.     |
|                         | Dialect pairing          | Identifying pairs of text that are translations of each other across different dialects.                        |
| **Classification**      |                          | Classification is the process of categorizing text into predefined groups or classes based on their content.    |
|                         | Political                | Categorizing text according to political orientation or content.                                                |
|                         | Language Identification  | Determining the language in which a given piece of text is written.                                             |
|                         | Linguistic Acceptability | Assessing whether a sentence is grammatically correct according to linguistic norms.                            |
|                         | Sentiment/Hate Speech    | Detecting the sentiment of text or identifying hate speech within the content.                                  |
|                         | Dialog Systems           | Creating or evaluating systems capable of conversing with humans in a natural manner.                           |
| **Clustering**          |                          | Clustering involves grouping sets of texts together based on their similarity without pre-defined labels.       |
|                         | Thematic Clustering      | Grouping texts based on their thematic similarity without prior labeling.                                       |
| **Reranking**           |                          | Reranking adjusts the order of items in a list to improve relevance or accuracy according to specific criteria. |
| **Pair Classification** |                          | Pair classification assesses relationships between pairs of items, such as texts, to classify their connection. |
| **STS**                 |                          | Semantic Textual Similarity measures the degree of semantic equivalence between two pieces of text.             |


</details>



## Submit a PR

Once you are finished create a PR to the [MTEB](https://github.com/embeddings-benchmark/mteb) repository. If you haven't created a PR before please refer to the [GitHub documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/)

The PR will be reviewed by one of the organizers or contributors who might ask you to change things. Once the PR is approved the dataset will be added into the main repository.


Before you commit here is a checklist you should consider completing before submitting:

- [ ] I have tested that the dataset runs with the `mteb` package.

An easy way to test it is using:
```python
from mteb import MTEB
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

model = SentenceTransformer(model_name)
evaluation = MTEB(tasks=[YourNewTask()])
```

- [ ] I have run the following models on the task (adding the results to the pr). These can be run using the `mteb run -m {model_name} -t {task_name}` command.
  - [ ] `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - [ ] `intfloat/multilingual-e5-small`
- [ ] I have checked that the performance is neither trivial (both models gain close to perfect scores) nor random (both models gain close to random scores).
- [ ] I have considered the size of the dataset and reduced it if it is too big (2048 examples is typically large enough for most tasks)
- [ ] Run tests locally to make sure nothing is broken using `make test`.
- [ ] Run the formatter to format the code using `make lint`.
