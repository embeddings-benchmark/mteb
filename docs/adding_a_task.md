
# Adding a Task

To add a new task to MTEB, you need to do three things:

1) Implement a task with the desired dataset, by subclassing an abstract task
2) Add metadata to the task
3) Submit the edits to the [MTEB](https://github.com/embeddings-benchmark/mteb) repository

If you have any questions regarding this process feel free to open a discussion [thread](https://github.com/embeddings-benchmark/mteb/discussions). 

> Note: When we mention adding a dataset we refer to a subclass of one of the abstasks.

## 1) Creating a new subclass

To add a new task, you need to implement a new class that inherits from the `AbsTask` associated with the task type (e.g. `AbsTaskReranking` for reranking tasks). You can find the supported task types in [here](https://github.com/embeddings-benchmark/mteb-draft/tree/main/mteb/abstasks).

```python
from mteb import MTEB
from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from sentence_transformers import SentenceTransformer


class MindSmallReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="SciDocsRR",
        description="Ranking of related scientific papers based on their title.",
        reference="https://allenai.org/data/scidocs",
        hf_hub_name="mteb/scidocs-reranking",
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="map",
        revision="d3c5e1fc0b855ab6097bf1cda04dd73947d7caab",
        date="2020",
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

# testing the task with a model:
model = SentenceTransformer("average_word_embeddings_komninos")
evaluation = MTEB(tasks=[MindSmallReranking()])
evaluation.run(model)
```

> **Note:** for multilingual tasks, make sure your class also inherits from the `MultilingualTask` class like in [this](https://github.com/embeddings-benchmark/mteb-draft/blob/main/mteb/tasks/Classification/MTOPIntentClassification.py) example.  
> For cross-lingual tasks, make sure your class also inherits from the `CrosslingualTask` class like in [this](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/BitextMining/TatoebaBitextMining.py).

Often the dataset from HuggingFace is not in the format expected by MTEB. To resolve this you can add a `dataset_transform` method to your dataset to ensure it is in the right format. Here is an example along with some design considerations: 


<details closed>
<summary>An additional example</summary>
<br>

```python
class VGClustering(AbsTaskClustering):
    @property
    def description(self) -> dict[str, Any]:
        return {
            "name": "VGClustering",
            "hf_hub_name": "navjordj/VG_summarization",
            "description": "Articles and their classes (e.g. sports) from VG news articles extracted from Norsk Aviskorpus.",
            "reference": "https://huggingface.co/datasets/navjordj/VG_summarization",
            "type": "Clustering",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["nb"],
            "main_score": "v_measure",
            "revision": "d4c5a8ba10ae71224752c727094ac4c46947fa29",
        }

    def load_data(self, **kwargs: dict):  # noqa: ARG002
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset: datasets.DatasetDict = datasets.load_dataset(
            self.description["hf_hub_name"],
            revision=self.description.get("revision"),
        )

        self.dataset_transform()
        self.data_loaded = True

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

Along with the dataset MMTEB requires metadata regarding the dataset. If the metadata isn't available please provide your best guess or leave the field blank.


```python
metadata = {
    "date": "2012-01-01/2020-01-01", # str # date range following ISO 8601
    "form": ["written"], # list[str]
    "domains": ["non-fiction", "news"], # list[str]
    "dialect": [], # list[str]
    "task_subtypes": ["Thematic Clustering"], # list[str]
    "license": "CC-BY-NC",
    "socioeconomic_status": "high", # str
    "annotations_creators": "derived", # str
    "text_creation": "found", # str
    "citation": """
@mastersthesis{navjord2023beyond,
  title={Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
  author={Navjord, J{\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
  year={2023},
  school={Norwegian University of Life Sciences, {\AA}s}
}
""",  # str
}

# remember to add it to the class
class MyNewDataset(AbsTaskClustering):
    ...

    @property
    def metadata(self):
        return metadata
```

The following tables give a description to each of the fields:  


| **Field**              | **Description**                                                                                                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `date`                 | When was the text written? Range should following ISO 8601 format (reasonable guesses are allowed)                                                                                          |
| `form`                 | Either "spoken" or "written"                                                                                                                                                                |
| `domains`              | See table below of full list of domains, feel free to suggest new domains                                                                                                                   |
| `task_subtypes`        | See table below of full list of task subtypes, feel free to suggest new subtypes                                                                                                            |
| `license`              | The license of the dataet                                                                                                                                                                   |
| `citation`             | How we will cite the dataset within the publication                                                                                                                                         |
| `socioeconomic_status` | The socioeconomic status of the writer. Can be "high", "low", "middle", "mixed", "unknown"                                                                                                  |
| `annotations_creators` | How was the annotations made? Options include "expert-annotated", "human-annotated", "derived" (e.g. derived from website structure)                                                        |
| `text_creation`        | How was the text created? Options include "found", "created", "machine-translated", "human-translated and localized", "machine-translated and verified", "machine-translated and localized" |

Note that these fields can be left blank if the information is not available and can be extended if necessary. We do not include any machine-translated (without verification) datasets in the benchmark.

<details closed>
<summary>Domains</summary>
<br>

The domains follow the categories used in the [Universal Dependencies project](https://universaldependencies.org), though we updated them were deemed appropriate. These do no have to be mutually exclusive.

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

These domains subtypes were introduced in the [Scandinavian Embedding Benchmark](https://openreview.net/pdf/f5f1953a9c798ec61bb050e62bc7a94037fd4fab.pdf) and is intended to be extended.



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



## submit a PR

Once you are finished create a PR to the [MTEB](https://github.com/embeddings-benchmark/mteb) repository. If you haven't created a PR before please refer to the [GitHub documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/)

The PR will be reviewed by one of the organizers or contributors who might ask you to change things. Once the PR is approved the dataset will be added into the main repository. 


Before you commit here is a checklist you should consider completing before submitting:

- [ ] I have tested that the dataset runs

E.g. by ensuring it is placed in the right folder

An easy way to test it is using:
```python
from mteb import MTEB
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

model = SentenceTransformer(model_name)
evaluation = MTEB(tasks=["{the name of your task}"])
```

- [ ] I have run at least the `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` on the dataset (and attached the results)
- [ ] I have considered the size of the dataset and reduced it if it is too big (2048 examples is typically large enough for most tasks)
- [ ] Run tests locally to make sure nothing is broken using `make test`. 
- [ ] Run black formatter to format the code using `make lint`.
- [ ] I have added the description of the dataset to the [table in the Readme.md](https://github.com/embeddings-benchmark/mteb?tab=readme-ov-file#available-tasks)https://github.com/embeddings-benchmark/mteb?tab=readme-ov-file#available-tasks
