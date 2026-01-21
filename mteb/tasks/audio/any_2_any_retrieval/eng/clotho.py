from collections import defaultdict

from datasets import Dataset, DatasetDict, load_dataset
from tqdm.auto import tqdm

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ClothoA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ClothoA2TRetrieval",
        description="An audio captioning datasetst containing audio clips and their corresponding captions.",
        reference="https://github.com/audio-captioning/clotho-dataset",
        dataset={
            "path": "mteb/Clotho",
            "revision": "c44521cd4067f134e5f5bace4290b59ed773b451",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{drossos2019clothoaudiocaptioningdataset,
  archiveprefix = {arXiv},
  author = {Konstantinos Drossos and Samuel Lipping and Tuomas Virtanen},
  eprint = {1910.09387},
  primaryclass = {cs.SD},
  title = {Clotho: An Audio Captioning Dataset},
  url = {https://arxiv.org/abs/1910.09387},
  year = {2019},
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        ds = load_dataset(**self.metadata.dataset, split="test", keep_in_memory=False)

        queries_ds = ds.select_columns(["index", "audio"]).rename_column("index", "id")

        # Corpus: need to split captions, so we build this
        corpus_data = {"id": [], "text": []}
        qrels_dict = defaultdict(dict)

        for row in tqdm(ds, total=len(ds), desc="Loading Clotho AT2 Retrieval Data"):
            index = row["index"]

            for i, text in enumerate(row["text"].split(".")):
                doc_id = f"d-{index}-{i}"
                corpus_data["id"].append(doc_id)
                corpus_data["text"].append(text)
                qrels_dict[index][doc_id] = 1

        self.corpus = DatasetDict({"test": Dataset.from_dict(corpus_data)})
        self.queries = DatasetDict({"test": queries_ds})
        self.relevant_docs = {"test": qrels_dict}
        self.data_loaded = True


class ClothoT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ClothoT2ARetrieval",
        description="An audio captioning datasetst containing audio clips from the Freesound platform and their corresponding captions.",
        reference="https://github.com/audio-captioning/clotho-dataset",
        dataset={
            "path": "mteb/Clotho",
            "revision": "c44521cd4067f134e5f5bace4290b59ed773b451",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{drossos2019clothoaudiocaptioningdataset,
  archiveprefix = {arXiv},
  author = {Konstantinos Drossos and Samuel Lipping and Tuomas Virtanen},
  eprint = {1910.09387},
  primaryclass = {cs.SD},
  title = {Clotho: An Audio Captioning Dataset},
  url = {https://arxiv.org/abs/1910.09387},
  year = {2019},
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        ds = load_dataset(**self.metadata.dataset, split="test", keep_in_memory=False)

        # Corpus: reuse dataset with column operations (no copy for audio)
        corpus_ds = ds.select_columns(["index", "audio"]).rename_column("index", "id")

        queries_data = {"id": [], "text": []}
        qrels_dict = defaultdict(dict)

        for row in ds:
            index = row["index"]

            for i, text in enumerate(row["text"].split(".")):
                query_id = f"q-{index}-{i}"
                queries_data["id"].append(query_id)
                queries_data["text"].append(text)
                qrels_dict[query_id][index] = 1

        self.corpus = DatasetDict({"test": corpus_ds})
        self.queries = DatasetDict({"test": Dataset.from_dict(queries_data)})
        self.relevant_docs = {"test": qrels_dict}
        self.data_loaded = True
