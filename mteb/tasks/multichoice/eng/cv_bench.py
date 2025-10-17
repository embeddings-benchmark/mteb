from collections import defaultdict

from datasets import Dataset, load_dataset

from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def _load_data(
    path: str,
    splits: list[str],
    revision: str | None = None,
    subtask: str = "Count",
):
    corpus = {}
    queries = {}
    relevant_docs = {}
    top_ranked = defaultdict(lambda: defaultdict(list))

    dataset = load_dataset(
        path,
        revision=revision,
    )
    dataset = dataset.filter(lambda example: example["task"] == subtask)
    for split in splits:
        split_dataset = dataset[split]

        split_dataset = split_dataset.map(
            transform_choices,
            remove_columns=[
                "idx",
                "type",
                "filename",
                "source",
                "source_dataset",
                "source_filename",
                "target_class",
                "target_size",
                "bbox",
                "prompt",
            ],
        )

        queries[split] = split_dataset.map(
            lambda x, idx: {
                "id": f"query-{split}-{idx}",
                "text": x["question"],
                "modality": "image,text",
            },
            with_indices=True,
            remove_columns=["answer", "choices", "task"],
        )

        corpus_element = []
        corpus_to_id = {}
        relevant_docs[split] = {}

        for idx, entry in enumerate(split_dataset):
            choices = entry["choices"]
            answer = choices[entry["answer"]]

            query_id = f"query-{split}-{idx}"

            for choice in choices:
                if choice not in corpus_to_id:
                    corpus_id = len(corpus_element)
                    corpus_element.append(choice)
                    corpus_to_id[choice] = f"corpus-{split}-{corpus_id}"

                is_relevant = 1 if choice == answer else 0
                if query_id not in relevant_docs[split]:
                    relevant_docs[split][query_id] = {}
                relevant_docs[split][query_id][corpus_to_id[choice]] = is_relevant
            corpus_ids = [corpus_id for _, corpus_id in corpus_to_id.items()]
            docs = [doc for doc, _ in corpus_to_id.items()]
        corpus_records = []
        for corpus_id, doc in zip(corpus_ids, docs):
            corpus_records.append({"id": corpus_id, "text": doc, "modality": "text"})
        corpus[split] = Dataset.from_list(corpus_records)

        for query_id, relevant in relevant_docs[split].items():
            for corpus_id, score in relevant.items():
                top_ranked[split][query_id].append(corpus_id)

    return corpus, queries, relevant_docs, top_ranked


def transform_choices(example):
    mapping = {"(A)": 0, "(B)": 1, "(C)": 2, "(D)": 3, "(E)": 4, "(F)": 5}
    example["answer"] = mapping[example["answer"]]
    return example


class CVBenchCount(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CVBenchCount",
        description="count the number of objects in the image.",
        reference="https://arxiv.org/pdf/2406.16860",
        dataset={
            "path": "nyu-visionx/CV-Bench",
            "revision": "22409a927ab5cf68e3655023d51694587455fc99",
        },
        type="VisionCentricQA",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-01-01", "2024-06-24"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="found",
        bibtex_citation=r"""
@article{tong2024cambrian,
  author = {Tong, Shengbang and Brown, Ellis and Wu, Penghao and Woo, Sanghyun and Middepogu, Manoj and Akula, Sai Charitha and Yang, Jihan and Yang, Shusheng and Iyer, Adithya and Pan, Xichen and others},
  journal = {arXiv preprint arXiv:2406.16860},
  title = {Cambrian-1: A fully open, vision-centric exploration of multimodal llms},
  year = {2024},
}
""",
    )

    def load_data(self) -> None:
        self.corpus, self.queries, self.relevant_docs, self.top_ranked = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
            subtask="Count",
        )
        self.data_loaded = True


class CVBenchRelation(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CVBenchRelation",
        description="decide the relation of the objects in the image.",
        reference="https://arxiv.org/pdf/2406.16860",
        dataset={
            "path": "nyu-visionx/CV-Bench",
            "revision": "22409a927ab5cf68e3655023d51694587455fc99",
        },
        type="VisionCentricQA",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-01-01", "2024-06-24"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{tong2024cambrian,
  author = {Tong, Shengbang and Brown, Ellis and Wu, Penghao and Woo, Sanghyun and Middepogu, Manoj and Akula, Sai Charitha and Yang, Jihan and Yang, Shusheng and Iyer, Adithya and Pan, Xichen and others},
  journal = {arXiv preprint arXiv:2406.16860},
  title = {Cambrian-1: A fully open, vision-centric exploration of multimodal llms},
  year = {2024},
}
""",
    )

    def load_data(self) -> None:
        self.corpus, self.queries, self.relevant_docs, self.top_ranked = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
            subtask="Relation",
        )
        self.data_loaded = True


class CVBenchDepth(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CVBenchDepth",
        description="judge the depth of the objects in the image with similarity matching.",
        reference="https://arxiv.org/pdf/2406.16860",
        dataset={
            "path": "nyu-visionx/CV-Bench",
            "revision": "22409a927ab5cf68e3655023d51694587455fc99",
        },
        type="VisionCentricQA",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-01-01", "2024-06-24"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{tong2024cambrian,
  author = {Tong, Shengbang and Brown, Ellis and Wu, Penghao and Woo, Sanghyun and Middepogu, Manoj and Akula, Sai Charitha and Yang, Jihan and Yang, Shusheng and Iyer, Adithya and Pan, Xichen and others},
  journal = {arXiv preprint arXiv:2406.16860},
  title = {Cambrian-1: A fully open, vision-centric exploration of multimodal llms},
  year = {2024},
}
""",
    )

    def load_data(self) -> None:
        self.corpus, self.queries, self.relevant_docs, self.top_ranked = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
            subtask="Depth",
        )
        self.data_loaded = True


class CVBenchDistance(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CVBenchDistance",
        description="judge the distance of the objects in the image with similarity matching.",
        reference="https://arxiv.org/pdf/2406.16860",
        dataset={
            "path": "nyu-visionx/CV-Bench",
            "revision": "22409a927ab5cf68e3655023d51694587455fc99",
        },
        type="VisionCentricQA",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-01-01", "2024-06-24"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{tong2024cambrian,
  author = {Tong, Shengbang and Brown, Ellis and Wu, Penghao and Woo, Sanghyun and Middepogu, Manoj and Akula, Sai Charitha and Yang, Jihan and Yang, Shusheng and Iyer, Adithya and Pan, Xichen and others},
  journal = {arXiv preprint arXiv:2406.16860},
  title = {Cambrian-1: A fully open, vision-centric exploration of multimodal llms},
  year = {2024},
}
""",
    )

    def load_data(self) -> None:
        self.corpus, self.queries, self.relevant_docs, self.top_ranked = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
            subtask="Distance",
        )
        self.data_loaded = True
