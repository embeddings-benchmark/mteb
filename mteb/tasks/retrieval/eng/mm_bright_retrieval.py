from __future__ import annotations

from typing import Any

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalDatasetLoader
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.timing import TimingStack

_COMMON_METADATA = dict(
    reference="https://arxiv.org/abs/2601.09562",
    eval_splits=["test"],
    main_score="ndcg_at_10",
    date=("2025-01-01", "2026-01-15"),
    domains=["Academic", "Web", "Medical", "Legal", "Religious"],
    license="cc-by-4.0",
    annotations_creators="expert-annotated",
    dialect=[],
    sample_creation="found",
    eval_langs=["eng-Latn"],
    bibtex_citation=r"""
@article{abdallah2026mmbright,
  archiveprefix = {arXiv},
  author = {Abdelrahman Abdallah and Mohamed Darwish Mounis and Mahmoud Abdalla and Mahmoud SalahEldin Kasem and Mostafa Farouk Senussi and Mohamed Mahmoud and Mohammed Ali and Adam Jatowt and Hyun-Soo Kang},
  eprint = {2601.09562},
  primaryclass = {cs.IR},
  title = {{MM-BRIGHT}: A Multi-Task Multimodal Benchmark for Reasoning-Intensive Retrieval},
  url = {https://arxiv.org/abs/2601.09562},
  year = {2026},
}
""",
)


def _t2t_dataset_transform(
    self: AbsTaskRetrieval, num_proc: int | None = None, **kwargs: Any
) -> None:
    """Drop the `image` column: the hub repo's queries table is shared with
    the IT2T task and carries an image for the multimodal-curated rows."""
    for subset in self.dataset:
        for split in self.dataset[subset]:
            queries = self.dataset[subset][split]["queries"]
            self.dataset[subset][split]["queries"] = queries.remove_columns(["image"])


def _it2t_dataset_transform(
    self: AbsTaskRetrieval, num_proc: int | None = None, **kwargs: Any
) -> None:
    """Keep only the rows of the shared queries table that were curated for
    the multimodal (image-bearing) evaluation set."""
    for subset in self.dataset:
        for split in self.dataset[subset]:
            data = self.dataset[subset][split]
            queries = data["queries"]
            keep_indices = [
                i for i, image in enumerate(queries["image"]) if image is not None
            ]
            queries = queries.select(keep_indices)
            keep_ids = set(queries["id"])
            data["queries"] = queries
            data["relevant_docs"] = {
                query_id: docs
                for query_id, docs in data["relevant_docs"].items()
                if query_id in keep_ids
            }
            if data["top_ranked"] is not None:
                data["top_ranked"] = {
                    query_id: docs
                    for query_id, docs in data["top_ranked"].items()
                    if query_id in keep_ids
                }


def _it2i_load_data(
    self: AbsTaskRetrieval,
    num_proc: int | None = None,
    *,
    timer: TimingStack | None = None,
    **kwargs: Any,
) -> None:
    """Build IT2I's split from two hf_subsets of the same repo: the shared
    "default" queries table (filtered to image-bearing rows sharing an id
    with the image qrels) and the "image" subset's corpus/qrels."""
    if self.data_loaded:
        return
    timer = timer or TimingStack()
    dataset_path = self.metadata.dataset["path"]
    revision = self.metadata.dataset["revision"]
    self.dataset = {"default": {}}
    with timer("Data loading", log_message=f"Loading dataset {self.metadata.name}..."):
        for split in self.eval_splits:
            default_data = RetrievalDatasetLoader(
                hf_repo=dataset_path,
                revision=revision,
                split=split,
                config="default",
            ).load(num_proc=num_proc)
            image_corpus = load_dataset(
                dataset_path, "image-corpus", split=split, revision=revision
            )
            image_qrels_ds = load_dataset(
                dataset_path, "image-qrels", split=split, revision=revision
            )
            image_qrels: dict[str, dict[str, int]] = {}
            for row in image_qrels_ds:
                image_qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = row[
                    "score"
                ]
            keep_ids = set(image_qrels.keys())
            queries = default_data["queries"]
            keep_indices = [
                i for i, query_id in enumerate(queries["id"]) if query_id in keep_ids
            ]
            self.dataset["default"][split] = {
                "corpus": image_corpus,
                "queries": queries.select(keep_indices),
                "relevant_docs": image_qrels,
                "top_ranked": None,
            }
    with timer("Dataset transform"):
        self.dataset_transform(num_proc=num_proc)
    self.data_loaded = True


class MMBrightAcademiaT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightAcademiaT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Academia domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightAcademiaRetrieval",
            "revision": "af4ca37a456ec4a028b02643d563b22cc45c1261",
        },
        **_COMMON_METADATA,
    )


class MMBrightAcademiaIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightAcademiaIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Academia domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightAcademiaRetrieval",
            "revision": "af4ca37a456ec4a028b02643d563b22cc45c1261",
        },
        **_COMMON_METADATA,
    )


class MMBrightAcademiaIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightAcademiaIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Academia domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightAcademiaRetrieval",
            "revision": "af4ca37a456ec4a028b02643d563b22cc45c1261",
        },
        **_COMMON_METADATA,
    )


class MMBrightAppleT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightAppleT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Apple domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightAppleRetrieval",
            "revision": "faa769cd153101a5acd102df6f85c8613e6665a5",
        },
        **_COMMON_METADATA,
    )


class MMBrightAppleIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightAppleIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Apple domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightAppleRetrieval",
            "revision": "faa769cd153101a5acd102df6f85c8613e6665a5",
        },
        **_COMMON_METADATA,
    )


class MMBrightAppleIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightAppleIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Apple domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightAppleRetrieval",
            "revision": "faa769cd153101a5acd102df6f85c8613e6665a5",
        },
        **_COMMON_METADATA,
    )


class MMBrightAskUbuntuT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightAskUbuntuT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Ask Ubuntu domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightAskUbuntuRetrieval",
            "revision": "2073594359f9a0afc6309eb7e64226ca89f0992c",
        },
        **_COMMON_METADATA,
    )


class MMBrightAskUbuntuIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightAskUbuntuIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Ask Ubuntu domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightAskUbuntuRetrieval",
            "revision": "2073594359f9a0afc6309eb7e64226ca89f0992c",
        },
        **_COMMON_METADATA,
    )


class MMBrightAskUbuntuIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightAskUbuntuIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Ask Ubuntu domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightAskUbuntuRetrieval",
            "revision": "2073594359f9a0afc6309eb7e64226ca89f0992c",
        },
        **_COMMON_METADATA,
    )


class MMBrightAviationT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightAviationT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Aviation domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightAviationRetrieval",
            "revision": "1d19dc4b843495bfee3d37b22d91c4090f190747",
        },
        **_COMMON_METADATA,
    )


class MMBrightAviationIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightAviationIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Aviation domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightAviationRetrieval",
            "revision": "1d19dc4b843495bfee3d37b22d91c4090f190747",
        },
        **_COMMON_METADATA,
    )


class MMBrightAviationIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightAviationIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Aviation domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightAviationRetrieval",
            "revision": "1d19dc4b843495bfee3d37b22d91c4090f190747",
        },
        **_COMMON_METADATA,
    )


class MMBrightBioacousticsT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightBioacousticsT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Bioacoustics domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightBioacousticsRetrieval",
            "revision": "070b2a0ca326bcaad71c34737a97a1684758d06c",
        },
        **_COMMON_METADATA,
    )


class MMBrightBioacousticsIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightBioacousticsIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Bioacoustics domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightBioacousticsRetrieval",
            "revision": "070b2a0ca326bcaad71c34737a97a1684758d06c",
        },
        **_COMMON_METADATA,
    )


class MMBrightBioacousticsIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightBioacousticsIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Bioacoustics domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightBioacousticsRetrieval",
            "revision": "070b2a0ca326bcaad71c34737a97a1684758d06c",
        },
        **_COMMON_METADATA,
    )


class MMBrightBioinformaticsT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightBioinformaticsT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Bioinformatics domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightBioinformaticsRetrieval",
            "revision": "ac53f9241fed97d227a1c50a7539c84ad79a71ec",
        },
        **_COMMON_METADATA,
    )


class MMBrightBioinformaticsIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightBioinformaticsIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Bioinformatics domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightBioinformaticsRetrieval",
            "revision": "ac53f9241fed97d227a1c50a7539c84ad79a71ec",
        },
        **_COMMON_METADATA,
    )


class MMBrightBioinformaticsIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightBioinformaticsIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Bioinformatics domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightBioinformaticsRetrieval",
            "revision": "ac53f9241fed97d227a1c50a7539c84ad79a71ec",
        },
        **_COMMON_METADATA,
    )


class MMBrightBiologyT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightBiologyT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Biology domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightBiologyRetrieval",
            "revision": "bf01be7bde5e4c9d8dbeb88a85e5be19f9600ec2",
        },
        **_COMMON_METADATA,
    )


class MMBrightBiologyIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightBiologyIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Biology domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightBiologyRetrieval",
            "revision": "bf01be7bde5e4c9d8dbeb88a85e5be19f9600ec2",
        },
        **_COMMON_METADATA,
    )


class MMBrightBiologyIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightBiologyIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Biology domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightBiologyRetrieval",
            "revision": "bf01be7bde5e4c9d8dbeb88a85e5be19f9600ec2",
        },
        **_COMMON_METADATA,
    )


class MMBrightBitcoinT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightBitcoinT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Bitcoin domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightBitcoinRetrieval",
            "revision": "6290eec70421f2c728c8aba4086a3a1e84846093",
        },
        **_COMMON_METADATA,
    )


class MMBrightBitcoinIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightBitcoinIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Bitcoin domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightBitcoinRetrieval",
            "revision": "6290eec70421f2c728c8aba4086a3a1e84846093",
        },
        **_COMMON_METADATA,
    )


class MMBrightBitcoinIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightBitcoinIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Bitcoin domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightBitcoinRetrieval",
            "revision": "6290eec70421f2c728c8aba4086a3a1e84846093",
        },
        **_COMMON_METADATA,
    )


class MMBrightChemistryT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightChemistryT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Chemistry domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightChemistryRetrieval",
            "revision": "dd744dd3b842ac78a7616e1256dbd59bcdc776aa",
        },
        **_COMMON_METADATA,
    )


class MMBrightChemistryIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightChemistryIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Chemistry domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightChemistryRetrieval",
            "revision": "dd744dd3b842ac78a7616e1256dbd59bcdc776aa",
        },
        **_COMMON_METADATA,
    )


class MMBrightChemistryIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightChemistryIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Chemistry domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightChemistryRetrieval",
            "revision": "dd744dd3b842ac78a7616e1256dbd59bcdc776aa",
        },
        **_COMMON_METADATA,
    )


class MMBrightChristianityT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightChristianityT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Christianity domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightChristianityRetrieval",
            "revision": "8d11b1aa16a8980f8363e04663de48b86a48b24b",
        },
        **_COMMON_METADATA,
    )


class MMBrightChristianityIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightChristianityIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Christianity domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightChristianityRetrieval",
            "revision": "8d11b1aa16a8980f8363e04663de48b86a48b24b",
        },
        **_COMMON_METADATA,
    )


class MMBrightChristianityIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightChristianityIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Christianity domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightChristianityRetrieval",
            "revision": "8d11b1aa16a8980f8363e04663de48b86a48b24b",
        },
        **_COMMON_METADATA,
    )


class MMBrightCryptoT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightCryptoT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Cryptography domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightCryptoRetrieval",
            "revision": "5c1989db39403982be54d88d2ce587f78bf3487a",
        },
        **_COMMON_METADATA,
    )


class MMBrightCryptoIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightCryptoIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Cryptography domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightCryptoRetrieval",
            "revision": "5c1989db39403982be54d88d2ce587f78bf3487a",
        },
        **_COMMON_METADATA,
    )


class MMBrightCryptoIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightCryptoIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Cryptography domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightCryptoRetrieval",
            "revision": "5c1989db39403982be54d88d2ce587f78bf3487a",
        },
        **_COMMON_METADATA,
    )


class MMBrightEarthScienceT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightEarthScienceT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Earth Science domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightEarthScienceRetrieval",
            "revision": "b5847d4bae53d791414fbefe6e9a179a9996ca92",
        },
        **_COMMON_METADATA,
    )


class MMBrightEarthScienceIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightEarthScienceIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Earth Science domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightEarthScienceRetrieval",
            "revision": "b5847d4bae53d791414fbefe6e9a179a9996ca92",
        },
        **_COMMON_METADATA,
    )


class MMBrightEarthScienceIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightEarthScienceIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Earth Science domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightEarthScienceRetrieval",
            "revision": "b5847d4bae53d791414fbefe6e9a179a9996ca92",
        },
        **_COMMON_METADATA,
    )


class MMBrightEconomicsT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightEconomicsT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Economics domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightEconomicsRetrieval",
            "revision": "e245567b8bf0dc2a42872a845870bcaddce6d316",
        },
        **_COMMON_METADATA,
    )


class MMBrightEconomicsIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightEconomicsIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Economics domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightEconomicsRetrieval",
            "revision": "e245567b8bf0dc2a42872a845870bcaddce6d316",
        },
        **_COMMON_METADATA,
    )


class MMBrightEconomicsIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightEconomicsIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Economics domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightEconomicsRetrieval",
            "revision": "e245567b8bf0dc2a42872a845870bcaddce6d316",
        },
        **_COMMON_METADATA,
    )


class MMBrightGamingT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightGamingT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Gaming domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightGamingRetrieval",
            "revision": "c85d91ddbf7d7708f4b73696deb0f76e59dce71a",
        },
        **_COMMON_METADATA,
    )


class MMBrightGamingIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightGamingIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Gaming domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightGamingRetrieval",
            "revision": "c85d91ddbf7d7708f4b73696deb0f76e59dce71a",
        },
        **_COMMON_METADATA,
    )


class MMBrightGamingIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightGamingIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Gaming domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightGamingRetrieval",
            "revision": "c85d91ddbf7d7708f4b73696deb0f76e59dce71a",
        },
        **_COMMON_METADATA,
    )


class MMBrightGIST2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightGIST2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the GIS domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightGISRetrieval",
            "revision": "8470bbfbff33477d5bd6a9f18d61fd8dd0ff642b",
        },
        **_COMMON_METADATA,
    )


class MMBrightGISIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightGISIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the GIS domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightGISRetrieval",
            "revision": "8470bbfbff33477d5bd6a9f18d61fd8dd0ff642b",
        },
        **_COMMON_METADATA,
    )


class MMBrightGISIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightGISIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the GIS domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightGISRetrieval",
            "revision": "8470bbfbff33477d5bd6a9f18d61fd8dd0ff642b",
        },
        **_COMMON_METADATA,
    )


class MMBrightIslamT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightIslamT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Islam domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightIslamRetrieval",
            "revision": "b7cde3a07cea5bbfffced75b48fa28fe9141c672",
        },
        **_COMMON_METADATA,
    )


class MMBrightIslamIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightIslamIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Islam domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightIslamRetrieval",
            "revision": "b7cde3a07cea5bbfffced75b48fa28fe9141c672",
        },
        **_COMMON_METADATA,
    )


class MMBrightIslamIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightIslamIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Islam domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightIslamRetrieval",
            "revision": "b7cde3a07cea5bbfffced75b48fa28fe9141c672",
        },
        **_COMMON_METADATA,
    )


class MMBrightLawT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightLawT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Law domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightLawRetrieval",
            "revision": "003bc8f29a8f2c0575544f455ad8111e51127e59",
        },
        **_COMMON_METADATA,
    )


class MMBrightLawIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightLawIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Law domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightLawRetrieval",
            "revision": "003bc8f29a8f2c0575544f455ad8111e51127e59",
        },
        **_COMMON_METADATA,
    )


class MMBrightLawIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightLawIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Law domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightLawRetrieval",
            "revision": "003bc8f29a8f2c0575544f455ad8111e51127e59",
        },
        **_COMMON_METADATA,
    )


class MMBrightMathT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightMathT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Mathematics domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightMathRetrieval",
            "revision": "51fc7683cc39ffeab4a0d2b6702c89b417158668",
        },
        **_COMMON_METADATA,
    )


class MMBrightMathIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightMathIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Mathematics domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightMathRetrieval",
            "revision": "51fc7683cc39ffeab4a0d2b6702c89b417158668",
        },
        **_COMMON_METADATA,
    )


class MMBrightMathIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightMathIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Mathematics domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightMathRetrieval",
            "revision": "51fc7683cc39ffeab4a0d2b6702c89b417158668",
        },
        **_COMMON_METADATA,
    )


class MMBrightMedicalSciencesT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightMedicalSciencesT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Medical Sciences domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightMedicalSciencesRetrieval",
            "revision": "31dd7c25c354bb6457195f3b13f9e7f85efb233e",
        },
        **_COMMON_METADATA,
    )


class MMBrightMedicalSciencesIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightMedicalSciencesIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Medical Sciences domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightMedicalSciencesRetrieval",
            "revision": "31dd7c25c354bb6457195f3b13f9e7f85efb233e",
        },
        **_COMMON_METADATA,
    )


class MMBrightMedicalSciencesIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightMedicalSciencesIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Medical Sciences domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightMedicalSciencesRetrieval",
            "revision": "31dd7c25c354bb6457195f3b13f9e7f85efb233e",
        },
        **_COMMON_METADATA,
    )


class MMBrightPhilosophyT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightPhilosophyT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Philosophy domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightPhilosophyRetrieval",
            "revision": "ecc3b668d90119ee72d36fbe6e7e13436ff5e0ed",
        },
        **_COMMON_METADATA,
    )


class MMBrightPhilosophyIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightPhilosophyIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Philosophy domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightPhilosophyRetrieval",
            "revision": "ecc3b668d90119ee72d36fbe6e7e13436ff5e0ed",
        },
        **_COMMON_METADATA,
    )


class MMBrightPhilosophyIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightPhilosophyIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Philosophy domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightPhilosophyRetrieval",
            "revision": "ecc3b668d90119ee72d36fbe6e7e13436ff5e0ed",
        },
        **_COMMON_METADATA,
    )


class MMBrightPhysicsT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightPhysicsT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Physics domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightPhysicsRetrieval",
            "revision": "90215bd3eecd6a8d08654fb9ec44ee0de67c4a77",
        },
        **_COMMON_METADATA,
    )


class MMBrightPhysicsIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightPhysicsIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Physics domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightPhysicsRetrieval",
            "revision": "90215bd3eecd6a8d08654fb9ec44ee0de67c4a77",
        },
        **_COMMON_METADATA,
    )


class MMBrightPhysicsIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightPhysicsIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Physics domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightPhysicsRetrieval",
            "revision": "90215bd3eecd6a8d08654fb9ec44ee0de67c4a77",
        },
        **_COMMON_METADATA,
    )


class MMBrightProjectManagementT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightProjectManagementT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Project Management domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightProjectManagementRetrieval",
            "revision": "a3bbeb082957673cf68a8aa9c827d40e0f363753",
        },
        **_COMMON_METADATA,
    )


class MMBrightProjectManagementIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightProjectManagementIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Project Management domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightProjectManagementRetrieval",
            "revision": "a3bbeb082957673cf68a8aa9c827d40e0f363753",
        },
        **_COMMON_METADATA,
    )


class MMBrightProjectManagementIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightProjectManagementIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Project Management domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightProjectManagementRetrieval",
            "revision": "a3bbeb082957673cf68a8aa9c827d40e0f363753",
        },
        **_COMMON_METADATA,
    )


class MMBrightPsychologyT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightPsychologyT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Psychology domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightPsychologyRetrieval",
            "revision": "8d5fbe318260d7e6527dd368451ecbd7cf0cfd08",
        },
        **_COMMON_METADATA,
    )


class MMBrightPsychologyIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightPsychologyIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Psychology domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightPsychologyRetrieval",
            "revision": "8d5fbe318260d7e6527dd368451ecbd7cf0cfd08",
        },
        **_COMMON_METADATA,
    )


class MMBrightPsychologyIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightPsychologyIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Psychology domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightPsychologyRetrieval",
            "revision": "8d5fbe318260d7e6527dd368451ecbd7cf0cfd08",
        },
        **_COMMON_METADATA,
    )


class MMBrightQuantT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightQuantT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Quantitative Finance domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightQuantRetrieval",
            "revision": "51084196e110f6bc55ca6be872a80742e070764c",
        },
        **_COMMON_METADATA,
    )


class MMBrightQuantIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightQuantIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Quantitative Finance domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightQuantRetrieval",
            "revision": "51084196e110f6bc55ca6be872a80742e070764c",
        },
        **_COMMON_METADATA,
    )


class MMBrightQuantIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightQuantIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Quantitative Finance domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightQuantRetrieval",
            "revision": "51084196e110f6bc55ca6be872a80742e070764c",
        },
        **_COMMON_METADATA,
    )


class MMBrightQuantumComputingT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightQuantumComputingT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Quantum Computing domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightQuantumComputingRetrieval",
            "revision": "629acc9d41bfdb12965aca6e2f9c65a33af432eb",
        },
        **_COMMON_METADATA,
    )


class MMBrightQuantumComputingIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightQuantumComputingIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Quantum Computing domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightQuantumComputingRetrieval",
            "revision": "629acc9d41bfdb12965aca6e2f9c65a33af432eb",
        },
        **_COMMON_METADATA,
    )


class MMBrightQuantumComputingIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightQuantumComputingIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Quantum Computing domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightQuantumComputingRetrieval",
            "revision": "629acc9d41bfdb12965aca6e2f9c65a33af432eb",
        },
        **_COMMON_METADATA,
    )


class MMBrightRoboticsT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightRoboticsT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Robotics domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightRoboticsRetrieval",
            "revision": "a59181d3dd1efc8d51cdcddb6b5983f8569cc68f",
        },
        **_COMMON_METADATA,
    )


class MMBrightRoboticsIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightRoboticsIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Robotics domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightRoboticsRetrieval",
            "revision": "a59181d3dd1efc8d51cdcddb6b5983f8569cc68f",
        },
        **_COMMON_METADATA,
    )


class MMBrightRoboticsIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightRoboticsIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Robotics domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightRoboticsRetrieval",
            "revision": "a59181d3dd1efc8d51cdcddb6b5983f8569cc68f",
        },
        **_COMMON_METADATA,
    )


class MMBrightSalesforceT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightSalesforceT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Salesforce domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightSalesforceRetrieval",
            "revision": "c65b2cad4e96f460065c38fa3fc9f87c139c9618",
        },
        **_COMMON_METADATA,
    )


class MMBrightSalesforceIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightSalesforceIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Salesforce domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightSalesforceRetrieval",
            "revision": "c65b2cad4e96f460065c38fa3fc9f87c139c9618",
        },
        **_COMMON_METADATA,
    )


class MMBrightSalesforceIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightSalesforceIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Salesforce domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightSalesforceRetrieval",
            "revision": "c65b2cad4e96f460065c38fa3fc9f87c139c9618",
        },
        **_COMMON_METADATA,
    )


class MMBrightSustainabilityT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightSustainabilityT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Sustainability domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightSustainabilityRetrieval",
            "revision": "f88b2c2e1adcdd722b3408c8d4c9785f1f22a473",
        },
        **_COMMON_METADATA,
    )


class MMBrightSustainabilityIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightSustainabilityIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Sustainability domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightSustainabilityRetrieval",
            "revision": "f88b2c2e1adcdd722b3408c8d4c9785f1f22a473",
        },
        **_COMMON_METADATA,
    )


class MMBrightSustainabilityIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightSustainabilityIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Sustainability domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightSustainabilityRetrieval",
            "revision": "f88b2c2e1adcdd722b3408c8d4c9785f1f22a473",
        },
        **_COMMON_METADATA,
    )


class MMBrightTravelT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _t2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightTravelT2TRetrieval",
        description="MM-BRIGHT text queries retrieving reasoning-intensive technical passages in the Travel domain.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        task_subtypes=["Reasoning as Retrieval"],
        prompt={
            "query": "Given a technical question, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightTravelRetrieval",
            "revision": "b64e37c79fb9453e8ddd2e6167b783038e01d1a8",
        },
        **_COMMON_METADATA,
    )


class MMBrightTravelIT2TRetrieval(AbsTaskRetrieval):
    dataset_transform = _it2t_dataset_transform

    metadata = TaskMetadata(
        name="MMBrightTravelIT2TRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving reasoning-intensive technical passages in the Travel domain.",
        type="Any2AnyRetrieval",
        category="it2t",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve passages that provide the reasoning needed to answer it."
        },
        dataset={
            "path": "mteb/MMBrightTravelRetrieval",
            "revision": "b64e37c79fb9453e8ddd2e6167b783038e01d1a8",
        },
        **_COMMON_METADATA,
    )


class MMBrightTravelIT2IRetrieval(AbsTaskRetrieval):
    load_data = _it2i_load_data

    metadata = TaskMetadata(
        name="MMBrightTravelIT2IRetrieval",
        description="MM-BRIGHT text-and-image queries retrieving relevant technical images in the Travel domain.",
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["text", "image"],
        task_subtypes=["Reasoning as Retrieval", "Image Text Retrieval"],
        prompt={
            "query": "Given a technical question and its images, retrieve images that provide relevant visual evidence."
        },
        dataset={
            "path": "mteb/MMBrightTravelRetrieval",
            "revision": "b64e37c79fb9453e8ddd2e6167b783038e01d1a8",
        },
        **_COMMON_METADATA,
    )
