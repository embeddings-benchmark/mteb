from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NIGHTSI2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NIGHTSI2IRetrieval",
        description="Retrieval identical image to the given image.",
        reference="https://proceedings.neurips.cc/paper_files/paper/2023/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html",
        dataset={
            "path": "MRBench/mbeir_nights_task4",
            "revision": "c9583e052be7ad52d870c62a207a2e887ba9b8aa",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Duplicate Image Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@article{fu2024dreamsim,
  author = {Fu, Stephanie and Tamir, Netanel and Sundaram, Shobhita and Chai, Lucy and Zhang, Richard and Dekel, Tali and Isola, Phillip},
  journal = {Advances in Neural Information Processing Systems},
  title = {DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data},
  volume = {36},
  year = {2024},
}
""",
        prompt={
            "query": "Find a day-to-day image that looks similar to the provided image."
        },
    )
