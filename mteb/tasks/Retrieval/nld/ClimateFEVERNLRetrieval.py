from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class ClimateFEVERNL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ClimateFEVER-NL",
        description="CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world "
        "claims regarding climate-change. ClimateFEVER-NL is a Dutch translation.",
        reference="https://huggingface.co/datasets/clips/beir-nl-climate-fever",
        dataset={
            "path": "clips/beir-nl-climate-fever",
            "revision": "8d4f81ded229ee2bfd63b2d30a9df5d678abb1f7",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2020-12-01", "2020-12-01"),  # best guess: based on publication date
        domains=["Encyclopaedic", "Written"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and verified",  # manually checked a small subset
        bibtex_citation=r"""
@misc{banar2024beirnlzeroshotinformationretrieval,
  archiveprefix = {arXiv},
  author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
  eprint = {2412.08329},
  primaryclass = {cs.CL},
  title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
  url = {https://arxiv.org/abs/2412.08329},
  year = {2024},
}
""",
        adapted_from=["ClimateFEVER"],
    )
