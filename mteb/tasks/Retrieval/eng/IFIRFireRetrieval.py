from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class IFIRFire(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="IFIRFire",
        dataset={
            "path": "if-ir/fire",
            "revision": "28e34d5",
        },
        description="Benchmark IFIR fire subset within instruction following abilities.",
        reference="https://arxiv.org/abs/2503.04644",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_20",
        date=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@inproceedings{song2025ifir,
  title={IFIR: A Comprehensive Benchmark for Evaluating Instruction-Following in Expert-Domain Information Retrieval},
  author={Song, Tingyu and Gan, Guo and Shang, Mingsheng and Zhao, Yilun},
  booktitle={Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={10186--10204},
  year={2025}
}
""",
    )
