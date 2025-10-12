from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class IFIRScifact(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="IFIRScifact",
        dataset={
            "path": "if-ir/scifact_open",
            "revision": "1690191",
        },
        description="Benchmark IFIR scifact_open subset within instruction following abilities. The instructions in this dataset simulate nuanced queries from students or researchers to retrieve relevant science literature. ",
        reference="https://arxiv.org/abs/2503.04644",
        type="InstructionRetrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_20",
        date=["2024-05-01", "2024-10-16"],
        domains=["Written", "Academic"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{song2025ifir,
  author = {Song, Tingyu and Gan, Guo and Shang, Mingsheng and Zhao, Yilun},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages = {10186--10204},
  title = {IFIR: A Comprehensive Benchmark for Evaluating Instruction-Following in Expert-Domain Information Retrieval},
  year = {2025},
}
""",
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        scores_dict = {"level_1": [], "level_2": [], "level_3": []}
        metric = "ndcg_cut_20"
        for k, v in scores.items():
            if "v1" in k:
                scores_dict["level_1"].append(v[metric])
            elif "v2" in k:
                scores_dict["level_2"].append(v[metric])
            elif "v3" in k:
                scores_dict["level_3"].append(v[metric])
        return {
            "level_scores": {
                "level_1": sum(scores_dict["level_1"]) / len(scores_dict["level_1"]),
                "level_2": sum(scores_dict["level_2"]) / len(scores_dict["level_2"]),
                "level_3": sum(scores_dict["level_3"]) / len(scores_dict["level_3"]),
            }
        }
