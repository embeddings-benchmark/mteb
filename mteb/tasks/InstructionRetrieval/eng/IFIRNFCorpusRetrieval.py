from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class IFIRNFCorpus(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="IFIRNFCorpus",
        dataset={
            "path": "if-ir/nfcorpus",
            "revision": "3520715",
        },
        description="Benchmark IFIR nfcorpus subset within instruction following abilities. The instructions in this dataset simulate nuanced queries from students or researchers to retrieve relevant science literature in the medical and biological domains. ",
        reference="https://arxiv.org/abs/2503.04644",
        type="InstructionRetrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_20",
        date=["2024-05-01", "2024-10-16"],
        domains=["Medical", "Academic", "Written"],
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
