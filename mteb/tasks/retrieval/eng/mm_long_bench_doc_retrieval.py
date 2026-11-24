from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MMLongBenchDocRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MMLongBenchDocRetrieval",
        description="MMLongBench-Doc is a long-context, multimodal document understanding benchmark built from lengthy PDFs with rich layouts and content including text, tables, charts, and images. This retrieval adaptation asks models to retrieve pages from the source document for a question, assigning higher graded relevance to annotated evidence pages.",
        reference="https://papers.nips.cc/paper_files/paper/2024/hash/ae0e43289bffea0c1fa34633fc608e92-Abstract-Datasets_and_Benchmarks_Track.html",
        dataset={
            "path": "mteb/MMLongBenchDocRetrieval",
            "revision": "f841f9e025ea21e0fee52f9a0b2c506b9f33b743",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic", "Government", "Legal", "Financial", "Engineering"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="multiple",
        bibtex_citation=r"""
@inproceedings{ma2024mmlongbenchdocbenchmarkinglongcontextdocument,
  archiveprefix = {arXiv},
  author = {Yubo Ma and Yuhang Zang and Liangyu Chen and Meiqi Chen and Yizhu Jiao and Xinze Li and Xinyuan Lu and Ziyu Liu and Yan Ma and Xiaoyi Dong and Pan Zhang and Liangming Pan and Yu-Gang Jiang and Jiaqi Wang and Yixin Cao and Aixin Sun},
  booktitle = {NeurIPS},
  eprint = {2407.01523},
  title = {MMLONGBENCH-DOC: Benchmarking Long-context Document Understanding with Visualizations},
  url = {http://papers.nips.cc/paper_files/paper/2024/hash/ae0e43289bffea0c1fa34633fc608e92-Abstract-Datasets_and_Benchmarks_Track.html},
  year = {2024},
}
""",
        prompt={"query": "Find a document image that matches the given query."},
    )
