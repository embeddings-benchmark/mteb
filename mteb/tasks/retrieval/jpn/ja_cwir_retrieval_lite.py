from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class JaCWIRRetrievalLite(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JaCWIRRetrievalLite",
        dataset={
            "path": "mteb/JaCWIRRetrievalLite",
            "revision": "79472b360242cf2692e24a6d9999ef50d350d672",
        },
        description=(
            "JaCWIR (Japanese Casual Web IR) is a dataset consisting of questions and webpage meta descriptions "
            "collected from Hatena Bookmark. This is the lightweight version with a reduced corpus "
            "(302,638 documents) constructed using hard negatives from 5 high-performance models."
        ),
        reference="https://huggingface.co/datasets/hotchpotch/JaCWIR",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2025-01-01"),
        domains=["Web", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        adapted_from=["JaCWIRRetrieval"],
        bibtex_citation=r"""
@inproceedings{li-etal-2026-jmteb,
  address = {Palma de Mallorca, Spain},
  author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide and Kawahara, Daisuke},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference},
  doi = {10.63317/5ouzpv2f2f6k},
  month = may,
  pages = {7423--7434},
  publisher = {ELRA Language Resource Association},
  title = {{JMTEB} and {JMTEB}-lite: {J}apanese Massive Text Embedding Benchmark and Its Lightweight Version},
  url = {https://aclanthology.org/2026.lrec-1.588/},
  year = {2026},
}
""",
    )
