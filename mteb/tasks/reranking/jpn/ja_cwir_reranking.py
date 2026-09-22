from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "test"


class JaCWIRReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JaCWIRReranking",
        description=(
            "JaCWIR is a small-scale Japanese information retrieval evaluation dataset consisting of "
            "5000 question texts and approximately 500k web page titles and web page introductions or summaries "
            "(meta descriptions, etc.). The question texts are created based on one of the 500k web pages, "
            "and that data is used as a positive example for the question text."
        ),
        reference="https://huggingface.co/datasets/hotchpotch/JaCWIR",
        dataset={
            "path": "mteb/JaCWIRReranking",
            "revision": "48d6b0851fb5ce83b648eb9d3689cf56a2e6d5b1",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["jpn-Jpan"],
        main_score="map_at_1000",
        date=("2020-01-01", "2024-12-31"),
        domains=["Web", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
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
