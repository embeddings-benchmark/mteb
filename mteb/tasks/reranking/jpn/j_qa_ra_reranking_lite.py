from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class JQaRARerankingLite(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JQaRARerankingLite",
        dataset={
            "path": "mteb/JQaRARerankingLite",
            "revision": "d23d3ad479f74824ed126052e810eac47e685558",
        },
        description=(
            "JQaRA (Japanese Question Answering with Retrieval Augmentation) is a reranking dataset "
            "consisting of questions from JAQKET and corpus from Japanese Wikipedia. This is the lightweight "
            "version with a reduced corpus (172,897 documents) constructed using hard negatives from "
            "5 high-performance models."
        ),
        reference="https://huggingface.co/datasets/hotchpotch/JQaRA",
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2025-01-01"),
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=["jpn-Jpan"],
        sample_creation="found",
        adapted_from=["JQaRAReranking"],
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
