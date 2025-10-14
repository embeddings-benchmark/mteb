from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class JSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="JSTS",
        dataset={
            "path": "mteb/JSTS",
            "revision": "5bac629e25799df4c9c80a6a5db983d6cba9e77d",
        },
        description="Japanese Semantic Textual Similarity Benchmark dataset construct from YJ Image Captions Dataset "
        + "(Miyazaki and Shimizu, 2016) and annotated by crowdsource annotators.",
        reference="https://aclanthology.org/2022.lrec-1.317.pdf#page=2.00",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation"],
        eval_langs=["jpn-Jpan"],
        main_score="cosine_spearman",
        date=("2016-01-01", "2022-12-31"),
        domains=["Web", "Written"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kurihara-etal-2022-jglue,
  address = {Marseille, France},
  author = {Kurihara, Kentaro  and
Kawahara, Daisuke  and
Shibata, Tomohide},
  booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  editor = {Calzolari, Nicoletta  and
B{\'e}chet, Fr{\'e}d{\'e}ric  and
Blache, Philippe  and
Choukri, Khalid  and
Cieri, Christopher  and
Declerck, Thierry  and
Goggi, Sara  and
Isahara, Hitoshi  and
Maegaard, Bente  and
Mariani, Joseph  and
Mazo, H{\'e}l{\`e}ne  and
Odijk, Jan  and
Piperidis, Stelios},
  month = jun,
  pages = {2957--2966},
  publisher = {European Language Resources Association},
  title = {{JGLUE}: {J}apanese General Language Understanding Evaluation},
  url = {https://aclanthology.org/2022.lrec-1.317},
  year = {2022},
}
""",
    )

    min_score = 0
    max_score = 5
