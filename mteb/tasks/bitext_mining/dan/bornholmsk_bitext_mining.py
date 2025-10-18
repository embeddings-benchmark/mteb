from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining


class BornholmBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="BornholmBitextMining",
        dataset={
            "path": "mteb/BornholmBitextMining",
            "revision": "5b02048bd75e79275aa91a1fce6cdfd3f4a391cb",
        },
        description="Danish Bornholmsk Parallel Corpus. Bornholmsk is a Danish dialect spoken on the island of Bornholm, Denmark. Historically it is a part of east Danish which was also spoken in Scania and Halland, Sweden.",
        reference="https://aclanthology.org/W19-6138/",
        type="BitextMining",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["dan-Latn"],
        main_score="f1",
        date=("2019-01-01", "2019-12-31"),
        domains=["Web", "Social", "Fiction", "Written"],
        license="cc-by-4.0",
        task_subtypes=["Dialect pairing"],
        annotations_creators="expert-annotated",
        dialect=["da-dan-bornholm"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{derczynskiBornholmskNaturalLanguage2019,
  author = {Derczynski, Leon and Kjeldsen, Alex Speed},
  booktitle = {Proceedings of the Nordic Conference of Computational Linguistics (2019)},
  date = {2019},
  file = {Available Version (via Google Scholar):/Users/au554730/Zotero/storage/FBQ73ZYN/Derczynski and Kjeldsen - 2019 - Bornholmsk natural language processing Resources .pdf:application/pdf},
  pages = {338--344},
  publisher = {Linköping University Electronic Press},
  shorttitle = {Bornholmsk natural language processing},
  title = {Bornholmsk natural language processing: Resources and tools},
  url = {https://pure.itu.dk/ws/files/84551091/W19_6138.pdf},
  urldate = {2024-04-24},
}
""",
        prompt="Retrieve parallel sentences.",
    )
