from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata

N_SAMPLES = 1000


class FinParaSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="FinParaSTS",
        dataset={
            "path": "mteb/FinParaSTS",
            "revision": "16d3834e2eb7f9faedbae0cf25df4b0962c97e71",
        },
        description="Finnish paraphrase-based semantic similarity corpus",
        reference="https://huggingface.co/datasets/TurkuNLP/turku_paraphrase_corpus",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["fin-Latn"],
        main_score="cosine_spearman",
        date=("2017-01-01", "2021-12-31"),
        domains=["News", "Subtitles", "Written"],
        task_subtypes=None,
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kanerva-etal-2021-finnish,
  address = {Reykjavik, Iceland (Online)},
  author = {Kanerva, Jenna  and
Ginter, Filip  and
Chang, Li-Hsin  and
Rastas, Iiro  and
Skantsi, Valtteri  and
Kilpel{\"a}inen, Jemina  and
Kupari, Hanna-Mari  and
Saarni, Jenna  and
Sev{\'o}n, Maija  and
Tarkka, Otto},
  booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
  editor = {Dobnik, Simon  and
{\\O}vrelid, Lilja},
  month = may # { 31--2 } # jun,
  pages = {288--298},
  publisher = {Link{\"o}ping University Electronic Press, Sweden},
  title = {{F}innish Paraphrase Corpus},
  url = {https://aclanthology.org/2021.nodalida-main.29},
  year = {2021},
}
""",
    )

    min_score = 2
    max_score = 4
