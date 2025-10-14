from mteb.abstasks import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class HUMESICKR(AbsTaskSTS):
    metadata = TaskMetadata(
        name="HUMESICK-R",
        dataset={
            "path": "mteb/mteb-human-sickr-sts",
            "revision": "cf7172d4b730a743570f25291d04abeee086c824",
        },
        description="Human evaluation subset of Semantic Textual Similarity SICK-R dataset",
        reference="https://aclanthology.org/L14-1314/",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=None,
        domains=["Web", "Written"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@inproceedings{marelli-etal-2014-sick,
  address = {Reykjavik, Iceland},
  author = {Marelli, Marco  and
Menini, Stefano  and
Baroni, Marco  and
Bentivogli, Luisa  and
Bernardi, Raffaella  and
Zamparelli, Roberto},
  booktitle = {Proceedings of the Ninth International Conference on Language Resources and Evaluation ({LREC}'14)},
  editor = {Calzolari, Nicoletta  and
Choukri, Khalid  and
Declerck, Thierry  and
Loftsson, Hrafn  and
Maegaard, Bente  and
Mariani, Joseph  and
Moreno, Asuncion  and
Odijk, Jan  and
Piperidis, Stelios},
  month = may,
  pages = {216--223},
  publisher = {European Language Resources Association (ELRA)},
  title = {A {SICK} cure for the evaluation of compositional distributional semantic models},
  url = {http://www.lrec-conf.org/proceedings/lrec2014/pdf/363_Paper.pdf},
  year = {2014},
}
""",
        adapted_from=["SICK-R"],
    )

    min_score = 0
    max_score = 5
