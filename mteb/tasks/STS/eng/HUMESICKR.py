from __future__ import annotations

from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.TaskMetadata import TaskMetadata


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
        category="s2s",
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
  abstract = {Shared and internationally recognized benchmarks are fundamental for the development of any computational system. We aim to help the research community working on compositional distributional semantic models (CDSMs) by providing SICK (Sentences Involving Compositional Knowldedge), a large size English benchmark tailored for them. SICK consists of about 10,000 English sentence pairs that include many examples of the lexical, syntactic and semantic phenomena that CDSMs are expected to account for, but do not require dealing with other aspects of existing sentential data sets (idiomatic multiword expressions, named entities, telegraphic language) that are not within the scope of CDSMs. By means of crowdsourcing techniques, each pair was annotated for two crucial semantic tasks: relatedness in meaning (with a 5-point rating scale as gold score) and entailment relation between the two elements (with three possible gold labels: entailment, contradiction, and neutral). The SICK data set was used in SemEval-2014 Task 1, and it freely available for research purposes.},
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

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
