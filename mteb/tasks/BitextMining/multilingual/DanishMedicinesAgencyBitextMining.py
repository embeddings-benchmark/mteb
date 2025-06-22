from __future__ import annotations

from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata


class DanishMedicinesAgencyBitextMining(AbsTaskBitextMining, MultilingualTask):
    metadata = TaskMetadata(
        name="DanishMedicinesAgencyBitextMining",
        dataset={
            "path": "mteb/english-danish-parallel-corpus",
            "revision": "0e45e58e9c360134cdb2bf023ad2606a27a2a086",
        },
        description="A Bilingual English-Danish parallel corpus from The Danish Medicines Agency.",
        reference="https://sprogteknologi.dk/dataset/bilingual-english-danish-parallel-corpus-from-the-danish-medicines-agency",
        type="BitextMining",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={"default": ["dan-Latn", "eng-Latn"]},
        main_score="f1",
        date=("2016-01-01", "2019-02-24"),
        domains=["Medical", "Written"],
        task_subtypes=[],
        license="https://opendefinition.org/od/2.1/en/",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{elrc_danish_medicines_agency_2018,
  author = {Rozis, Roberts},
  institution = {European Union},
  license = {Open Under-PSI},
  note = {Dataset created within the European Language Resource Coordination (ELRC) project under the Connecting Europe Facility - Automated Translation (CEF.AT) actions SMART 2014/1074 and SMART 2015/1091.},
  title = {Bilingual English-Danish Parallel Corpus from the Danish Medicines Agency},
  url = {https://sprogteknologi.dk/dataset/bilingual-english-danish-parallel-corpus-from-the-danish-medicines-agency},
  year = {2019},
}
""",
    )
