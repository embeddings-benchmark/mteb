from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class FinParaSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="FinParaSTS",
        dataset={
            "path": "TurkuNLP/turku_paraphrase_corpus",
            "revision": "e4428e399de70a21b8857464e76f0fe859cabe05",
            "name": "plain",
        },
        description="Finnish paraphrase-based semantic similarity corpus",
        reference="https://huggingface.co/datasets/TurkuNLP/turku_paraphrase_corpus",
        type="STS",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["fin-Latn"],
        main_score="cosine_spearman",
        date=("2017-01-01", "2021-12-31"),
        form=["written"],
        domains=["News", "Subtitles"],
        task_subtypes=None,
        license="cc-by-sa-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @inproceedings{kanerva-etal-2021-finnish,
            title = "{F}innish Paraphrase Corpus",
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
            editor = "Dobnik, Simon  and
            {\O}vrelid, Lilja",
            booktitle = "Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)",
            month = may # " 31--2 " # jun,
            year = "2021",
            address = "Reykjavik, Iceland (Online)",
            publisher = {Link{\"o}ping University Electronic Press, Sweden},
            url = "https://aclanthology.org/2021.nodalida-main.29",
            pages = "288--298",
            abstract = "In this paper, we introduce the first fully manually annotated paraphrase corpus for Finnish containing 53,572 paraphrase pairs harvested from alternative subtitles and news headings. Out of all paraphrase pairs in our corpus 98{\%} are manually classified to be paraphrases at least in their given context, if not in all contexts. Additionally, we establish a manual candidate selection method and demonstrate its feasibility in high quality paraphrase selection in terms of both cost and quality.",
        }        
        """,
        n_samples={"test": 1000, "validation": 1000},
        avg_character_length={"test": 59.3, "validation": 58.1},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 2
        metadata_dict["max_score"] = 4
        return metadata_dict

    def dataset_transform(self):
        self.dataset = self.dataset.shuffle(seed=self.seed)
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].select(range(1000))
        rename_dict = {"text1": "sentence1", "text2": "sentence2", "label": "score"}
        self.dataset = self.dataset.rename_columns(rename_dict)
        self.dataset = self.dataset.select_columns(list(rename_dict.values()))
        self.dataset = self.dataset.map(lambda x: {"score": int(x["score"][0])})
