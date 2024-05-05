from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskSTS, MultilingualTask

_LANGUAGES = {
    "en-as": ["eng-Latn", "asm-Beng"],
    "en-bn": ["eng-Latn", "ben-Beng"],
    "en-gu": ["eng-Latn", "guj-Gujr"],
    "en-hi": ["eng-Latn", "hin-Deva"],
    "en-kn": ["eng-Latn", "kan-Knda"],
    "en-ml": ["eng-Latn", "mal-Mlym"],
    "en-mr": ["eng-Latn", "mar-Deva"],
    "en-or": ["eng-Latn", "ory-Orya"],
    "en-pa": ["eng-Latn", "pan-Guru"],
    "en-ta": ["eng-Latn", "tam-Taml"],
    "en-te": ["eng-Latn", "tel-Telu"],
    "en-ur": ["eng-Latn", "urd-Arab"],
}


def categorize_float(float_value):
    left_bound = int(float_value)
    right_bound = left_bound + 1
    if float_value - left_bound < right_bound - float_value:
        return left_bound
    else:
        return right_bound


class IndicCrosslingualSTS(AbsTaskSTS, MultilingualTask):
    fast_loading = True
    metadata = TaskMetadata(
        name="IndicCrosslingualSTS",
        dataset={
            "path": "mteb/indic_sts",
            "revision": "91b14b94dab7e75bd82ecfa8c3404c6a88fe3790",
        },
        description="This is a Semantic Textual Similarity testset between English and 12 high-resource Indic languages.",
        reference="https://huggingface.co/datasets/jaygala24/indic_sts",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="cosine_spearman",
        date=("2021-04-30", "2021-06-09"),
        form=["written", "spoken"],
        domains=["News", "Non-fiction", "Web", "Spoken", "Government"],
        task_subtypes=[],
        license="CC0",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""@article{10.1162/tacl_a_00452,
    author = {Ramesh, Gowtham and Doddapaneni, Sumanth and Bheemaraj, Aravinth and Jobanputra, Mayank and AK, Raghavan and Sharma, Ajitesh and Sahoo, Sujit and Diddee, Harshita and J, Mahalakshmi and Kakwani, Divyanshu and Kumar, Navneet and Pradeep, Aswin and Nagaraj, Srihari and Deepak, Kumar and Raghavan, Vivek and Kunchukuttan, Anoop and Kumar, Pratyush and Khapra, Mitesh Shantadevi},
    title = "{Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {10},
    pages = {145-162},
    year = {2022},
    month = {02},
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00452},
    url = {https://doi.org/10.1162/tacl\_a\_00452},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00452/1987010/tacl\_a\_00452.pdf},
}""",
        n_samples={"test": 10020},
        avg_character_length={"test": 76.22},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict

    def dataset_transform(self) -> None:
        # Convert to standard format
        for lang in self.langs:
            self.dataset[lang] = self.dataset[lang].rename_columns(
                {"english_sentence": "sentence1", "indic_sentence": "sentence2"}
            )
            self.dataset[lang] = (
                self.dataset[lang]
                .map(lambda x: {"label": round(x["score"])})
                .class_encode_column("label")
            )
            self.dataset[lang]["test"] = self.dataset[lang]["test"].train_test_split(
                test_size=256,
                seed=self.seed,
                stratify_by_column="label",
            )["test"]
