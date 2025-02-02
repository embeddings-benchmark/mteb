from __future__ import annotations

from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.TaskMetadata import TaskMetadata

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


class IndicCrosslingualSTS(AbsTaskSTS):
    is_multilingual = True
    metadata = TaskMetadata(
        name="IndicCrosslingualSTS",
        dataset={
            "path": "mteb/IndicCrosslingualSTS",
            "revision": "f0366eb5a20087355c0e131162bbed943ba54b51",
        },
        description="This is a Semantic Textual Similarity testset between English and 12 high-resource Indic languages.",
        reference="https://huggingface.co/datasets/jaygala24/indic_sts",
        type="STS",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="cosine_spearman",
        date=("2021-04-30", "2021-06-09"),
        domains=[
            "News",
            "Non-fiction",
            "Web",
            "Spoken",
            "Government",
            "Written",
            "Spoken",
        ],
        task_subtypes=[],
        license="cc0-1.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
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
    )

    min_score = 0
    max_score = 5
