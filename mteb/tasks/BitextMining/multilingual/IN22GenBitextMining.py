import datasets

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining

_LANGUAGES = [
    "asm_Beng",
    "ben_Beng",
    "brx_Deva",
    "doi_Deva",
    "eng_Latn",
    "gom_Deva",
    "guj_Gujr",
    "hin_Deva",
    "kan_Knda",
    "kas_Arab",
    "mai_Deva",
    "mal_Mlym",
    "mar_Deva",
    "mni_Mtei",
    "npi_Deva",
    "ory_Orya",
    "pan_Guru",
    "san_Deva",
    "sat_Olck",
    "snd_Deva",
    "tam_Taml",
    "tel_Telu",
    "urd_Arab",
]
_SPLIT = ["test"]


def extend_lang_pairs() -> dict[str, list[str]]:
    # add all possible language pairs
    hf_lang_subset2isolang = {}
    for x in _LANGUAGES:
        for y in _LANGUAGES:
            if x != y:
                pair = f"{x}-{y}"
                hf_lang_subset2isolang[pair] = [
                    x.replace("_", "-"),
                    y.replace("_", "-"),
                ]
    return hf_lang_subset2isolang


_LANGUAGES_MAPPING = extend_lang_pairs()


class IN22GenBitextMining(AbsTaskBitextMining):
    parallel_subsets = True

    metadata = TaskMetadata(
        name="IN22GenBitextMining",
        dataset={
            "path": "mteb/IN22GenBitextMining",
            "revision": "aed6ff7eed647cd6eead02f70efa23a4e67329dc",
        },
        description="IN22-Gen is a n-way parallel general-purpose multi-domain benchmark dataset for machine translation spanning English and 22 Indic languages.",
        reference="https://huggingface.co/datasets/ai4bharat/IN22-Gen",
        type="BitextMining",
        category="t2t",
        modalities=["text"],
        eval_splits=_SPLIT,
        eval_langs=_LANGUAGES_MAPPING,
        main_score="f1",
        date=("2022-10-01", "2023-03-01"),
        domains=[
            "Web",
            "Legal",
            "Government",
            "News",
            "Religious",
            "Non-fiction",
            "Written",
        ],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@article{gala2023indictrans,
  author = {Jay Gala and Pranjal A Chitale and A K Raghavan and Varun Gumma and Sumanth Doddapaneni and Aswanth Kumar M and Janki Atul Nawale and Anupama Sujatha and Ratish Puduppully and Vivek Raghavan and Pratyush Kumar and Mitesh M Khapra and Raj Dabre and Anoop Kunchukuttan},
  issn = {2835-8856},
  journal = {Transactions on Machine Learning Research},
  note = {},
  title = {IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages},
  url = {https://openreview.net/forum?id=vfT4YuzAYA},
  year = {2023},
}
""",
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(**self.metadata.dataset)
        self.data_loaded = True
