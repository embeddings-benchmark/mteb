from __future__ import annotations

from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

# Consider only those language pairs with at least 250 samples
_LANGUAGES = {
    # Afro-Asiatic, Indo-European (Iranian)
    "ara-fas": ["ara-Arab", "fas-Arab"],  # Samples: 609
    "ara-heb": ["ara-Arab", "heb-Hebr"],  # Samples: 978
    #
    # Austroasiatic, Japonic, Koreanic, Sino-Tibetan
    "jpn-kor": ["jpn-Jpan", "kor-Kore"],  # Samples: 4,820
    "jpn-vie": ["jpn-Jpan", "vie-Latn"],  # Samples: 1,356
    "jpn-zho": ["jpn-Jpan", "zho-Hans"],  # Samples: 1,728
    "kor-vie": ["kor-Kore", "vie-Latn"],  # Samples: 1,386
    "kor-zho": ["kor-Kore", "zho-Hans"],  # Samples: 1,087
    "vie-zho": ["vie-Latn", "zho-Hans"],  # Samples: 646
    #
    # Austronesian, Kra-Dai
    "ind-msa": ["ind-Latn", "msa-Latn"],  # Samples: 455
    "ind-tgl": ["ind-Latn", "tgl-Latn"],  # Samples: 378
    "ind-tha": ["ind-Latn", "tha-Thai"],  # Samples: 1,258
    #
    # Caucasian, Indo-European (Baltic), Indo-European (Slavic)
    "bul-ces": ["bul-Cyrl", "ces-Latn"],  # Samples: 1,485
    "bul-lav": ["bul-Cyrl", "lav-Latn"],  # Samples: 710
    "bul-lit": ["bul-Cyrl", "lit-Latn"],  # Samples: 803
    "bul-pol": ["bul-Cyrl", "pol-Latn"],  # Samples: 1,635
    "bul-rus": ["bul-Cyrl", "rus-Cyrl"],  # Samples: 1,476
    "bul-slk": ["bul-Cyrl", "slk-Latn"],  # Samples: 1,154
    "bul-slv": ["bul-Cyrl", "slv-Latn"],  # Samples: 1,034
    "bul-srp": ["bul-Cyrl", "srp-Cyrl"],  # Samples: 296
    "bul-ukr": ["bul-Cyrl", "ukr-Cyrl"],  # Samples: 1,074
    "ces-lav": ["ces-Latn", "lav-Latn"],  # Samples: 875
    "ces-lit": ["ces-Latn", "lit-Latn"],  # Samples: 1,002
    "ces-pol": ["ces-Latn", "pol-Latn"],  # Samples: 3,367
    "ces-rus": ["ces-Latn", "rus-Cyrl"],  # Samples: 2,144
    "ces-slk": ["ces-Latn", "slk-Latn"],  # Samples: 2,551
    "ces-slv": ["ces-Latn", "slv-Latn"],  # Samples: 1,370
    "ces-srp": ["ces-Latn", "srp-Cyrl"],  # Samples: 362
    "ces-ukr": ["ces-Latn", "ukr-Cyrl"],  # Samples: 1,285
    "hrv-slk": ["hrv-Latn", "slk-Latn"],  # Samples: 313
    "kat-rus": ["kat-Geor", "rus-Cyrl"],  # Samples: 262
    "lav-lit": ["lav-Latn", "lit-Latn"],  # Samples: 1,061
    "lav-pol": ["lav-Latn", "pol-Latn"],  # Samples: 951
    "lav-rus": ["lav-Latn", "rus-Cyrl"],  # Samples: 1,412
    "lav-slk": ["lav-Latn", "slk-Latn"],  # Samples: 789
    "lav-slv": ["lav-Latn", "slv-Latn"],  # Samples: 518
    "lav-ukr": ["lav-Latn", "ukr-Cyrl"],  # Samples: 579
    "lit-pol": ["lit-Latn", "pol-Latn"],  # Samples: 1,026
    "lit-rus": ["lit-Latn", "rus-Cyrl"],  # Samples: 961
    "lit-slk": ["lit-Latn", "slk-Latn"],  # Samples: 859
    "lit-slv": ["lit-Latn", "slv-Latn"],  # Samples: 607
    "lit-ukr": ["lit-Latn", "ukr-Cyrl"],  # Samples: 639
    "pol-rus": ["pol-Latn", "rus-Cyrl"],  # Samples: 5,014
    "pol-slk": ["pol-Latn", "slk-Latn"],  # Samples: 1,918
    "pol-slv": ["pol-Latn", "slv-Latn"],  # Samples: 1,382
    "pol-srp": ["pol-Latn", "srp-Cyrl"],  # Samples: 492
    "pol-ukr": ["pol-Latn", "ukr-Cyrl"],  # Samples: 2,370
    "rus-slk": ["rus-Cyrl", "slk-Latn"],  # Samples: 1,263
    "rus-slv": ["rus-Cyrl", "slv-Latn"],  # Samples: 1,096
    "rus-srp": ["rus-Cyrl", "srp-Cyrl"],  # Samples: 455
    "rus-ukr": ["rus-Cyrl", "ukr-Cyrl"],  # Samples: 15,251
    "slk-slv": ["slk-Latn", "slv-Latn"],  # Samples: 1,259
    "slk-srp": ["slk-Latn", "srp-Cyrl"],  # Samples: 561
    "slk-ukr": ["slk-Latn", "ukr-Cyrl"],  # Samples: 944
    "slv-srp": ["slv-Latn", "srp-Cyrl"],  # Samples: 499
    "slv-ukr": ["slv-Latn", "ukr-Cyrl"],  # Samples: 733
    #
    # Indo-European (Germanic), Indo-European (Romance)
    "cat-deu": ["cat-Latn", "deu-Latn"],  # Samples: 302
    "cat-fra": ["cat-Latn", "fra-Latn"],  # Samples: 598
    "cat-ita": ["cat-Latn", "ita-Latn"],  # Samples: 418
    "cat-por": ["cat-Latn", "por-Latn"],  # Samples: 370
    "cat-spa": ["cat-Latn", "spa-Latn"],  # Samples: 2,648
    "dan-deu": ["dan-Latn", "deu-Latn"],  # Samples: 4,337
    "dan-fra": ["dan-Latn", "fra-Latn"],  # Samples: 3,802
    "dan-isl": ["dan-Latn", "isl-Latn"],  # Samples: 327
    "dan-ita": ["dan-Latn", "ita-Latn"],  # Samples: 3,818
    "dan-nld": ["dan-Latn", "nld-Latn"],  # Samples: 4,099
    "dan-nor": ["dan-Latn", "nor-Latn"],  # Samples: 2,603
    "dan-por": ["dan-Latn", "por-Latn"],  # Samples: 3,206
    "dan-ron": ["dan-Latn", "ron-Latn"],  # Samples: 2,052
    "dan-spa": ["dan-Latn", "spa-Latn"],  # Samples: 3,571
    "dan-swe": ["dan-Latn", "swe-Latn"],  # Samples: 4,268
    "deu-fra": ["deu-Latn", "fra-Latn"],  # Samples: 27,727
    "deu-isl": ["deu-Latn", "isl-Latn"],  # Samples: 294
    "deu-ita": ["deu-Latn", "ita-Latn"],  # Samples: 18,787
    "deu-nld": ["deu-Latn", "nld-Latn"],  # Samples: 14,211
    "deu-nor": ["deu-Latn", "nor-Latn"],  # Samples: 2,783
    "deu-por": ["deu-Latn", "por-Latn"],  # Samples: 11,319
    "deu-ron": ["deu-Latn", "ron-Latn"],  # Samples: 3,598
    "deu-spa": ["deu-Latn", "spa-Latn"],  # Samples: 19,739
    "deu-swe": ["deu-Latn", "swe-Latn"],  # Samples: 5,772
    "fra-isl": ["fra-Latn", "isl-Latn"],  # Samples: 347
    "fra-ita": ["fra-Latn", "ita-Latn"],  # Samples: 20,002
    "fra-nld": ["fra-Latn", "nld-Latn"],  # Samples: 14,684
    "fra-nor": ["fra-Latn", "nor-Latn"],  # Samples: 2,558
    "fra-por": ["fra-Latn", "por-Latn"],  # Samples: 13,265
    "fra-ron": ["fra-Latn", "ron-Latn"],  # Samples: 3,295
    "fra-spa": ["fra-Latn", "spa-Latn"],  # Samples: 23,311
    "fra-swe": ["fra-Latn", "swe-Latn"],  # Samples: 5,006
    "isl-ita": ["isl-Latn", "ita-Latn"],  # Samples: 421
    "isl-nld": ["isl-Latn", "nld-Latn"],  # Samples: 311
    "isl-por": ["isl-Latn", "por-Latn"],  # Samples: 341
    "isl-spa": ["isl-Latn", "spa-Latn"],  # Samples: 366
    "isl-swe": ["isl-Latn", "swe-Latn"],  # Samples: 312
    "ita-nld": ["ita-Latn", "nld-Latn"],  # Samples: 9,160
    "ita-nor": ["ita-Latn", "nor-Latn"],  # Samples: 2,516
    "ita-por": ["ita-Latn", "por-Latn"],  # Samples: 10,924
    "ita-ron": ["ita-Latn", "ron-Latn"],  # Samples: 3,360
    "ita-spa": ["ita-Latn", "spa-Latn"],  # Samples: 16,534
    "ita-swe": ["ita-Latn", "swe-Latn"],  # Samples: 4,741
    "nld-nor": ["nld-Latn", "nor-Latn"],  # Samples: 2,664
    "nld-por": ["nld-Latn", "por-Latn"],  # Samples: 7,021
    "nld-ron": ["nld-Latn", "ron-Latn"],  # Samples: 2,888
    "nld-spa": ["nld-Latn", "spa-Latn"],  # Samples: 9,555
    "nld-swe": ["nld-Latn", "swe-Latn"],  # Samples: 5,072
    "nor-por": ["nor-Latn", "por-Latn"],  # Samples: 2,096
    "nor-ron": ["nor-Latn", "ron-Latn"],  # Samples: 1,412
    "nor-spa": ["nor-Latn", "spa-Latn"],  # Samples: 2,603
    "nor-swe": ["nor-Latn", "swe-Latn"],  # Samples: 3,165
    "por-ron": ["por-Latn", "ron-Latn"],  # Samples: 3,026
    "por-spa": ["por-Latn", "spa-Latn"],  # Samples: 16,084
    "por-swe": ["por-Latn", "swe-Latn"],  # Samples: 4,235
    "ron-spa": ["ron-Latn", "spa-Latn"],  # Samples: 3,375
    "ron-swe": ["ron-Latn", "swe-Latn"],  # Samples: 2,154
    "spa-swe": ["spa-Latn", "swe-Latn"],  # Samples: 4,884
    #
    # Indo-European (Indo-Aryan)
    "ben-hin": ["ben-Beng", "hin-Deva"],  # Samples: 1,174
    "ben-mar": ["ben-Beng", "mar-Deva"],  # Samples: 566
    "ben-urd": ["ben-Beng", "urd-Arab"],  # Samples: 488
    "hin-mar": ["hin-Deva", "mar-Deva"],  # Samples: 615
    "hin-urd": ["hin-Deva", "urd-Arab"],  # Samples: 545
    "mar-urd": ["mar-Deva", "urd-Arab"],  # Samples: 270
    #
    # Turkic
    "aze-kaz": ["aze-Latn", "kaz-Cyrl"],  # Samples: 412
    "aze-tur": ["aze-Latn", "tur-Latn"],  # Samples: 388
    "kaz-tur": ["kaz-Cyrl", "tur-Latn"],  # Samples: 340
    #
    # Uralic
    "est-fin": ["est-Latn", "fin-Latn"],  # Samples: 790
    "est-hun": ["est-Latn", "hun-Latn"],  # Samples: 674
    "fin-hun": ["fin-Latn", "hun-Latn"],  # Samples: 1,542
    #
    # Any2English
    "ara-eng": ["ara-Arab", "eng-Latn"],  # Samples: 5,698
    "aze-eng": ["aze-Latn", "eng-Latn"],  # Samples: 603
    "ben-eng": ["ben-Beng", "eng-Latn"],  # Samples: 1,367
    "bul-eng": ["bul-Cyrl", "eng-Latn"],  # Samples: 2,133
    "cat-eng": ["cat-Latn", "eng-Latn"],  # Samples: 1,152
    "ces-eng": ["ces-Latn", "eng-Latn"],  # Samples: 3,775
    "dan-eng": ["dan-Latn", "eng-Latn"],  # Samples: 4,512
    "deu-eng": ["deu-Latn", "eng-Latn"],  # Samples: 37,348
    "ell-eng": ["ell-Grek", "eng-Latn"],  # Samples: 2,790
    "eng-est": ["eng-Latn", "est-Latn"],  # Samples: 755
    "eng-fas": ["eng-Latn", "fas-Arab"],  # Samples: 556
    "eng-fin": ["eng-Latn", "fin-Latn"],  # Samples: 3,443
    "eng-fra": ["eng-Latn", "fra-Latn"],  # Samples: 37,208
    "eng-heb": ["eng-Latn", "heb-Hebr"],  # Samples: 882
    "eng-hin": ["eng-Latn", "hin-Deva"],  # Samples: 2,219
    "eng-hrv": ["eng-Latn", "hrv-Latn"],  # Samples: 336
    "eng-hun": ["eng-Latn", "hun-Latn"],  # Samples: 2,185
    "eng-ind": ["eng-Latn", "ind-Latn"],  # Samples: 3,454
    "eng-isl": ["eng-Latn", "isl-Latn"],  # Samples: 358
    "eng-ita": ["eng-Latn", "ita-Latn"],  # Samples: 19,661
    "eng-jpn": ["eng-Latn", "jpn-Jpan"],  # Samples: 3,807
    "eng-kaz": ["eng-Latn", "kaz-Cyrl"],  # Samples: 346
    "eng-kor": ["eng-Latn", "kor-Kore"],  # Samples: 2,558
    "eng-lav": ["eng-Latn", "lav-Latn"],  # Samples: 1,079
    "eng-lit": ["eng-Latn", "lit-Latn"],  # Samples: 1,185
    "eng-mar": ["eng-Latn", "mar-Deva"],  # Samples: 280
    "eng-msa": ["eng-Latn", "msa-Latn"],  # Samples: 469
    "eng-nld": ["eng-Latn", "nld-Latn"],  # Samples: 15,613
    "eng-nor": ["eng-Latn", "nor-Latn"],  # Samples: 2,666
    "eng-pol": ["eng-Latn", "pol-Latn"],  # Samples: 6,868
    "eng-por": ["eng-Latn", "por-Latn"],  # Samples: 12,406
    "eng-ron": ["eng-Latn", "ron-Latn"],  # Samples: 3,039
    "eng-rus": ["eng-Latn", "rus-Cyrl"],  # Samples: 9,360
    "eng-slk": ["eng-Latn", "slk-Latn"],  # Samples: 1,823
    "eng-slv": ["eng-Latn", "slv-Latn"],  # Samples: 1,450
    "eng-spa": ["eng-Latn", "spa-Latn"],  # Samples: 35,446
    "eng-srp": ["eng-Latn", "srp-Cyrl"],  # Samples: 303
    "eng-swe": ["eng-Latn", "swe-Latn"],  # Samples: 6,005
    "eng-tgl": ["eng-Latn", "tgl-Latn"],  # Samples: 551
    "eng-tha": ["eng-Latn", "tha-Thai"],  # Samples: 814
    "eng-tur": ["eng-Latn", "tur-Latn"],  # Samples: 4,606
    "eng-ukr": ["eng-Latn", "ukr-Cyrl"],  # Samples: 3,778
    "eng-urd": ["eng-Latn", "urd-Arab"],  # Samples: 268
    "eng-vie": ["eng-Latn", "vie-Latn"],  # Samples: 1,264
    "eng-zho": ["eng-Latn", "zho-Hans"],  # Samples: 4,959
}

_SPLITS = ["default"]


class WebFAQBitextMiningQuestions(AbsTaskBitextMining, MultilingualTask):
    metadata = TaskMetadata(
        name="WebFAQBitextMiningQuestions",
        description="""The WebFAQ Bitext Dataset consists of natural FAQ-style Question-Answer pairs that align across languages.
A sentence in the "WebFAQBitextMiningQuestions" task is the question originating from an aligned QA.
The dataset is sourced from FAQ pages on the web.""",
        reference="https://huggingface.co/PaDaS-Lab",
        dataset={
            "path": "PaDaS-Lab/webfaq-bitexts",
            "revision": "a1bc0e8fd36c3d5015bd64c14ca098596774784a",
        },
        type="BitextMining",
        category="s2s",
        modalities=["text"],
        eval_splits=_SPLITS,
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2022-09-01", "2024-10-01"),
        domains=["Web", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="human-translated",
        bibtex_citation=r"""
@misc{dinzinger2025webfaq,
  archiveprefix = {arXiv},
  author = {Michael Dinzinger and Laura Caspari and Kanishka Ghosh Dastidar and Jelena Mitrović and Michael Granitzer},
  eprint = {2502.20936},
  primaryclass = {cs.CL},
  title = {WebFAQ: A Multilingual Collection of Natural Q&amp;A Datasets for Dense Retrieval},
  url = {https://arxiv.org/abs/2502.20936},
  year = {2025},
}
""",
    )

    def dataset_transform(self):
        dataset = {}
        for langs in self.dataset:
            dataset[langs] = {}
            for split in _SPLITS:
                sentence1 = []
                sentence2 = []
                for document in self.dataset[langs][split]:
                    sentence1.append(document["question1"])
                    sentence2.append(document["question2"])

                dataset[langs][split] = {
                    "sentence1": sentence1,
                    "sentence2": sentence2,
                    "gold": [(i, i) for i in range(len(sentence1))],
                }
        self.dataset = dataset


class WebFAQBitextMiningQAs(AbsTaskBitextMining, MultilingualTask):
    metadata = TaskMetadata(
        name="WebFAQBitextMiningQAs",
        description="""The WebFAQ Bitext Dataset consists of natural FAQ-style Question-Answer pairs that align across languages.
A sentence in the "WebFAQBitextMiningQAs" task is a concatenation of a question and its corresponding answer.
The dataset is sourced from FAQ pages on the web.""",
        reference="https://huggingface.co/PaDaS-Lab",
        dataset={
            "path": "PaDaS-Lab/webfaq-bitexts",
            "revision": "a1bc0e8fd36c3d5015bd64c14ca098596774784a",
        },
        type="BitextMining",
        category="p2p",
        modalities=["text"],
        eval_splits=_SPLITS,
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2022-09-01", "2024-10-01"),
        domains=["Web", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="human-translated",
        bibtex_citation=r"""
@misc{dinzinger2025webfaq,
  archiveprefix = {arXiv},
  author = {Michael Dinzinger and Laura Caspari and Kanishka Ghosh Dastidar and Jelena Mitrović and Michael Granitzer},
  eprint = {2502.20936},
  primaryclass = {cs.CL},
  title = {WebFAQ: A Multilingual Collection of Natural Q&amp;A Datasets for Dense Retrieval},
  url = {https://arxiv.org/abs/2502.20936},
  year = {2025},
}
""",
    )

    def dataset_transform(self):
        dataset = {}
        for langs in self.dataset:
            dataset[langs] = {}
            for split in _SPLITS:
                sentence1 = []
                sentence2 = []
                for document in self.dataset[langs][split]:
                    sentence1.append(document["question1"] + " " + document["answer1"])
                    sentence2.append(document["question2"] + " " + document["answer2"])

                dataset[langs][split] = {
                    "sentence1": sentence1,
                    "sentence2": sentence2,
                    "gold": [(i, i) for i in range(len(sentence1))],
                }
        self.dataset = dataset
