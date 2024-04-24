import datasets

from mteb.abstasks import AbsTaskRetrieval, CrosslingualTask, TaskMetadata

_LANGUAGES = {
    "mlqa.ar.ar": ["ara-Arab", "ara-Arab"],
    "mlqa.ar.de": ["ara-Arab", "deu-Latn"],
    "mlqa.ar.en": ["ara-Arab", "eng-Latn"],
    "mlqa.ar.es": ["ara-Arab", "spa-Latn"],
    "mlqa.ar.hi": ["ara-Arab", "hin-Deva"],
    "mlqa.ar.vi": ["ara-Arab", "vie-Latn"],
    "mlqa.ar.zh": ["ara-Arab", "cmn-Hans"],
    "mlqa.de.ar": ["deu-Latn", "ara-Arab"],
    "mlqa.de.de": ["deu-Latn", "deu-Latn"],
    "mlqa.de.en": ["deu-Latn", "eng-Latn"],
    "mlqa.de.es": ["deu-Latn", "spa-Latn"],
    "mlqa.de.hi": ["deu-Latn", "hin-Deva"],
    "mlqa.de.vi": ["deu-Latn", "vie-Latn"],
    "mlqa.de.zh": ["deu-Latn", "cmn-Hans"],
    "mlqa.en.ar": ["eng-Latn", "ara-Arab"],
    "mlqa.en.de": ["eng-Latn", "deu-Latn"],
    "mlqa.en.en": ["eng-Latn", "eng-Latn"],
    "mlqa.en.es": ["eng-Latn", "spa-Latn"],
    "mlqa.en.hi": ["eng-Latn", "hin-Deva"],
    "mlqa.en.vi": ["eng-Latn", "vie-Latn"],
    "mlqa.en.zh": ["eng-Latn", "cmn-Hans"],
    "mlqa.es.ar": ["spa-Latn", "ara-Arab"],
    "mlqa.es.de": ["spa-Latn", "deu-Latn"],
    "mlqa.es.en": ["spa-Latn", "eng-Latn"],
    "mlqa.es.es": ["spa-Latn", "spa-Latn"],
    "mlqa.es.hi": ["spa-Latn", "hin-Deva"],
    "mlqa.es.vi": ["spa-Latn", "vie-Latn"],
    "mlqa.es.zh": ["spa-Latn", "cln-Hans"],
    "mlqa.hi.ar": ["hin-Deva", "ara-Arab"],
    "mlqa.hi.de": ["hin-Deva", "deu-Latn"],
    "mlqa.hi.en": ["hin-Deva", "eng-Latn"],
    "mlqa.hi.es": ["hin-Deva", "spa-Latn"],
    "mlqa.hi.hi": ["hin-Deva", "hin-Deva"],
    "mlqa.hi.vi": ["hin-Deva", "vie-Latn"],
    "mlqa.hi.zh": ["hin-Deva", "cmn-Hans"],
    "mlqa.vi.ar": ["vie-Latn", "ara-Arab"],
    "mlqa.vi.de": ["vie-Latn", "deu-Latn"],
    "mlqa.vi.en": ["vie-Latn", "eng-Latn"],
    "mlqa.vi.es": ["vie-Latn", "spa-Latn"],
    "mlqa.vi.hi": ["vie-Latn", "hin-Deva"],
    "mlqa.vi.vi": ["vie-Latn", "vie-Latn"],
    "mlqa.vi.zh": ["vie-Latn", "cmn-Hans"],
    "mlqa.zh.ar": ["cmn-Hans", "ara-Arab"],
    "mlqa.zh.de": ["cmn-Hans", "deu-Latn"],
    "mlqa.zh.en": ["cmn-Hans", "eng-Latn"],
    "mlqa.zh.es": ["cmn-Hans", "spa-Latn"],
    "mlqa.zh.hi": ["cmn-Hans", "hin-Deva"],
    "mlqa.zh.vi": ["cmn-Hans", "vie-Latn"],
    "mlqa.zh.zh": ["cmn-Hans", "cmn-Hans"],
}

_ISO6393_to_ISO6391 = {
    "ara-Arab": "ar",
    "deu-Latn": "de",
    "eng-Latn": "en",
    "spa-Latn": "es",
    "hin-Deva": "hi",
    "vie-Latn": "vi",
    "zho-Hans": "zh",
}


def extend_lang_pairs() -> dict[str, list[str]]:
    # add all possible language pairs
    hf_lang_subset2isolang = {}
    for x in _ISO6393_to_ISO6391.keys():
        for y in _ISO6393_to_ISO6391.keys():
            if x != y:
                pair = f"{x}_{y}"
                hf_lang_subset2isolang[pair] = [
                    x,
                    y,
                ]
    return hf_lang_subset2isolang


_EVAL_LANGS = extend_lang_pairs()


class MLQARetrieval(AbsTaskRetrieval, CrosslingualTask):
    metadata = TaskMetadata(
        name="MLQARetrieval",
        description="""MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating cross-lingual question answering performance.
        MLQA consists of over 5K extractive QA instances (12K in English) in SQuAD format in seven languages - English, Arabic,
        German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA is highly parallel, with QA instances parallel between
        4 different languages on average.""",
        reference="https://huggingface.co/datasets/mlqa",
        dataset={
            "path": "mlqa",
            "revision": "397ed406c1a7902140303e7faf60fff35b58d285",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["validation", "test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@article{lewis2019mlqa,
        title = {MLQA: Evaluating Cross-lingual Extractive Question Answering},
        author = {Lewis, Patrick and Oguz, Barlas and Rinott, Ruty and Riedel, Sebastian and Schwenk, Holger},
        journal = {arXiv preprint arXiv:1910.07475},
        year = 2019,
        eid = {arXiv: 1910.07475}
        }""",
        n_samples=None,
        avg_character_length=None,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        _dataset_raw = {}
        self.queries, self.corpus, self.relevant_docs = {}, {}, {}

        for lang in _LANGUAGES.keys():
            lang_pair = _LANGUAGES[lang][0] + "_" + _LANGUAGES[lang][1]
            _dataset_raw[lang_pair] = datasets.load_dataset(
                name=lang,
                **self.metadata_dict["dataset"],
            )
            _dataset_raw[lang_pair] = _dataset_raw[lang_pair].rename_column(
                "context", "text"
            )

            self.queries[lang_pair] = {
                eval_split: {
                    str(i): q["question"]
                    for i, q in enumerate(_dataset_raw[lang_pair][eval_split])
                }
                for eval_split in self.metadata_dict["eval_splits"]
            }

            self.corpus[lang_pair] = {
                eval_split: {
                    str(row["id"]): row for row in _dataset_raw[lang_pair][eval_split]
                }
                for eval_split in self.metadata_dict["eval_splits"]
            }

            self.relevant_docs[lang_pair] = {
                eval_split: {
                    str(i): {str(q["id"]): 1}
                    for i, q in enumerate(_dataset_raw[lang_pair][eval_split])
                }
                for eval_split in self.metadata_dict["eval_splits"]
            }

        self.data_loaded = True
