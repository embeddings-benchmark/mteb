from typing import Dict, List

import datasets

from mteb.abstasks import AbsTaskRetrieval, MultilingualTask, TaskMetadata

_LANGUAGES = {
    "mlqa.ar.ar": ["ara-Arab", "ara-Arab"],
    "mlqa.ar.de": ["ara-Arab", "deu-Latn"],
    "mlqa.ar.en": ["ara-Arab", "eng-Latn"],
    "mlqa.ar.es": ["ara-Arab", "spa-Latn"],
    "mlqa.ar.hi": ["ara-Arab", "hin-Deva"],
    "mlqa.ar.vi": ["ara-Arab", "vie-Latn"],
    "mlqa.ar.zh": ["ara-Arab", "zho-Hans"],
    "mlqa.de.ar": ["deu-Latn", "ara-Arab"],
    "mlqa.de.de": ["deu-Latn", "deu-Latn"],
    "mlqa.de.en": ["deu-Latn", "eng-Latn"],
    "mlqa.de.es": ["deu-Latn", "spa-Latn"],
    "mlqa.de.hi": ["deu-Latn", "hin-Deva"],
    "mlqa.de.vi": ["deu-Latn", "vie-Latn"],
    "mlqa.de.zh": ["deu-Latn", "zho-Hans"],
    "mlqa.en.ar": ["eng-Latn", "ara-Arab"],
    "mlqa.en.de": ["eng-Latn", "deu-Latn"],
    "mlqa.en.en": ["eng-Latn", "eng-Latn"],
    "mlqa.en.es": ["eng-Latn", "spa-Latn"],
    "mlqa.en.hi": ["eng-Latn", "hin-Deva"],
    "mlqa.en.vi": ["eng-Latn", "vie-Latn"],
    "mlqa.en.zh": ["eng-Latn", "zho-Hans"],
    "mlqa.es.ar": ["spa-Latn", "ara-Arab"],
    "mlqa.es.de": ["spa-Latn", "deu-Latn"],
    "mlqa.es.en": ["spa-Latn", "eng-Latn"],
    "mlqa.es.es": ["spa-Latn", "spa-Latn"],
    "mlqa.es.hi": ["spa-Latn", "hin-Deva"],
    "mlqa.es.vi": ["spa-Latn", "vie-Latn"],
    "mlqa.es.zh": ["spa-Latn", "zho-Hans"],
    "mlqa.hi.ar": ["hin-Deva", "ara-Arab"],
    "mlqa.hi.de": ["hin-Deva", "deu-Latn"],
    "mlqa.hi.en": ["hin-Deva", "eng-Latn"],
    "mlqa.hi.es": ["hin-Deva", "spa-Latn"],
    "mlqa.hi.hi": ["hin-Deva", "hin-Deva"],
    "mlqa.hi.vi": ["hin-Deva", "vie-Latn"],
    "mlqa.hi.zh": ["hin-Deva", "zho-Hans"],
    "mlqa.vi.ar": ["vie-Latn", "ara-Arab"],
    "mlqa.vi.de": ["vie-Latn", "deu-Latn"],
    "mlqa.vi.en": ["vie-Latn", "eng-Latn"],
    "mlqa.vi.es": ["vie-Latn", "spa-Latn"],
    "mlqa.vi.hi": ["vie-Latn", "hin-Deva"],
    "mlqa.vi.vi": ["vie-Latn", "vie-Latn"],
    "mlqa.vi.zh": ["vie-Latn", "zho-Hans"],
    "mlqa.zh.ar": ["zho-Hans", "ara-Arab"],
    "mlqa.zh.de": ["zho-Hans", "deu-Latn"],
    "mlqa.zh.en": ["zho-Hans", "eng-Latn"],
    "mlqa.zh.es": ["zho-Hans", "spa-Latn"],
    "mlqa.zh.hi": ["zho-Hans", "hin-Deva"],
    "mlqa.zh.vi": ["zho-Hans", "vie-Latn"],
    "mlqa.zh.zh": ["zho-Hans", "zho-Hans"],
}


def _build_lang_pair(langs: List[str]) -> str:
    """Builds a language pair separated by a dash.
    e.g., ['eng-Latn', 'deu-Latn'] -> 'eng-deu'.
    """
    return langs[0].split("-")[0] + "-" + langs[1].split("-")[0]


def extend_lang_pairs() -> Dict[str, List[str]]:
    eval_langs = {}
    for langs in _LANGUAGES.values():
        lang_pair = _build_lang_pair(langs)
        eval_langs[lang_pair] = langs
    return eval_langs


_EVAL_LANGS = extend_lang_pairs()


class MLQARetrieval(AbsTaskRetrieval, MultilingualTask):
    metadata = TaskMetadata(
        name="MLQARetrieval",
        description="""MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating cross-lingual question answering performance.
        MLQA consists of over 5K extractive QA instances (12K in English) in SQuAD format in seven languages - English, Arabic,
        German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA is highly parallel, with QA instances parallel between
        4 different languages on average.""",
        reference="https://huggingface.co/datasets/mlqa",
        dataset={
            "path": "facebook/mlqa",
            "revision": "397ed406c1a7902140303e7faf60fff35b58d285",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2019-01-01", "2020-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{lewis2019mlqa,
        title = {MLQA: Evaluating Cross-lingual Extractive Question Answering},
        author = {Lewis, Patrick and Oguz, Barlas and Rinott, Ruty and Riedel, Sebastian and Schwenk, Holger},
        journal = {arXiv preprint arXiv:1910.07475},
        year = 2019,
        eid = {arXiv: 1910.07475}
        }""",
        descriptive_stats={
            "n_samples": {"test": 158083, "validation": 15747},
            "avg_character_length": {
                "validation": {
                    "ara-ara": {
                        "average_document_length": 693.8883826879271,
                        "average_query_length": 42.321083172147,
                        "num_documents": 439,
                        "num_queries": 517,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "ara-deu": {
                        "average_document_length": 759.3882352941176,
                        "average_query_length": 55.14492753623188,
                        "num_documents": 170,
                        "num_queries": 207,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "ara-eng": {
                        "average_document_length": 693.8883826879271,
                        "average_query_length": 50.029013539651835,
                        "num_documents": 439,
                        "num_queries": 517,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "ara-spa": {
                        "average_document_length": 654.3071428571428,
                        "average_query_length": 53.68944099378882,
                        "num_documents": 140,
                        "num_queries": 161,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "ara-hin": {
                        "average_document_length": 626.5935483870968,
                        "average_query_length": 51.956989247311824,
                        "num_documents": 155,
                        "num_queries": 186,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "ara-vie": {
                        "average_document_length": 804.6216216216217,
                        "average_query_length": 49.57055214723926,
                        "num_documents": 148,
                        "num_queries": 163,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "ara-zho": {
                        "average_document_length": 787.3161290322581,
                        "average_query_length": 15.617021276595745,
                        "num_documents": 155,
                        "num_queries": 188,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "deu-ara": {
                        "average_document_length": 702.1675977653631,
                        "average_query_length": 43.06280193236715,
                        "num_documents": 179,
                        "num_queries": 207,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "deu-deu": {
                        "average_document_length": 721.405701754386,
                        "average_query_length": 52.572265625,
                        "num_documents": 456,
                        "num_queries": 512,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "deu-eng": {
                        "average_document_length": 721.405701754386,
                        "average_query_length": 48.33984375,
                        "num_documents": 456,
                        "num_queries": 512,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "deu-spa": {
                        "average_document_length": 677.2762430939226,
                        "average_query_length": 50.60204081632653,
                        "num_documents": 181,
                        "num_queries": 196,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "deu-hin": {
                        "average_document_length": 685.917808219178,
                        "average_query_length": 47.01840490797546,
                        "num_documents": 146,
                        "num_queries": 163,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "deu-vie": {
                        "average_document_length": 921.6196319018405,
                        "average_query_length": 46.81868131868132,
                        "num_documents": 163,
                        "num_queries": 182,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "deu-zho": {
                        "average_document_length": 736.6347305389221,
                        "average_query_length": 14.936842105263159,
                        "num_documents": 167,
                        "num_queries": 190,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "eng-ara": {
                        "average_document_length": 979.3447488584475,
                        "average_query_length": 42.321083172147,
                        "num_documents": 438,
                        "num_queries": 517,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "eng-deu": {
                        "average_document_length": 947.3109619686801,
                        "average_query_length": 52.572265625,
                        "num_documents": 447,
                        "num_queries": 512,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "eng-eng": {
                        "average_document_length": 940.2842535787321,
                        "average_query_length": 49.01480836236934,
                        "num_documents": 978,
                        "num_queries": 1148,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "eng-spa": {
                        "average_document_length": 904.3166287015945,
                        "average_query_length": 52.146,
                        "num_documents": 439,
                        "num_queries": 500,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "eng-hin": {
                        "average_document_length": 926.9621749408983,
                        "average_query_length": 49.3905325443787,
                        "num_documents": 423,
                        "num_queries": 507,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "eng-vie": {
                        "average_document_length": 1011.8296460176991,
                        "average_query_length": 48.082191780821915,
                        "num_documents": 452,
                        "num_queries": 511,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "eng-zho": {
                        "average_document_length": 1001.5046511627907,
                        "average_query_length": 15.39484126984127,
                        "num_documents": 430,
                        "num_queries": 504,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "spa-ara": {
                        "average_document_length": 674.3586206896551,
                        "average_query_length": 41.36024844720497,
                        "num_documents": 145,
                        "num_queries": 161,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "spa-deu": {
                        "average_document_length": 544.0489130434783,
                        "average_query_length": 51.86734693877551,
                        "num_documents": 184,
                        "num_queries": 196,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "spa-eng": {
                        "average_document_length": 641.8215859030837,
                        "average_query_length": 49.156,
                        "num_documents": 454,
                        "num_queries": 500,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "spa-spa": {
                        "average_document_length": 641.8215859030837,
                        "average_query_length": 52.146,
                        "num_documents": 454,
                        "num_queries": 500,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "spa-hin": {
                        "average_document_length": 703.3212121212122,
                        "average_query_length": 48.080213903743314,
                        "num_documents": 165,
                        "num_queries": 187,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "spa-vie": {
                        "average_document_length": 737.8579545454545,
                        "average_query_length": 48.82539682539682,
                        "num_documents": 176,
                        "num_queries": 189,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "spa-zho": {
                        "average_document_length": 605.52,
                        "average_query_length": 15.590062111801242,
                        "num_documents": 150,
                        "num_queries": 161,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "hin-ara": {
                        "average_document_length": 670.0394736842105,
                        "average_query_length": 43.623655913978496,
                        "num_documents": 152,
                        "num_queries": 186,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "hin-deu": {
                        "average_document_length": 596.9718309859155,
                        "average_query_length": 51.41717791411043,
                        "num_documents": 142,
                        "num_queries": 163,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "hin-eng": {
                        "average_document_length": 691.5482352941176,
                        "average_query_length": 49.75936883629191,
                        "num_documents": 425,
                        "num_queries": 507,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "hin-spa": {
                        "average_document_length": 718.4904458598726,
                        "average_query_length": 52.75935828877005,
                        "num_documents": 157,
                        "num_queries": 187,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "hin-hin": {
                        "average_document_length": 691.5482352941176,
                        "average_query_length": 49.3905325443787,
                        "num_documents": 425,
                        "num_queries": 507,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "hin-vie": {
                        "average_document_length": 778.484076433121,
                        "average_query_length": 48.35028248587571,
                        "num_documents": 157,
                        "num_queries": 177,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "hin-zho": {
                        "average_document_length": 685.0679012345679,
                        "average_query_length": 15.97883597883598,
                        "num_documents": 162,
                        "num_queries": 189,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "vie-ara": {
                        "average_document_length": 886.6052631578947,
                        "average_query_length": 41.214723926380366,
                        "num_documents": 152,
                        "num_queries": 163,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "vie-deu": {
                        "average_document_length": 981.4534161490683,
                        "average_query_length": 51.27472527472528,
                        "num_documents": 161,
                        "num_queries": 182,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "vie-eng": {
                        "average_document_length": 892.7250554323725,
                        "average_query_length": 48.09001956947162,
                        "num_documents": 451,
                        "num_queries": 511,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "vie-spa": {
                        "average_document_length": 936.6746987951807,
                        "average_query_length": 51.851851851851855,
                        "num_documents": 166,
                        "num_queries": 189,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "vie-hin": {
                        "average_document_length": 869.0509554140127,
                        "average_query_length": 46.44632768361582,
                        "num_documents": 157,
                        "num_queries": 177,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "vie-vie": {
                        "average_document_length": 892.7250554323725,
                        "average_query_length": 48.082191780821915,
                        "num_documents": 451,
                        "num_queries": 511,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "vie-zho": {
                        "average_document_length": 960.7349397590361,
                        "average_query_length": 15.048913043478262,
                        "num_documents": 166,
                        "num_queries": 184,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "zho-ara": {
                        "average_document_length": 238.75155279503105,
                        "average_query_length": 44.34574468085106,
                        "num_documents": 161,
                        "num_queries": 188,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "zho-deu": {
                        "average_document_length": 257.109756097561,
                        "average_query_length": 53.84736842105263,
                        "num_documents": 164,
                        "num_queries": 190,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "zho-eng": {
                        "average_document_length": 246.65237020316027,
                        "average_query_length": 50.15079365079365,
                        "num_documents": 443,
                        "num_queries": 504,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "zho-spa": {
                        "average_document_length": 249.6081081081081,
                        "average_query_length": 52.857142857142854,
                        "num_documents": 148,
                        "num_queries": 161,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "zho-hin": {
                        "average_document_length": 238.5521472392638,
                        "average_query_length": 52.05291005291005,
                        "num_documents": 163,
                        "num_queries": 189,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "zho-vie": {
                        "average_document_length": 268.32142857142856,
                        "average_query_length": 49.33695652173913,
                        "num_documents": 168,
                        "num_queries": 184,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "zho-zho": {
                        "average_document_length": 246.65237020316027,
                        "average_query_length": 15.39484126984127,
                        "num_documents": 443,
                        "num_queries": 504,
                        "average_relevant_docs_per_query": 1.0,
                    },
                },
                "test": {
                    "ara-ara": {
                        "average_document_length": 698.5714593198451,
                        "average_query_length": 41.26176636039752,
                        "num_documents": 4646,
                        "num_queries": 5333,
                        "average_relevant_docs_per_query": 1.000375023438965,
                    },
                    "ara-deu": {
                        "average_document_length": 592.5728542914171,
                        "average_query_length": 51.27730582524272,
                        "num_documents": 1503,
                        "num_queries": 1648,
                        "average_relevant_docs_per_query": 1.0006067961165048,
                    },
                    "ara-eng": {
                        "average_document_length": 698.5714593198451,
                        "average_query_length": 48.556451612903224,
                        "num_documents": 4646,
                        "num_queries": 5332,
                        "average_relevant_docs_per_query": 1.000562640660165,
                    },
                    "ara-spa": {
                        "average_document_length": 713.4833239118146,
                        "average_query_length": 51.406471183013146,
                        "num_documents": 1769,
                        "num_queries": 1978,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "ara-hin": {
                        "average_document_length": 702.1388888888889,
                        "average_query_length": 48.71818678317859,
                        "num_documents": 1512,
                        "num_queries": 1831,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "ara-vie": {
                        "average_document_length": 745.4528096017458,
                        "average_query_length": 48.815828041035665,
                        "num_documents": 1833,
                        "num_queries": 2047,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "ara-zho": {
                        "average_document_length": 774.4593639575971,
                        "average_query_length": 14.985355648535565,
                        "num_documents": 1698,
                        "num_queries": 1912,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "deu-ara": {
                        "average_document_length": 719.6800267201069,
                        "average_query_length": 39.54578532443905,
                        "num_documents": 1497,
                        "num_queries": 1649,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "deu-deu": {
                        "average_document_length": 725.5304712558599,
                        "average_query_length": 51.610680257035234,
                        "num_documents": 4053,
                        "num_queries": 4513,
                        "average_relevant_docs_per_query": 1.0008863283846665,
                    },
                    "deu-eng": {
                        "average_document_length": 725.5304712558599,
                        "average_query_length": 47.07777531575449,
                        "num_documents": 4053,
                        "num_queries": 4513,
                        "average_relevant_docs_per_query": 1.0008863283846665,
                    },
                    "deu-spa": {
                        "average_document_length": 740.5414052697616,
                        "average_query_length": 50.098591549295776,
                        "num_documents": 1594,
                        "num_queries": 1775,
                        "average_relevant_docs_per_query": 1.0005633802816902,
                    },
                    "deu-hin": {
                        "average_document_length": 674.3714063714064,
                        "average_query_length": 45.146153846153844,
                        "num_documents": 1287,
                        "num_queries": 1430,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "deu-vie": {
                        "average_document_length": 760.1198945981555,
                        "average_query_length": 46.64358208955224,
                        "num_documents": 1518,
                        "num_queries": 1675,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "deu-zho": {
                        "average_document_length": 771.3367697594501,
                        "average_query_length": 14.942592592592593,
                        "num_documents": 1455,
                        "num_queries": 1620,
                        "average_relevant_docs_per_query": 1.0006172839506173,
                    },
                    "eng-ara": {
                        "average_document_length": 1008.3584455058619,
                        "average_query_length": 41.26176636039752,
                        "num_documents": 4606,
                        "num_queries": 5333,
                        "average_relevant_docs_per_query": 1.000375023438965,
                    },
                    "eng-deu": {
                        "average_document_length": 910.3226686507936,
                        "average_query_length": 51.610680257035234,
                        "num_documents": 4032,
                        "num_queries": 4513,
                        "average_relevant_docs_per_query": 1.0008863283846665,
                    },
                    "eng-eng": {
                        "average_document_length": 983.0993344090359,
                        "average_query_length": 47.960714902434816,
                        "num_documents": 9916,
                        "num_queries": 11582,
                        "average_relevant_docs_per_query": 1.000690726990157,
                    },
                    "eng-spa": {
                        "average_document_length": 967.4622376109068,
                        "average_query_length": 50.923252713768804,
                        "num_documents": 4621,
                        "num_queries": 5251,
                        "average_relevant_docs_per_query": 1.000380879832413,
                    },
                    "eng-hin": {
                        "average_document_length": 986.0465631929046,
                        "average_query_length": 47.328315703824245,
                        "num_documents": 4059,
                        "num_queries": 4916,
                        "average_relevant_docs_per_query": 1.000406834825061,
                    },
                    "eng-vie": {
                        "average_document_length": 1048.6062197940744,
                        "average_query_length": 48.094085532302095,
                        "num_documents": 4759,
                        "num_queries": 5495,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "eng-zho": {
                        "average_document_length": 1063.8536257833482,
                        "average_query_length": 15.019080996884735,
                        "num_documents": 4468,
                        "num_queries": 5136,
                        "average_relevant_docs_per_query": 1.0001947040498442,
                    },
                    "spa-ara": {
                        "average_document_length": 645.5182320441988,
                        "average_query_length": 40.78412537917088,
                        "num_documents": 1810,
                        "num_queries": 1978,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "spa-deu": {
                        "average_document_length": 586.6057810578105,
                        "average_query_length": 51.870913190529876,
                        "num_documents": 1626,
                        "num_queries": 1774,
                        "average_relevant_docs_per_query": 1.0011273957158964,
                    },
                    "spa-eng": {
                        "average_document_length": 630.6735979836169,
                        "average_query_length": 47.827907862173994,
                        "num_documents": 4761,
                        "num_queries": 5253,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "spa-spa": {
                        "average_document_length": 630.6735979836169,
                        "average_query_length": 50.923252713768804,
                        "num_documents": 4761,
                        "num_queries": 5251,
                        "average_relevant_docs_per_query": 1.000380879832413,
                    },
                    "spa-hin": {
                        "average_document_length": 613.3478260869565,
                        "average_query_length": 46.36680208937899,
                        "num_documents": 1518,
                        "num_queries": 1723,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "spa-vie": {
                        "average_document_length": 659.6179295624333,
                        "average_query_length": 48.1595639246779,
                        "num_documents": 1874,
                        "num_queries": 2018,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "spa-zho": {
                        "average_document_length": 668.6646171045277,
                        "average_query_length": 15.115562403697997,
                        "num_documents": 1789,
                        "num_queries": 1947,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "hin-ara": {
                        "average_document_length": 765.0352862849534,
                        "average_query_length": 42.04642271982523,
                        "num_documents": 1502,
                        "num_queries": 1831,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "hin-deu": {
                        "average_document_length": 719.676862745098,
                        "average_query_length": 51.002799160251925,
                        "num_documents": 1275,
                        "num_queries": 1429,
                        "average_relevant_docs_per_query": 1.000699790062981,
                    },
                    "hin-eng": {
                        "average_document_length": 760.9956086850451,
                        "average_query_length": 47.91232709519935,
                        "num_documents": 4099,
                        "num_queries": 4916,
                        "average_relevant_docs_per_query": 1.000406834825061,
                    },
                    "hin-spa": {
                        "average_document_length": 753.5010281014394,
                        "average_query_length": 50.46689895470383,
                        "num_documents": 1459,
                        "num_queries": 1722,
                        "average_relevant_docs_per_query": 1.0005807200929153,
                    },
                    "hin-hin": {
                        "average_document_length": 760.9956086850451,
                        "average_query_length": 47.328315703824245,
                        "num_documents": 4099,
                        "num_queries": 4916,
                        "average_relevant_docs_per_query": 1.000406834825061,
                    },
                    "hin-vie": {
                        "average_document_length": 789.9253822629969,
                        "average_query_length": 48.21160760143811,
                        "num_documents": 1635,
                        "num_queries": 1947,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "hin-zho": {
                        "average_document_length": 834.2057448229793,
                        "average_query_length": 15.101301641199774,
                        "num_documents": 1497,
                        "num_queries": 1767,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "vie-ara": {
                        "average_document_length": 992.2129527991218,
                        "average_query_length": 41.82462139716659,
                        "num_documents": 1822,
                        "num_queries": 2047,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "vie-deu": {
                        "average_document_length": 861.0610079575597,
                        "average_query_length": 51.58721624850657,
                        "num_documents": 1508,
                        "num_queries": 1674,
                        "average_relevant_docs_per_query": 1.0005973715651135,
                    },
                    "vie-eng": {
                        "average_document_length": 913.8633993743483,
                        "average_query_length": 48.11086837793555,
                        "num_documents": 4795,
                        "num_queries": 5493,
                        "average_relevant_docs_per_query": 1.0003640997633352,
                    },
                    "vie-spa": {
                        "average_document_length": 940.0322580645161,
                        "average_query_length": 51.13386217154189,
                        "num_documents": 1829,
                        "num_queries": 2017,
                        "average_relevant_docs_per_query": 1.0004957858205255,
                    },
                    "vie-hin": {
                        "average_document_length": 838.1713414634146,
                        "average_query_length": 47.484334874165384,
                        "num_documents": 1640,
                        "num_queries": 1947,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "vie-vie": {
                        "average_document_length": 913.8633993743483,
                        "average_query_length": 48.094085532302095,
                        "num_documents": 4795,
                        "num_queries": 5495,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "vie-zho": {
                        "average_document_length": 999.064534883721,
                        "average_query_length": 15.045805455481215,
                        "num_documents": 1720,
                        "num_queries": 1943,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "zho-ara": {
                        "average_document_length": 253.71303841676368,
                        "average_query_length": 42.04866562009419,
                        "num_documents": 1718,
                        "num_queries": 1911,
                        "average_relevant_docs_per_query": 1.000523286237572,
                    },
                    "zho-deu": {
                        "average_document_length": 241.84631147540983,
                        "average_query_length": 52.25107958050586,
                        "num_documents": 1464,
                        "num_queries": 1621,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "zho-eng": {
                        "average_document_length": 247.55609326880776,
                        "average_query_length": 48.64167478091529,
                        "num_documents": 4546,
                        "num_queries": 5135,
                        "average_relevant_docs_per_query": 1.0003894839337877,
                    },
                    "zho-spa": {
                        "average_document_length": 254.44552196235026,
                        "average_query_length": 51.90446841294299,
                        "num_documents": 1753,
                        "num_queries": 1947,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "zho-hin": {
                        "average_document_length": 229.60590163934427,
                        "average_query_length": 49.06625141562854,
                        "num_documents": 1525,
                        "num_queries": 1766,
                        "average_relevant_docs_per_query": 1.0005662514156286,
                    },
                    "zho-vie": {
                        "average_document_length": 266.1140401146132,
                        "average_query_length": 49.27328872876994,
                        "num_documents": 1745,
                        "num_queries": 1943,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "zho-zho": {
                        "average_document_length": 247.55609326880776,
                        "average_query_length": 15.019080996884735,
                        "num_documents": 4546,
                        "num_queries": 5136,
                        "average_relevant_docs_per_query": 1.0001947040498442,
                    },
                },
            },
        },
    )

    def load_data(self, **kwargs):
        """In this retrieval datasets, corpus is in lang XX and queries in lang YY."""
        if self.data_loaded:
            return

        _dataset_raw = {}
        self.queries, self.corpus, self.relevant_docs = {}, {}, {}

        for hf_subset, langs in _LANGUAGES.items():
            # Builds a language pair separated by an underscore. e.g., "ara-Arab_eng-Latn".
            # Corpus is in ara-Arab and queries in eng-Latn
            lang_pair = _build_lang_pair(langs)

            _dataset_raw[lang_pair] = datasets.load_dataset(
                name=hf_subset,
                **self.metadata_dict["dataset"],
            )

            self.queries[lang_pair] = {}
            self.corpus[lang_pair] = {}
            self.relevant_docs[lang_pair] = {}

            for eval_split in self.metadata.eval_splits:
                self.queries[lang_pair][eval_split] = {}
                self.corpus[lang_pair][eval_split] = {}
                self.relevant_docs[lang_pair][eval_split] = {}

                split_data = _dataset_raw[lang_pair][eval_split]
                query_ids = {
                    query: f"Q{i}"
                    for i, query in enumerate(set(split_data["question"]))
                }
                context_ids = {
                    text: f"C{i}" for i, text in enumerate(set(split_data["context"]))
                }

                for row in split_data:
                    query = row["question"]
                    context = row["context"]
                    query_id = query_ids[query]
                    context_id = context_ids[context]
                    self.queries[lang_pair][eval_split][query_id] = query
                    self.corpus[lang_pair][eval_split][context_id] = {
                        "title": "",
                        "text": context,
                    }
                    if query_id not in self.relevant_docs[lang_pair][eval_split]:
                        self.relevant_docs[lang_pair][eval_split][query_id] = {}
                    self.relevant_docs[lang_pair][eval_split][query_id][context_id] = 1

        self.data_loaded = True
