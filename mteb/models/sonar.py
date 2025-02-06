from __future__ import annotations

from mteb.model_meta import ModelMeta
from mteb.models.no_model_implementation import no_model_implementation_available

sonar_langs = [
    "ace-Arab",
    "ace-Latn",
    "acm-Arab",
    "acq-Arab",
    "aeb-Arab",
    "afr-Latn",
    "ajp-Arab",
    "aka-Latn",
    "als-Latn",
    "amh-Ethi",
    "apc-Arab",
    "arb-Arab",
    "arb-Latn",
    "ars-Arab",
    "ary-Arab",
    "arz-Arab",
    "asm-Beng",
    "ast-Latn",
    "awa-Deva",
    "ayr-Latn",
    "azb-Arab",
    "azj-Latn",
    "bak-Cyrl",
    "bam-Latn",
    "ban-Latn",
    "bel-Cyrl",
    "bem-Latn",
    "ben-Beng",
    "bho-Deva",
    "bjn-Arab",
    "bjn-Latn",
    "bod-Tibt",
    "bos-Latn",
    "bug-Latn",
    "bul-Cyrl",
    "cat-Latn",
    "ceb-Latn",
    "ces-Latn",
    "cjk-Latn",
    "ckb-Arab",
    "crh-Latn",
    "cym-Latn",
    "dan-Latn",
    "deu-Latn",
    "dik-Latn",
    "dyu-Latn",
    "dzo-Tibt",
    "ell-Grek",
    "eng-Latn",
    "epo-Latn",
    "est-Latn",
    "eus-Latn",
    "ewe-Latn",
    "fao-Latn",
    "fij-Latn",
    "fin-Latn",
    "fon-Latn",
    "fra-Latn",
    "fur-Latn",
    "fuv-Latn",
    "gaz-Latn",
    "gla-Latn",
    "gle-Latn",
    "glg-Latn",
    "grn-Latn",
    "guj-Gujr",
    "hat-Latn",
    "hau-Latn",
    "heb-Hebr",
    "hin-Deva",
    "hne-Deva",
    "hrv-Latn",
    "hun-Latn",
    "hye-Armn",
    "ibo-Latn",
    "ilo-Latn",
    "ind-Latn",
    "isl-Latn",
    "ita-Latn",
    "jav-Latn",
    "jpn-Jpan",
    "kab-Latn",
    "kac-Latn",
    "kam-Latn",
    "kan-Knda",
    "kas-Arab",
    "kas-Deva",
    "kat-Geor",
    "kaz-Cyrl",
    "kbp-Latn",
    "kea-Latn",
    "khk-Cyrl",
    "khm-Khmr",
    "kik-Latn",
    "kin-Latn",
    "kir-Cyrl",
    "kmb-Latn",
    "kmr-Latn",
    "knc-Arab",
    "knc-Latn",
    "kon-Latn",
    "kor-Hang",
    "lao-Laoo",
    "lij-Latn",
    "lim-Latn",
    "lin-Latn",
    "lit-Latn",
    "lmo-Latn",
    "ltg-Latn",
    "ltz-Latn",
    "lua-Latn",
    "lug-Latn",
    "luo-Latn",
    "lus-Latn",
    "lvs-Latn",
    "mag-Deva",
    "mai-Deva",
    "mal-Mlym",
    "mar-Deva",
    "min-Arab",
    "min-Latn",
    "mkd-Cyrl",
    "mlt-Latn",
    "mni-Beng",
    "mos-Latn",
    "mri-Latn",
    "mya-Mymr",
    "nld-Latn",
    "nno-Latn",
    "nob-Latn",
    "npi-Deva",
    "nso-Latn",
    "nus-Latn",
    "nya-Latn",
    "oci-Latn",
    "ory-Orya",
    "pag-Latn",
    "pan-Guru",
    "pap-Latn",
    "pbt-Arab",
    "pes-Arab",
    "plt-Latn",
    "pol-Latn",
    "por-Latn",
    "prs-Arab",
    "quy-Latn",
    "ron-Latn",
    "run-Latn",
    "rus-Cyrl",
    "sag-Latn",
    "san-Deva",
    "sat-Olck",
    "scn-Latn",
    "shn-Mymr",
    "sin-Sinh",
    "slk-Latn",
    "slv-Latn",
    "smo-Latn",
    "sna-Latn",
    "snd-Arab",
    "som-Latn",
    "sot-Latn",
    "spa-Latn",
    "srd-Latn",
    "srp-Cyrl",
    "ssw-Latn",
    "sun-Latn",
    "swe-Latn",
    "swh-Latn",
    "szl-Latn",
    "tam-Taml",
    "taq-Latn",
    "taq-Tfng",
    "tat-Cyrl",
    "tel-Telu",
    "tgk-Cyrl",
    "tgl-Latn",
    "tha-Thai",
    "tir-Ethi",
    "tpi-Latn",
    "tsn-Latn",
    "tso-Latn",
    "tuk-Latn",
    "tum-Latn",
    "tur-Latn",
    "twi-Latn",
    "tzm-Tfng",
    "uig-Arab",
    "ukr-Cyrl",
    "umb-Latn",
    "urd-Arab",
    "uzn-Latn",
    "vec-Latn",
    "vie-Latn",
    "war-Latn",
    "wol-Latn",
    "xho-Latn",
    "ydd-Hebr",
    "yor-Latn",
    "yue-Hant",
    "zho-Hans",
    "zho-Hant",
    "zsm-Latn",
    "zul-Latn",
]

sonar = ModelMeta(
    loader=no_model_implementation_available,
    name="facebook/SONAR",
    languages=sonar_langs,
    open_weights=True,
    use_instructions=False,  # it does take a language code as input
    revision="a551c586dcf4a49c8fd847de369412d556a7f2f2",
    release_date="2021-05-21",
    n_parameters=None,  # it is really multiple models so not sure how to calculate this
    max_tokens=None,  # couldn't find this
    embed_dim=1024,
    license="mit",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    reference="https://ai.meta.com/research/publications/sonar-sentence-level-multimodal-and-language-agnostic-representations/",
    training_datasets={
        # "FloresBitextMining": ["train"], # I believe it only used for evaluation
        # "IndicGenBenchFloresBitextMining": ["train"], # extension of Flores so I would say not trained on
    },
    public_training_code="https://github.com/facebookresearch/SONAR",
    public_training_data=None,  # couldn't find this
)
