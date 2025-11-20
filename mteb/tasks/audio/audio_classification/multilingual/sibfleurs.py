from mteb.abstasks.audio.abs_task_audio_classification import AbsTaskAudioClassification
from mteb.abstasks.task_metadata import TaskMetadata

EVAL_LANGS_MAP = {
    "afr_Latn": ["afr-Latn"],  # Afrikaans
    "amh_Ethi": ["amh-Ethi"],  # Amharic
    "arb_Arab": ["arb-Arab"],  # Arabic
    "asm_Beng": ["asm-Beng"],  # Assamese
    "ast_Latn": ["ast-Latn"],  # Asturian
    "azj_Latn": ["azj-Latn"],  # South Azerbaijani
    "bel_Cyrl": ["bel-Cyrl"],  # Belarusian
    "bul_Cyrl": ["bul-Cyrl"],  # Bulgarian
    "ben_Beng": ["ben-Beng"],  # Bengali
    "bos_Latn": ["bos-Latn"],  # Bosnian
    "cat_Latn": ["cat-Latn"],  # Catalan
    "ceb_Latn": ["ceb-Latn"],  # Cebuano
    "ckb_Arab": ["ckb-Arab"],  # Central Kurdish
    "zho_Hans": ["zho-Hans"],  # Chinese (Simplified)
    "ces_Latn": ["ces-Latn"],  # Czech
    "cym_Latn": ["cym-Latn"],  # Welsh
    "dan_Latn": ["dan-Latn"],  # Danish
    "deu_Latn": ["deu-Latn"],  # German
    "ell_Grek": ["ell-Grek"],  # Greek
    "eng_Latn": ["eng-Latn"],  # English
    "spa_Latn": ["spa-Latn"],  # Spanish
    "est_Latn": ["est-Latn"],  # Estonian
    "pes_Arab": ["pes-Arab"],  # Persian
    "fin_Latn": ["fin-Latn"],  # Finnish
    "tgl_Latn": ["tgl-Latn"],  # Tagalog
    "fra_Latn": ["fra-Latn"],  # French
    "gle_Latn": ["gle-Latn"],  # Irish
    "glg_Latn": ["glg-Latn"],  # Galician
    "guj_Gujr": ["guj-Gujr"],  # Gujarati
    "hau_Latn": ["hau-Latn"],  # Hausa
    "heb_Hebr": ["heb-Hebr"],  # Hebrew
    "hin_Deva": ["hin-Deva"],  # Hindi
    "hrv_Latn": ["hrv-Latn"],  # Croatian
    "hun_Latn": ["hun-Latn"],  # Hungarian
    "hye_Armn": ["hye-Armn"],  # Armenian
    "ind_Latn": ["ind-Latn"],  # Indonesian
    "ibo_Latn": ["ibo-Latn"],  # Igbo
    "isl_Latn": ["isl-Latn"],  # Icelandic
    "ita_Latn": ["ita-Latn"],  # Italian
    "jpn_Jpan": ["jpn-Jpan"],  # Japanese
    "jav_Latn": ["jav-Latn"],  # Javanese
    "kat_Geor": ["kat-Geor"],  # Georgian
    "kam_Latn": ["kam-Latn"],  # Kamba
    "kea_Latn": ["kea-Latn"],  # Kabuverdianu
    "kaz_Cyrl": ["kaz-Cyrl"],  # Kazakh
    "khm_Khmr": ["khm-Khmr"],  # Khmer
    "kan_Knda": ["kan-Knda"],  # Kannada
    "kor_Hang": ["kor-Hang"],  # Korean
    "kir_Cyrl": ["kir-Cyrl"],  # Kyrgyz
    "ltz_Latn": ["ltz-Latn"],  # Luxembourgish
    "lug_Latn": ["lug-Latn"],  # Ganda
    "lin_Latn": ["lin-Latn"],  # Lingala
    "lao_Laoo": ["lao-Laoo"],  # Lao
    "lit_Latn": ["lit-Latn"],  # Lithuanian
    "luo_Latn": ["luo-Latn"],  # Luo
    "lvs_Latn": ["lvs-Latn"],  # Latvian
    "mri_Latn": ["mri-Latn"],  # Maori
    "mkd_Cyrl": ["mkd-Cyrl"],  # Macedonian
    "mal_Mlym": ["mal-Mlym"],  # Malayalam
    "khk_Cyrl": ["khk-Cyrl"],  # Mongolian (Halh)
    "mar_Deva": ["mar-Deva"],  # Marathi
    "zsm_Latn": ["zsm-Latn"],  # Malay
    "mlt_Latn": ["mlt-Latn"],  # Maltese
    "mya_Mymr": ["mya-Mymr"],  # Burmese
    "nob_Latn": ["nob-Latn"],  # Norwegian Bokmål
    "npi_Deva": ["npi-Deva"],  # Nepali
    "nld_Latn": ["nld-Latn"],  # Dutch
    "nso_Latn": ["nso-Latn"],  # Northern Sotho
    "nya_Latn": ["nya-Latn"],  # Chichewa
    "oci_Latn": ["oci-Latn"],  # Occitan
    "ory_Orya": ["ory-Orya"],  # Odia
    "pan_Guru": ["pan-Guru"],  # Punjabi
    "pol_Latn": ["pol-Latn"],  # Polish
    "pbt_Arab": ["pbt-Arab"],  # Pashto
    "por_Latn": ["por-Latn"],  # Portuguese
    "ron_Latn": ["ron-Latn"],  # Romanian
    "rus_Cyrl": ["rus-Cyrl"],  # Russian
    "snd_Arab": ["snd-Arab"],  # Sindhi
    "slk_Latn": ["slk-Latn"],  # Slovak
    "slv_Latn": ["slv-Latn"],  # Slovenian
    "sna_Latn": ["sna-Latn"],  # Shona
    "som_Latn": ["som-Latn"],  # Somali
    "srp_Cyrl": ["srp-Cyrl"],  # Serbian
    "swe_Latn": ["swe-Latn"],  # Swedish
    "swh_Latn": ["swh-Latn"],  # Swahili
    "tam_Taml": ["tam-Taml"],  # Tamil
    "tel_Telu": ["tel-Telu"],  # Telugu
    "tgk_Cyrl": ["tgk-Cyrl"],  # Tajik
    "tha_Thai": ["tha-Thai"],  # Thai
    "tur_Latn": ["tur-Latn"],  # Turkish
    "ukr_Cyrl": ["ukr-Cyrl"],  # Ukrainian
    "umb_Latn": ["umb-Latn"],  # Umbundu
    "urd_Arab": ["urd-Arab"],  # Urdu
    "uzn_Latn": ["uzn-Latn"],  # Northern Uzbek
    "vie_Latn": ["vie-Latn"],  # Vietnamese
    "wol_Latn": ["wol-Latn"],  # Wolof
    "xho_Latn": ["xho-Latn"],  # Xhosa
    "yor_Latn": ["yor-Latn"],  # Yoruba
    "zho_Hant": ["zho-Hant"],  # Chinese (Traditional)
    "zul_Latn": ["zul-Latn"],  # Zulu
    "fuv_Latn": ["fuv-Latn"],  # Fulfulde
    "gaz_Latn": ["gaz-Latn"],  # Lower Sorbian
}


class SIBFLEURSMultilingualClassification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="SIBFLEURS",
        description="Topic Classification for multilingual audio dataset. This dataset is a stratified and downsampled subset of the SIBFLEURS dataset, which is a collection of 1000+ hours of audio data in 100+ languages.",
        reference="https://huggingface.co/datasets/WueNLP/sib-fleurs",
        dataset={
            "path": "mteb/sib-fleurs-multilingual-mini",
            "revision": "186b61175fbd77059f769b6bf1110d449a2ff311",
        },
        type="AudioMultilabelClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=EVAL_LANGS_MAP,
        main_score="accuracy",
        date=(
            "2024-12-09",
            "2024-12-13",
        ),
        domains=[
            "Encyclopaedic"
        ],  # original FLORES-101 dataset is read-out wikipedia corpus
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{schmidt2025fleursslumassivelymultilingualbenchmark,
  archiveprefix = {arXiv},
  author = {Fabian David Schmidt and Ivan Vulić and Goran Glavaš and David Ifeoluwa Adelani},
  eprint = {2501.06117},
  primaryclass = {cs.CL},
  title = {Fleurs-SLU: A Massively Multilingual Benchmark for Spoken Language Understanding},
  url = {https://arxiv.org/abs/2501.06117},
  year = {2025},
}
""",
    )

    audio_column_name: str = "audio"
    label_column_name: str = "category"
    samples_per_label: int = 10
    is_cross_validation: bool = True
