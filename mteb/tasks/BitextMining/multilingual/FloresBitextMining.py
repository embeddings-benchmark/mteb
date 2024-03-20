import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskBitextMining, CrosslingualTask

_LANGUAGES = [
    "ace_Arab",
    "bam_Latn",
    "dzo_Tibt",
    "hin_Deva",
    "khm_Khmr",
    "mag_Deva",
    "pap_Latn",
    "sot_Latn",
    "tur_Latn",
    "ace_Latn",
    "ban_Latn",
    "ell_Grek",
    "hne_Deva",
    "kik_Latn",
    "mai_Deva",
    "pbt_Arab",
    "spa_Latn",
    "twi_Latn",
    "acm_Arab",
    "bel_Cyrl",
    "eng_Latn",
    "hrv_Latn",
    "kin_Latn",
    "mal_Mlym",
    "pes_Arab",
    "srd_Latn",
    "tzm_Tfng",
    "acq_Arab",
    "bem_Latn",
    "epo_Latn",
    "hun_Latn",
    "kir_Cyrl",
    "mar_Deva",
    "plt_Latn",
    "srp_Cyrl",
    "uig_Arab",
    "aeb_Arab",
    "ben_Beng",
    "est_Latn",
    "hye_Armn",
    "kmb_Latn",
    "min_Arab",
    "pol_Latn",
    "ssw_Latn",
    "ukr_Cyrl",
    "afr_Latn",
    "bho_Deva",
    "eus_Latn",
    "ibo_Latn",
    "kmr_Latn",
    "min_Latn",
    "por_Latn",
    "sun_Latn",
    "umb_Latn",
    "ajp_Arab",
    "bjn_Arab",
    "ewe_Latn",
    "ilo_Latn",
    "knc_Arab",
    "mkd_Cyrl",
    "prs_Arab",
    "swe_Latn",
    "urd_Arab",
    "aka_Latn",
    "bjn_Latn",
    "fao_Latn",
    "ind_Latn",
    "knc_Latn",
    "mlt_Latn",
    "quy_Latn",
    "swh_Latn",
    "uzn_Latn",
    "als_Latn",
    "bod_Tibt",
    "fij_Latn",
    "isl_Latn",
    "kon_Latn",
    "mni_Beng",
    "ron_Latn",
    "szl_Latn",
    "vec_Latn",
    "amh_Ethi",
    "bos_Latn",
    "fin_Latn",
    "ita_Latn",
    "kor_Hang",
    "mos_Latn",
    "run_Latn",
    "tam_Taml",
    "vie_Latn",
    "apc_Arab",
    "bug_Latn",
    "fon_Latn",
    "jav_Latn",
    "lao_Laoo",
    "mri_Latn",
    "rus_Cyrl",
    "taq_Latn",
    "war_Latn",
    "arb_Arab",
    "bul_Cyrl",
    "fra_Latn",
    "jpn_Jpan",
    "lij_Latn",
    "mya_Mymr",
    "sag_Latn",
    "taq_Tfng",
    "wol_Latn",
    "arb_Latn",
    "cat_Latn",
    "fur_Latn",
    "kab_Latn",
    "lim_Latn",
    "nld_Latn",
    "san_Deva",
    "tat_Cyrl",
    "xho_Latn",
    "ars_Arab",
    "ceb_Latn",
    "fuv_Latn",
    "kac_Latn",
    "lin_Latn",
    "nno_Latn",
    "sat_Olck",
    "tel_Telu",
    "ydd_Hebr",
    "ary_Arab",
    "ces_Latn",
    "gaz_Latn",
    "kam_Latn",
    "lit_Latn",
    "nob_Latn",
    "scn_Latn",
    "tgk_Cyrl",
    "yor_Latn",
    "arz_Arab",
    "cjk_Latn",
    "gla_Latn",
    "kan_Knda",
    "lmo_Latn",
    "npi_Deva",
    "shn_Mymr",
    "tgl_Latn",
    "yue_Hant",
    "asm_Beng",
    "ckb_Arab",
    "gle_Latn",
    "kas_Arab",
    "ltg_Latn",
    "nso_Latn",
    "sin_Sinh",
    "tha_Thai",
    "zho_Hans",
    "ast_Latn",
    "crh_Latn",
    "glg_Latn",
    "kas_Deva",
    "ltz_Latn",
    "nus_Latn",
    "slk_Latn",
    "tir_Ethi",
    "zho_Hant",
    "awa_Deva",
    "cym_Latn",
    "grn_Latn",
    "kat_Geor",
    "lua_Latn",
    "nya_Latn",
    "slv_Latn",
    "tpi_Latn",
    "zsm_Latn",
    "ayr_Latn",
    "dan_Latn",
    "guj_Gujr",
    "kaz_Cyrl",
    "lug_Latn",
    "oci_Latn",
    "smo_Latn",
    "tsn_Latn",
    "zul_Latn",
    "azb_Arab",
    "deu_Latn",
    "hat_Latn",
    "kbp_Latn",
    "luo_Latn",
    "ory_Orya",
    "sna_Latn",
    "tso_Latn",
    "azj_Latn",
    "dik_Latn",
    "hau_Latn",
    "kea_Latn",
    "lus_Latn",
    "pag_Latn",
    "snd_Arab",
    "tuk_Latn",
    "bak_Cyrl",
    "dyu_Latn",
    "heb_Hebr",
    "khk_Cyrl",
    "lvs_Latn",
    "pan_Guru",
    "som_Latn",
    "tum_Latn",
]
_LANGUAGES_PAIRS = []

_SPLIT = ["dev", "devtest"]


def extend_lang_pairs():
    # add all possible language pairs
    for x in _LANGUAGES:
        if "-" not in x:
            for y in _LANGUAGES:
                if "-" not in y:
                    if x != y:
                        pair = f"{x}-{y}"
                        if pair not in _LANGUAGES:
                            _LANGUAGES_PAIRS.append(pair)


extend_lang_pairs()


class FloresBitextMining(AbsTaskBitextMining, CrosslingualTask):
    metadata = TaskMetadata(
        name="FloresBitextMining",
        hf_hub_name="facebook/flores",
        description="FLORES is a benchmark dataset for machine translation between English and low-resource languages.",
        reference="https://huggingface.co/datasets/facebook/flores",
        type="BitextMining",
        category="s2s",
        eval_splits=_SPLIT,
        eval_langs=_LANGUAGES_PAIRS,
        main_score="f1",
        revision="80dc3040d19756742c9a18267ab30f54fb8e226b",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.langs:
            self.dataset[lang] = datasets.load_dataset(
                self.metadata_dict["hf_hub_name"],
                lang,
                revision=self.metadata_dict.get("revision", None),
            )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        # Convert to standard format
        for lang in self.langs:
            lang1 = lang.split("-")[0]
            lang2 = lang.split("-")[1]
            for split in _SPLIT:
                self.dataset[lang][split] = self.dataset[lang][split].rename_column(
                    "sentence_" + lang1, "sentence1"
                )
                self.dataset[lang][split] = self.dataset[lang][split].rename_column(
                    "sentence_" + lang2, "sentence2"
                )
