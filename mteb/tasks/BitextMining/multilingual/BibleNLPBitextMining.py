from __future__ import annotations

from typing import Any

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskBitextMining, CrosslingualTask

_LANGUAGES = [
    "aai_Latn",  # Apinayé
    "aak_Arab",  # Ankave
    "aau_Latn",  # Abau
    "aaz_Latn",  # Amarasi
    "abt_Latn",  # Ambulas
    "abx_Latn",  # Inabaknon
    "aby_Latn",  # Aneme Wake
    "acf_Latn",  # Saint Lucian Creole French
    "acr_Latn",  # Achi
    "acu_Latn",  # Achuar-Shiwiar
    "adz_Latn",  # Adzera
    "aer_Latn",  # Eastern Arrernte
    "aey_Latn",  # Amele
    "agd_Latn",  # Agarabi
    "agg_Latn",  # Angor
    "agm_Latn",  # Angaataha
    "agn_Latn",  # Agutaynen
    "agr_Latn",  # Aguaruna
    "agt_Latn",  # Central Cagayan Agta
    "agu_Latn",  # Aguacateco
    "aia_Latn",  # Arosi
    "aii_Syrc",  # Assyrian Neo-Aramaic
    "aka_Latn",  # Akan
    "ake_Latn",  # Akawaio
    "alp_Latn",  # Alune
    "alq_Latn",  # Algonquin
    "als_Latn",  # Tosk Albanian
    "aly_Latn",  # Alyawarr
    "ame_Latn",  # Yanesha'
    "amf_Latn",  # Hamer-Banna
    "amk_Latn",  # Ambai
    "amm_Latn",  # Ama (Papua New Guinea)
    "amn_Latn",  # Amanab
    "amo_Latn",  # Amo
    "amp_Latn",  # Alamblak
    "amr_Latn",  # Amarakaeri
    "amu_Latn",  # Amuzgo
    "amx_Latn",  # Anmatyerre
    "anh_Latn",  # Nend
    "anv_Latn",  # Denya
    "aoi_Latn",  # Anindilyakwa
    "aoj_Latn",  # Mufian
    "aom_Latn",  # Ömie
    "aon_Latn",  # Bumbita Arapesh
    "apb_Latn",  # Sa'a
    "ape_Latn",  # Bukiyip
    "apn_Latn",  # Apinayé
    "apr_Latn",  # Arop-Lokep
    "apu_Latn",  # Apurinã
    "apw_Latn",  # Western Apache
    "apz_Latn",  # Safeyoka
    "arb_Arab",  # Standard Arabic
    "are_Latn",  # Western Arrarnta
    "arl_Latn",  # Arabela
    "arn_Latn",  # Mapudungun
    "arp_Latn",  # Arapaho
    "asm_Beng",  # Assamese
    "aso_Latn",  # Dano
    "ata_Latn",  # Pele-Ata
    "atb_Latn",  # Zaiwa
    "atd_Latn",  # Ata Manobo
    "atg_Latn",  # Ivbie North-Okpela-Arhe
    "att_Latn",  # Pamplona Atta
    "auc_Latn",  # Waorani
    "aui_Latn",  # Anuki
    "auy_Latn",  # Awiyaana
    "avt_Latn",  # Au
    "awb_Latn",  # Awa (Papua New Guinea)
    "awk_Latn",  # Awabakal
    "awx_Latn",  # Awara
    "azb_Arab",  # South Azerbaijani
    "azg_Latn",  # San Pedro Amuzgos Amuzgo
    "azz_Latn",  # Highland Puebla Nahuatl
    "bao_Latn",  # Waimaha
    "bba_Latn",  # Baatonum
    "bbb_Latn",  # Barai
    "bbr_Latn",  # Girawa
    "bch_Latn",  # Bariai
    "bco_Latn",  # Kaluli
    "bdd_Latn",  # Bunama
    "bea_Latn",  # Beaver
    "bef_Latn",  # Benabena
    "bel_Cyrl",  # Belarusian
    "ben_Beng",  # Bengali
    "beo_Latn",  # Beami
    "beu_Latn",  # Blagar
    "bgs_Latn",  # Tagabawa
    "bgt_Latn",  # Bughotu
    "bhg_Latn",  # Binandere
    "bhl_Latn",  # Bimin
    "big_Latn",  # Biangai
    "bjk_Latn",  # Barok
    "bjp_Latn",  # Fanamaket
    "bjr_Latn",  # Binumarien
    "bjv_Latn",  # Bedjond
    "bjz_Latn",  # Baruga
    "bkd_Latn",  # Binukid
    "bki_Latn",  # Bikol
    "bkq_Latn",  # Bakairí
    "bkx_Latn",  # Baikeno
    "blw_Latn",  # Balangao
    "blz_Latn",  # Balantak
    "bmh_Latn",  # Kein
    "bmk_Latn",  # Ghayavi
    "bmr_Latn",  # Muinane
    "bmu_Latn",  # Somba-Siawari
    "bnp_Latn",  # Bola
    "boa_Latn",  # Bora
    "boj_Latn",  # Anjam
    "bon_Latn",  # Bine
    "box_Latn",  # Buamu
    "bpr_Latn",  # Koronadal Blaan
    "bps_Latn",  # Sarangani Blaan
    "bqc_Latn",  # Boko (Benin)
    "bqp_Latn",  # Busa
    "bre_Latn",  # Breton
    "bsj_Latn",  # Bangwinji
    "bsn_Latn",  # Barasana-Eduria
    "bsp_Latn",  # Baga Sitemu
    "bss_Latn",  # Akoose
    "buk_Latn",  # Bugawac
    "bus_Latn",  # Bokobaru
    "bvd_Latn",  # Baeggu
    "bvr_Latn",  # Burarra
    "bxh_Latn",  # Buhutu
    "byr_Latn",  # Baruya
    "byx_Latn",  # Qaqet
    "bzd_Latn",  # Bribri
    "bzh_Latn",  # Mapos Buang
    "bzj_Latn",  # Belize Kriol English
    "caa_Latn",  # Ch'orti'
    "cab_Latn",  # Garifuna
    "cac_Latn",  # Chuj
    "caf_Latn",  # Southern Carrier
    "cak_Latn",  # Kaqchikel
    "cao_Latn",  # Chácobo
    "cap_Latn",  # Chipaya
    "car_Latn",  # Galibi Carib
    "cav_Latn",  # Cavineña
    "cax_Latn",  # Chiquitano
    "cbc_Latn",  # Carapana
    "cbi_Latn",  # Chachi
    "cbk_Latn",  # Chavacano
    "cbr_Latn",  # Cashibo-Cacataibo
    "cbs_Latn",  # Cashinahua
    "cbt_Latn",  # Chayahuita
    "cbu_Latn",  # Candoshi-Shapra
    "cbv_Latn",  # Cacua
    "cco_Latn",  # Chamicuro
    "ceb_Latn",  # Cebuano
    "cek_Latn",  # Eastern Khumi Chin
    "ces_Latn",  # Czech
    "cgc_Latn",  # Kagayanen
    "cha_Latn",  # Chamorro
    "chd_Latn",  # Highland Oaxaca Chontal
    "chf_Latn",  # Tabasco Chontal
    "chk_Latn",  # Chuukese
    "chq_Latn",  # Quiotepec Chinantec
    "chz_Latn",  # Ozumacín Chinantec
    "cjo_Latn",  # Ashéninka Pajonal
    "cjv_Latn",  # Chuave
    "ckb_Arab",  # Central Kurdish
    "cle_Latn",  # Lealao Chinantec
    "clu_Latn",  # Caluyanun
    "cme_Latn",  # Cerma
    "cmn_Hans",  # Mandarin Chinese (Simplified)
    "cni_Latn",  # Asháninka
    "cnl_Latn",  # Lalana Chinantec
    "cnt_Latn",  # Tepetotutla Chinantec
    "cof_Latn",  # Colorado
    "con_Latn",  # Cofán
    "cop_Copt",  # Coptic
    "cot_Latn",  # Caquinte
    "cpa_Latn",  # Palantla Chinantec
    "cpb_Latn",  # Ucayali-Yurúa Ashéninka
    "cpc_Latn",  # Ajyíninka Apurucayali
    "cpu_Latn",  # Pichis Ashéninka
    "cpy_Latn",  # South Ucayali Ashéninka
    "crn_Latn",  # El Nayar Cora
    "crx_Latn",  # Carrier
    "cso_Latn",  # Sochiapam Chinantec
    "csy_Latn",  # Siyin Chin
    "cta_Latn",  # Tataltepec Chatino
    "cth_Latn",  # Thaiphum Chin
    "ctp_Latn",  # Western Highland Chatino
    "ctu_Latn",  # Chol
    "cub_Latn",  # Cubeo
    "cuc_Latn",  # Usila Chinantec
    "cui_Latn",  # Cuiba
    "cuk_Latn",  # San Blas Kuna
    "cut_Latn",  # Teutila Cuicatec
    "cux_Latn",  # Tepeuxila Cuicatec
    "cwe_Latn",  # Kwere
    "cya_Latn",  # Nopala Chatino
    "daa_Latn",  # Dangaléat
    "dad_Latn",  # Marik
    "dah_Latn",  # Gwahatike
    "dan_Latn",  # Danish
    "ded_Latn",  # Dedua
    "deu_Latn",  # German
    "dgc_Latn",  # Agta
    "dgr_Latn",  # Dogrib
    "dgz_Latn",  # Daga
    "dhg_Latn",  # Dhangu
    "dif_Latn",  # Dieri
    "dik_Latn",  # Southwestern Dinka
    "dji_Latn",  # Djinang
    "djk_Latn",  # Aukan
    "djr_Latn",  # Djambarrpuyngu
    "dob_Latn",  # Dobu
    "dop_Latn",  # Lukpa
    "dov_Latn",  # Dombe
    "dwr_Latn",  # Dawro
    "dww_Latn",  # Dawawa
    "dwy_Latn",  # Dhuwaya
    "ebk_Latn",  # Eastern Bontok
    "eko_Latn",  # Kota (Gabon)
    "emi_Latn",  # Mussau-Emira
    "emp_Latn",  # Northern Emberá
    "eng_Latn",  # English
    "enq_Latn",  # Enga
    "epo_Latn",  # Esperanto
    "eri_Latn",  # Ogea
    "ese_Latn",  # Ese Ejja
    "esk_Latn",  # Northwest Alaska Inupiatun
    "etr_Latn",  # Edolo
    "ewe_Latn",  # Ewe
    "faa_Latn",  # Fasu
    "fai_Latn",  # Faiwol
    "far_Latn",  # Fataleka
    "ffm_Latn",  # Maasina Fulfulde
    "for_Latn",  # Fore
    "fra_Latn",  # French
    "fue_Latn",  # Borgu Fulfulde
    "fuf_Latn",  # Pular
    "fuh_Latn",  # Western Niger Fulfulde
    "gah_Latn",  # Alekano
    "gai_Latn",  # Borei
    "gam_Latn",  # Kandawo
    "gaw_Latn",  # Nobonob
    "gdn_Latn",  # Umanakaina
    "gdr_Latn",  # Wipi
    "geb_Latn",  # Kire
    "gfk_Latn",  # Patpatar
    "ghs_Latn",  # Guhu-Samane
    "glk_Arab",  # Gilaki
    "gmv_Latn",  # Gamo
    "gng_Latn",  # Ngangam
    "gnn_Latn",  # Gumatj
    "gnw_Latn",  # Western Bolivian Guaraní
    "gof_Latn",  # Gofa
    "grc_Grek",  # Ancient Greek
    "gub_Latn",  # Guajajára
    "guh_Latn",  # Guahibo
    "gui_Latn",  # Eastern Bolivian Guaraní
    "guj_Gujr",  # Gujarati
    "gul_Latn",  # Sea Island Creole English
    "gum_Latn",  # Guambiano
    "gun_Latn",  # Mbyá Guaraní
    "guo_Latn",  # Guayabero
    "gup_Latn",  # Gunwinggu
    "gux_Latn",  # Gourmanchéma
    "gvc_Latn",  # Guanano
    "gvf_Latn",  # Golin
    "gvn_Latn",  # Kuku-Yalanji
    "gvs_Latn",  # Gumawana
    "gwi_Latn",  # Gwichʼin
    "gym_Latn",  # Ngäbere
    "gyr_Latn",  # Guarayu
    "hat_Latn",  # Haitian Creole
    "hau_Latn",  # Hausa
    "haw_Latn",  # Hawaiian
    "hbo_Hebr",  # Ancient Hebrew
    "hch_Latn",  # Huichol
    "heb_Hebr",  # Hebrew
    "heg_Latn",  # Helong
    "hin_Deva",  # Hindi
    "hix_Latn",  # Hoocąk
    "hla_Latn",  # Halia
    "hlt_Latn",  # Matu Chin
    "hmo_Latn",  # Hiri Motu
    "hns_Latn",  # Caribbean Hindustani
    "hop_Latn",  # Hopi
    "hot_Latn",  # Hote
    "hrv_Latn",  # Croatian
    "hto_Latn",  # Minica Huitoto
    "hub_Latn",  # Huambisa
    "hui_Latn",  # Huli
    "hun_Latn",  # Hungarian
    "hus_Latn",  # Huastec
    "huu_Latn",  # Huitoto, Murui
    "huv_Latn",  # San Mateo Del Mar Huave
    "hvn_Latn",  # Sabu
    "ian_Latn",  # Iatmul
    "ign_Latn",  # Ignaciano
    "ikk_Latn",  # Ika
    "ikw_Latn",  # Ikwere
    "ilo_Latn",  # Iloko
    "imo_Latn",  # Imbongu
    "inb_Latn",  # Inga
    "ind_Latn",  # Indonesian
    "ino_Latn",  # Inoke-Yate
    "iou_Latn",  # Tuma-Irumu
    "ipi_Latn",  # Ipili
    "isn_Latn",  # Isanzu
    "ita_Latn",  # Italian
    "iws_Latn",  # Sepik Iwam
    "ixl_Latn",  # Ixil
    "jac_Latn",  # Popti'
    "jae_Latn",  # Yabem
    "jao_Latn",  # Yanyuwa
    "jic_Latn",  # Tol
    "jid_Latn",  # Bu
    "jiv_Latn",  # Shuar
    "jni_Latn",  # Janji
    "jpn_Jpan",  # Japanese
    "jvn_Latn",  # Caribbean Javanese
    "kan_Knda",  # Kannada
    "kaq_Latn",  # Capanahua
    "kbc_Latn",  # Kadiwéu
    "kbh_Latn",  # Camsá
    "kbm_Latn",  # Iwal
    "kbq_Latn",  # Kamano
    "kdc_Latn",  # Kutu
    "kde_Latn",  # Makonde
    "kdl_Latn",  # Tsikimba
    "kek_Latn",  # Kekchí
    "ken_Latn",  # Kenyang
    "kew_Latn",  # West Kewa
    "kgf_Latn",  # Kube
    "kgk_Latn",  # Kagoma
    "kgp_Latn",  # Kaingang
    "khs_Latn",  # Kasua
    "khz_Latn",  # Keapara
    "kik_Latn",  # Kikuyu
    "kiw_Latn",  # Northeast Kiwai
    "kiz_Latn",  # Kisi
    "kje_Latn",  # Kisar
    "kjs_Latn",  # East Kewa
    "kkc_Latn",  # Odoodee
    "kkl_Latn",  # Kokota
    "klt_Latn",  # Nukna
    "klv_Latn",  # Maskelynes
    "kmg_Latn",  # Kâte
    "kmh_Latn",  # Kalam
    "kmk_Latn",  # Limos Kalinga
    "kmo_Latn",  # Kwoma
    "kms_Latn",  # Kamasau
    "kmu_Latn",  # Kanite
    "kne_Latn",  # Kankanaey
    "knf_Latn",  # Mankanya
    "knj_Latn",  # Western Kanjobal
    "knv_Latn",  # Tabo
    "kos_Latn",  # Kosraean
    "kpf_Latn",  # Komba
    "kpg_Latn",  # Kapingamarangi
    "kpj_Latn",  # Karajá
    "kpr_Latn",  # Korafe-Yegha
    "kpw_Latn",  # Kobon
    "kpx_Latn",  # Mountain Koiali
    "kqa_Latn",  # Mum
    "kqc_Latn",  # Doromu-Koki
    "kqf_Latn",  # Kakabai
    "kql_Latn",  # Kyenele
    "kqw_Latn",  # Kandas
    "ksd_Latn",  # Kuanua
    "ksj_Latn",  # Uare
    "ksr_Latn",  # Borong
    "ktm_Latn",  # Kotan
    "kto_Latn",  # Kuot
    "kud_Latn",  # 'Auhelawa
    "kue_Latn",  # Kuman
    "kup_Latn",  # Kunimaipa
    "kvg_Latn",  # Kuria
    "kvn_Latn",  # Border Kuna
    "kwd_Latn",  # Kwaio
    "kwf_Latn",  # Kwara'ae
    "kwi_Latn",  # Awa-Cuaiquer
    "kwj_Latn",  # Kwanga
    "kyc_Latn",  # Kyaka
    "kyf_Latn",  # Kaiy
    "kyg_Latn",  # Keyagana
    "kyq_Latn",  # Kenga
    "kyz_Latn",  # Kayabí
    "kze_Latn",  # Kosena
    "lac_Latn",  # Lacandon
    "lat_Latn",  # Latin
    "lbb_Latn",  # Label
    "lbk_Latn",  # Central Bontok
    "lcm_Latn",  # Tungag
    "leu_Latn",  # Kara (Papua New Guinea)
    "lex_Latn",  # Luang
    "lgl_Latn",  # Wala
    "lid_Latn",  # Nyindrou
    "lif_Deva",  # Limbu
    "lin_Latn",  # Lingala
    "lit_Latn",  # Lithuanian
    "llg_Latn",  # Lole
    "lug_Latn",  # Ganda
    "luo_Latn",  # Luo (Kenya and Tanzania)
    "lww_Latn",  # Lewo Eleng
    "maa_Latn",  # San Jerónimo Tecóatl Mazatec
    "maj_Latn",  # Jalapa De Díaz Mazatec
    "mal_Mlym",  # Malayalam
    "mam_Latn",  # Mam
    "maq_Latn",  # Chiquihuitlán Mazatec
    "mar_Deva",  # Marathi
    "mau_Latn",  # Maumere
    "mav_Latn",  # Sateré-Mawé
    "maz_Latn",  # Central Mazahua
    "mbb_Latn",  # Manobo
    "mbc_Latn",  # Macushi
    "mbh_Latn",  # Mangseng
    "mbj_Latn",  # Nadëb
    "mbl_Latn",  # Maxakalí
    "mbs_Latn",  # Sarangani Manobo
    "mbt_Latn",  # Matigsalug Manobo
    "mca_Latn",  # Maca
    "mcb_Latn",  # Macushi
    "mcd_Latn",  # Sharanahua
    "mcf_Latn",  # Matsés
    "mco_Latn",  # Mixe, Coatlán
    "mcp_Latn",  # Maka (Chad)
    "mcq_Latn",  # Ese
    "mcr_Latn",  # Menya
    "mdy_Latn",  # Male (Ethiopia)
    "med_Latn",  # Melpa
    "mee_Latn",  # Mundurukú
    "mek_Latn",  # Mekeo
    "meq_Latn",  # Merey
    "met_Latn",  # Mato
    "meu_Latn",  # Motu
    "mgc_Latn",  # Morokodo
    "mgh_Latn",  # Makhuwa-Meetto
    "mgw_Latn",  # Matumbi
    "mhl_Latn",  # Mauwake
    "mib_Latn",  # Mixtepec Mixtec
    "mic_Latn",  # Mi'kmaq
    "mie_Latn",  # Mixtec, Ocotepec
    "mig_Latn",  # San Miguel El Grande Mixtec
    "mih_Latn",  # Mixtec, Chayuco
    "mil_Latn",  # Peñoles Mixtec
    "mio_Latn",  # Mixtec, Pinotepa Nacional
    "mir_Latn",  # Mixtec, Isthmus
    "mit_Latn",  # Mixtec, Southern Puebla
    "miz_Latn",  # Mixtec, Coatzospan
    "mjc_Latn",  # Mixtec, San Juan Colorado
    "mkj_Latn",  # Mokilko
    "mkl_Latn",  # Mokole
    "mkn_Latn",  # Kupang Malay
    "mks_Latn",  # Mixtec, Silacayoapan
    "mle_Latn",  # Manambu
    "mlh_Latn",  # Mape
    "mlp_Latn",  # Bargam
    "mmo_Latn",  # Mangga Buang
    "mmx_Latn",  # Madak
    "mna_Latn",  # Mbula
    "mop_Latn",  # Mopán Maya
    "mox_Latn",  # Molima
    "mph_Latn",  # Maung
    "mpj_Latn",  # Martu Wangka
    "mpm_Latn",  # Yosondúa Mixtec
    "mpp_Latn",  # Migabac
    "mps_Latn",  # Dadibi
    "mpt_Latn",  # Mian
    "mpx_Latn",  # Misima-Panaeati
    "mqb_Latn",  # Mbuko
    "mqj_Latn",  # Mamasa
    "msb_Latn",  # Masbatenyo
    "msc_Latn",  # Sankaran Maninka
    "msk_Latn",  # Mansaka
    "msm_Latn",  # Agusan Manobo
    "msy_Latn",  # Aruamu
    "mti_Latn",  # Maiwa (Papua New Guinea)
    "mto_Latn",  # Totontepec Mixe
    "mux_Latn",  # Bo-Ung
    "muy_Latn",  # Muyang
    "mva_Latn",  # Manam
    "mvn_Latn",  # Minaveha
    "mwc_Latn",  # Are
    "mwe_Latn",  # Mwera (Chimwera)
    "mwf_Latn",  # Murrinh-Patha
    "mwp_Latn",  # Kala Lagaw Ya
    "mxb_Latn",  # Tezoatlán Mixtec
    "mxp_Latn",  # Tlahuitoltepec Mixe
    "mxq_Latn",  # Juquila Mixe
    "mxt_Latn",  # Jamiltepec Mixtec
    "mya_Latn",  # Muya
    "myk_Latn",  # Mamara Senoufo
    "myu_Latn",  # Mundurukú
    "myw_Latn",  # Muyuw
    "myy_Latn",  # Macuna
    "mzz_Latn",  # Masimasi
    "nab_Latn",  # Southern Nambikuára
    "naf_Latn",  # Nabak
    "nak_Latn",  # Nakanai
    "nas_Latn",  # Naasioi
    "nbq_Latn",  # Nggem
    "nca_Latn",  # Iyo
    "nch_Latn",  # Central Huasteca Nahuatl
    "ncj_Latn",  # Northern Puebla Nahuatl
    "ncl_Latn",  # Michoacán Nahuatl
    "ncu_Latn",  # Chumburung
    "ndg_Latn",  # Ndengereko
    "ndj_Latn",  # Ndjem
    "nfa_Latn",  # Dhao
    "ngp_Latn",  # Ngulu
    "ngu_Latn",  # Guerrero Nahuatl
    "nhe_Latn",  # Eastern Huasteca Nahuatl
    "nhg_Latn",  # Tetelcingo Nahuatl
    "nhi_Latn",  # Zacatlán-Ahuacatlán-Tepetzintla Nahuatl
    "nho_Latn",  # Takuu
    "nhr_Latn",  # Naro
    "nhu_Latn",  # Noone
    "nhw_Latn",  # Western Huasteca Nahuatl
    "nhy_Latn",  # Northern Oaxaca Nahuatl
    "nif_Latn",  # Nek
    "nii_Latn",  # Nii
    "nin_Latn",  # Ninia Yali
    "nko_Latn",  # Nkonya
    "nld_Latn",  # Dutch
    "nlg_Latn",  # Gela
    "nna_Latn",  # North Ndebele
    "nnq_Latn",  # Ngindo
    "noa_Latn",  # Woun Meu
    "nop_Latn",  # Numanggang
    "not_Latn",  # Nomatsiguenga
    "nou_Latn",  # Ewage-Notu
    "npi_Deva",  # Nepali (individual language)
    "npl_Latn",  # Southeastern Puebla Nahuatl
    "nsn_Latn",  # Nehan
    "nss_Latn",  # Nali
    "ntj_Latn",  # Ngaanyatjarra
    "ntp_Latn",  # Tepehuan
    "ntu_Latn",  # Natügu
    "nuy_Latn",  # Nunggubuyu
    "nvm_Latn",  # Namiae
    "nwi_Latn",  # Southwest Tanna
    "nya_Latn",  # Nyanja
    "nys_Latn",  # Nyungar
    "nyu_Latn",  # Nyungwe
    "obo_Latn",  # Obo Manobo
    "okv_Latn",  # Orokaiva
    "omw_Latn",  # South Tairora
    "ong_Latn",  # Olo
    "ons_Latn",  # Ono
    "ood_Latn",  # Tohono O'odham
    "opm_Latn",  # Oksapmin
    "ory_Orya",  # Odia (Oriya)
    "ote_Latn",  # Mezquital Otomi
    "otm_Latn",  # Eastern Highland Otomi
    "otn_Latn",  # Tenango Otomi
    "otq_Latn",  # Querétaro Otomi
    "ots_Latn",  # Estado de México Otomi
    "pab_Latn",  # Parecís
    "pad_Latn",  # Paumarí
    "pah_Latn",  # Tenharim
    "pan_Guru",  # Panjabi
    "pao_Latn",  # Northern Paiute
    "pes_Arab",  # Iranian Persian
    "pib_Latn",  # Yine
    "pio_Latn",  # Piapoco
    "pir_Latn",  # Piratapuyo
    "piu_Latn",  # Pintupi-Luritja
    "pjt_Latn",  # Pitjantjatjara
    "pls_Latn",  # San Marcos Tlalcoyalco Popoloca
    "plu_Latn",  # Palikur
    "pma_Latn",  # Paama
    "poe_Latn",  # San Juan Atzingo Popoloca
    "poh_Latn",  # Poqomchi'
    "poi_Latn",  # Highland Popoluca
    "pol_Latn",  # Polish
    "pon_Latn",  # Pohnpeian
    "por_Latn",  # Portuguese
    "poy_Latn",  # Pogolo
    "ppo_Latn",  # Folopa
    "prf_Latn",  # Paranan
    "pri_Latn",  # Paicî
    "ptp_Latn",  # Patep
    "ptu_Latn",  # Bambam
    "pwg_Latn",  # Gapapaiwa
    "qub_Latn",  # Huallaga Huánuco Quechua
    "quc_Latn",  # K'iche'
    "quf_Latn",  # Lambayeque Quechua
    "quh_Latn",  # South Bolivian Quechua
    "qul_Latn",  # North Bolivian Quechua
    "qup_Latn",  # Southern Pastaza Quechua
    "qvc_Latn",  # Cajamarca Quechua
    "qve_Latn",  # Eastern Apurímac Quechua
    "qvh_Latn",  # Huamalíes-Dos de Mayo Huánuco Quechua
    "qvm_Latn",  # Margos-Yarowilca-Lauricocha Quechua
    "qvn_Latn",  # North Junín Quechua
    "qvs_Latn",  # San Martín Quechua
    "qvw_Latn",  # Huaylla Wanca Quechua
    "qvz_Latn",  # Northern Pastaza Quichua
    "qwh_Latn",  # Huaylas Ancash Quechua
    "qxh_Latn",  # Panao Huánuco Quechua
    "qxn_Latn",  # Northern Conchucos Ancash Quechua
    "qxo_Latn",  # Southern Conchucos Ancash Quechua
    "rai_Latn",  # Ramoaaina
    "reg_Latn",  # Kara (Papua New Guinea)
    "rgu_Latn",  # Ringgou
    "rkb_Latn",  # Rikbaktsa
    "rmc_Latn",  # Carpathian Romani
    "rmy_Latn",  # Vlax Romani
    "ron_Latn",  # Romanian
    "roo_Latn",  # Rotokas
    "rop_Latn",  # Kriol
    "row_Latn",  # Dela-Oenale
    "rro_Latn",  # Waima
    "ruf_Latn",  # Luguru
    "rug_Latn",  # Roviana
    "rus_Cyrl",  # Russian
    "rwo_Latn",  # Rawa
    "sab_Latn",  # Buglere
    "san_Latn",  # Sanskrit
    "sbe_Latn",  # Saliba
    "sbk_Latn",  # Safwa
    "sbs_Latn",  # Subiya
    "seh_Latn",  # Sena
    "sey_Latn",  # Secoya
    "sgb_Latn",  # Ayta Mag-antsi
    "sgz_Latn",  # Sursurunga
    "shj_Latn",  # Shatt
    "shp_Latn",  # Shipibo-Conibo
    "sim_Latn",  # Mende (Papua New Guinea)
    "sja_Latn",  # Epena
    "sll_Latn",  # Salt-Yui
    "smk_Latn",  # Bolinao
    "snc_Latn",  # Sinaugoro
    "snn_Latn",  # Siona
    "snp_Latn",  # Siane
    "snx_Latn",  # Sam
    "sny_Latn",  # Saniyo-Hiyewe
    "som_Latn",  # Somali
    "soq_Latn",  # Kanasi
    "soy_Latn",  # Miyobe
    "spa_Latn",  # Spanish
    "spl_Latn",  # Selepet
    "spm_Latn",  # Akukem
    "spp_Latn",  # Supyire Senoufo
    "sps_Latn",  # Saposa
    "spy_Latn",  # Sabaot
    "sri_Latn",  # Siriano
    "srm_Latn",  # Saramaccan
    "srn_Latn",  # Sranan Tongo
    "srp_Latn",  # Serbian
    "srq_Latn",  # Sirionó
    "ssd_Latn",  # Siroi
    "ssg_Latn",  # Seimat
    "ssx_Latn",  # Samberigi
    "stp_Latn",  # Southeastern Tepehuan
    "sua_Latn",  # Sulka
    "sue_Latn",  # Suena
    "sus_Arab",  # Susu
    "suz_Latn",  # Sunwar
    "swe_Latn",  # Swedish
    "swh_Latn",  # Swahili (Kiswahili)
    "swp_Latn",  # Suau
    "sxb_Latn",  # Suba
    "tac_Latn",  # Lowland Tarahumara
    "taj_Deva",  # Tajik
    "tam_Taml",  # Tamil
    "tav_Latn",  # Tatuyo
    "taw_Latn",  # Tai
    "tbc_Latn",  # Takia
    "tbf_Latn",  # Mandara
    "tbg_Latn",  # North Tairora
    "tbo_Latn",  # Tawala
    "tbz_Latn",  # Ditammari
    "tca_Latn",  # Ticuna
    "tcs_Latn",  # Torres Strait Creole
    "tcz_Latn",  # Thado Chin
    "tdt_Latn",  # Tetun Dili
    "tee_Latn",  # Huehuetla Tepehua
    "tel_Telu",  # Telugu
    "ter_Latn",  # Terêna
    "tet_Latn",  # Tetum
    "tew_Latn",  # Tewa (USA)
    "tfr_Latn",  # Me'phaa
    "tgk_Cyrl",  # Tajik
    "tgl_Latn",  # Tagalog
    "tgo_Latn",  # Sudest
    "tgp_Latn",  # Tangoa
    "tha_Thai",  # Thai
    "tif_Latn",  # Tifal
    "tim_Latn",  # Timbe
    "tiw_Latn",  # Tiwi
    "tiy_Latn",  # Tiruray
    "tke_Latn",  # Tsikimba
    "tku_Latn",  # Upper Necaxa Totonac
    "tlf_Latn",  # Telefol
    "tmd_Latn",  # Haruai
    "tna_Latn",  # Tacana
    "tnc_Latn",  # Tanimuca-Retuarã
    "tnk_Latn",  # Kwamera
    "tnn_Latn",  # North Tanna
    "tnp_Latn",  # Whitesands
    "toc_Latn",  # Coyutla Totonac
    "tod_Latn",  # Toma
    "tof_Latn",  # Columbia-Wenatchi
    "toj_Latn",  # Tojolabal
    "ton_Latn",  # Tonga (Tonga Islands)
    "too_Latn",  # Xicotepec De Juárez Totonac
    "top_Latn",  # Papantla Totonac
    "tos_Latn",  # Highland Totonac
    "tpa_Latn",  # Taupota
    "tpi_Latn",  # Tok Pisin
    "tpt_Latn",  # Tlachichilco Tepehua
    "tpz_Latn",  # Tinputz
    "trc_Latn",  # Copala Triqui
    "tsw_Latn",  # Tsishingini
    "ttc_Latn",  # Tektiteko
    "tte_Latn",  # Bwanabwana
    "tuc_Latn",  # Mutu
    "tue_Latn",  # Tuyuca
    "tuf_Latn",  # Central Tunebo
    "tuo_Latn",  # Tucano
    "tur_Latn",  # Turkish
    "tvk_Latn",  # Southeast Ambrym
    "twi_Latn",  # Twi
    "txq_Latn",  # Tii
    "txu_Latn",  # Kayapó
    "tzj_Latn",  # Tz'utujil
    "tzo_Latn",  # Tzotzil
    "ubr_Latn",  # Ubir
    "ubu_Latn",  # Umbu-Ungu
    "udu_Latn",  # Uduk
    "uig_Latn",  # Uighur
    "ukr_Cyrl",  # Ukrainian
    "uli_Latn",  # Ulithian
    "ulk_Latn",  # Meriam
    "upv_Latn",  # Uripiv-Wala-Rano-Atchin
    "ura_Latn",  # Urarina
    "urb_Latn",  # Kaapor
    "urd_Arab",  # Urdu
    "uri_Latn",  # Urim
    "urt_Latn",  # Urartian
    "urw_Latn",  # Sop
    "usa_Latn",  # Usarufa
    "usp_Latn",  # Uspanteco
    "uvh_Latn",  # Uri
    "uvl_Latn",  # Lote
    "vid_Latn",  # Vidunda
    "vie_Latn",  # Vietnamese
    "viv_Latn",  # Iduna
    "vmy_Latn",  # Mixtec, many Mixtec languages use Latin script, though there are multiple variants of Mixtec.
    "waj_Latn",  # Waffa
    "wal_Ethi",  # Wolaytta, uses the Ethiopic script
    "wap_Latn",  # Wapishana
    "wat_Latn",  # Kaninuwa
    "wbi_Latn",  # Vwanji
    "wbp_Latn",  # Warlpiri
    "wed_Latn",  # Wedau
    "wer_Latn",  # Weri
    "wim_Latn",  # Wik-Mungkan
    "wiu_Latn",  # Wiru
    "wiv_Latn",  # Vitu
    "wmt_Latn",  # Mwani
    "wmw_Latn",  # Mwani
    "wnc_Latn",  # Wantoat
    "wnu_Latn",  # Usan
    "wol_Latn",  # Wolof, primarily uses Latin script but also uses Arabic in some contexts
    "wos_Latn",  # Hanga Hundi
    "wrk_Latn",  # Garrwa
    "wro_Latn",  # Worrorra
    "wrs_Latn",  # Waris
    "wsk_Latn",  # Waskia
    "wuv_Latn",  # Wuvulu-Aua
    "xav_Latn",  # Xavante
    "xbi_Latn",  # Kombio
    "xed_Latn",  # Hdi
    "xla_Latn",  # Kamula
    "xnn_Latn",  # Northern Kankanay
    "xon_Latn",  # Konkomba
    "xsi_Latn",  # Sio
    "xtd_Latn",  # Diuxi-Tilantongo Mixtec
    "xtm_Latn",  # Magdalena Peñasco Mixtec
    "yaa_Latn",  # Yaminahua
    "yad_Latn",  # Yagua
    "yal_Latn",  # Yalunka
    "yap_Latn",  # Yapese
    "yaq_Latn",  # Yaqui
    "yby_Latn",  # Yawalapití
    "ycn_Latn",  # Yucuna
    "yka_Latn",  # Yakan
    "yle_Latn",  # Yele
    "yml_Latn",  # Iamalele
    "yon_Latn",  # Yongkom
    "yor_Latn",  # Yoruba, primarily uses the Latin script
    "yrb_Latn",  # Yareba
    "yre_Latn",  # Yaouré
    "yss_Latn",  # Yessan-Mayo
    "yuj_Latn",  # Karkar-Yuri
    "yut_Latn",  # Yopno
    "yuw_Latn",  # Yau (Morobe Province)
    "yva_Latn",  # Yawa
    "zaa_Latn",  # Sierra de Juárez Zapotec
    "zab_Latn",  # San Juan Guelavía Zapotec
    "zac_Latn",  # Ocotlán Zapotec
    "zad_Latn",  # Cajonos Zapotec
    "zai_Latn",  # Isthmus Zapotec
    "zaj_Latn",  # Zaramo
    "zam_Latn",  # Miahuatlán Zapotec
    "zao_Latn",  # Ozolotepec Zapotec
    "zap_Latn",  # Zapotec, uses Latin script across many varieties
    "zar_Latn",  # Rincón Zapotec
    "zas_Latn",  # Santo Domingo Albarradas Zapotec
    "zat_Latn",  # Tabaa Zapotec
    "zav_Latn",  # Yatzachi Zapotec
    "zaw_Latn",  # Mitla Zapotec
    "zca_Latn",  # Coatecas Altas Zapotec
    "zga_Latn",  # Ganza
    "zia_Latn",  # Zia
    "ziw_Latn",  # Zigula
    "zlm_Latn",  # Malay (individual language), Latin script
    "zos_Latn",  # Francisco León Zoque
    "zpc_Latn",  # Choapan Zapotec
    "zpl_Latn",  # Lachixío Zapotec
    "zpm_Latn",  # Mixtepec Zapotec
    "zpo_Latn",  # Amatlán Zapotec
    "zpq_Latn",  # Zoogocho Zapotec
    "zpu_Latn",  # Yalálag Zapotec
    "zpv_Latn",  # Chichicapan Zapotec
    "zpz_Latn",  # Texmelucan Zapotec
    "zsr_Latn",  # Southern Rincon Zapotec
    "ztq_Latn",  # Quioquitani-Quierí Zapotec
    "zty_Latn",  # Yatee Zapotec
    "zyp_Latn",  # Zyphe Chin
]

# train split because validation/test splits are extremely small in a lot of cases
_SPLIT = ["train"]

_N = 256


def extend_lang_pairs_english_centric() -> dict[str, list[str]]:
    # add all language pairs with english as source or target
    hf_lang_subset2isolang = {}
    for lang in _LANGUAGES:
        pair = f"eng_Latn-{lang}"
        hf_lang_subset2isolang[pair] = ["eng-Latn", lang.replace("_", "-")]
        pair = f"{lang}-eng_Latn"
        hf_lang_subset2isolang[pair] = [lang.replace("_", "-"), "eng-Latn"]
    return hf_lang_subset2isolang


_LANGUAGES_MAPPING = extend_lang_pairs_english_centric()


class BibleNLPBitextMining(AbsTaskBitextMining, CrosslingualTask):
    metadata = TaskMetadata(
        name="BibleNLPBitextMining",
        dataset={
            "path": "davidstap/biblenlp-corpus-mmteb",
            "revision": "264a18480c529d9e922483839b4b9758e690b762",
            "split": f"train[:{_N}]",
            "trust_remote_code": True,
        },
        description="Partial Bible translations in 829 languages, aligned by verse.",
        reference="https://arxiv.org/abs/2304.09919",
        type="BitextMining",
        category="s2s",
        eval_splits=_SPLIT,
        eval_langs=_LANGUAGES_MAPPING,
        main_score="f1",
        # World English Bible (WEB) first draft 1997, finished 2020
        date=("1997-01-01", "2020-12-31"),
        form=["written"],
        domains=["Religious"],
        task_subtypes=[],
        license="CC-BY-SA-4.0",
        socioeconomic_status="medium",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="created",
        n_samples={"train": _N},
        avg_character_length={"train": 120},
        bibtex_citation="""@article{akerman2023ebible,
            title={The eBible Corpus: Data and Model Benchmarks for Bible Translation for Low-Resource Languages},
            author={Akerman, Vesa and Baines, David and Daspit, Damien and Hermjakob, Ulf and Jang, Taeho and Leong, Colin and Martin, Michael and Mathew, Joel and Robie, Jonathan and Schwarting, Marcus},
            journal={arXiv preprint arXiv:2304.09919},
            year={2023}
        }""",
    )

    def load_data(self, **kwargs: Any) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return

        self.dataset = {}
        seen_pairs = []
        for lang in self.langs:
            # e.g. 'aai_Latn-eng_Latn' -> 'eng-aai'
            hf_lang_name = self._transform_lang_name_hf(lang)

            if hf_lang_name in seen_pairs:
                self.dataset[lang] = self.dataset[self._swap_substrings(lang)]
            else:
                dataset = datasets.load_dataset(
                    name=self._transform_lang_name_hf(lang),
                    **self.metadata_dict["dataset"],
                )
                self.dataset[lang] = datasets.DatasetDict({"train": dataset})
                seen_pairs.append(hf_lang_name)

        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        # Convert to standard format
        for lang in self.langs:
            l1, l2 = [l.split("_")[0] for l in lang.split("-")]

            self.dataset[lang] = self.dataset[lang].rename_columns(
                {l1: "sentence1", l2: "sentence2"}
            )

    @staticmethod
    def _transform_lang_name_hf(lang):
        # Transform language name to match huggingface configuration
        langs = [l.split("_")[0] for l in lang.split("-")]
        if langs[1] == "eng":
            langs[0], langs[1] = langs[1], langs[0]
        return "-".join(langs)

    @staticmethod
    def _swap_substrings(string):
        substring1, substring2 = string.split("-")
        parts = string.split("-")
        if substring1 in parts and substring2 in parts:
            index1 = parts.index(substring1)
            index2 = parts.index(substring2)
            parts[index1], parts[index2] = parts[index2], parts[index1]
            new_string = "-".join(parts)
            return new_string
        else:
            return string
