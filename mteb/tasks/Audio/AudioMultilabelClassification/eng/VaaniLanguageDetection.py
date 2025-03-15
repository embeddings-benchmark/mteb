from __future__ import annotations

from ast import literal_eval

from datasets import Audio

from mteb.abstasks.Audio.AbsTaskAudioMultilabelClassification import (
    AbsTaskAudioMultilabelClassification,
)
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata


# Deva used for north indian languages classed as Hindi dialects and Maithili
# khortha lacks an ISO 639-3 code, using mag-Deva as closest match
# bearibashe and khadiboli are classed as Malayalam and Hindi respectively
class VaaniLanguageDetection(AbsTaskAudioMultilabelClassification, MultilingualTask):
    metadata = TaskMetadata(
        name="IISc-Vaani",
        description="Multilingual Language Detection - Multilabel Speech Classification.",
        reference="https://huggingface.co/datasets/ARTPARK-IISc/Vaani",
        dataset={
            "path": "ARTPARK-IISc/Vaani",
            "revision": "b133cc2e158905798a723f29685e483517e61275",
        },
        type="AudioMultilabelClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs={
            # Telugu: 161.407, Hindi: 51.933, Urdu: 9.869, Marathi: 1.378, Bengali: 1.188, Tamil: 1.167, Kannada: 0.131
            "AndhraPradesh_Anantpur": [
                "tel-Telu",
                "urd-Arab",
                "ben-Beng",
                "hin-Deva",
                "mar-Deva",
                "tam-Taml",
                "kan-Knda",
            ],
            # Telugu: 169.261, Hindi: 27.124, Tamil: 2.476, Bengali: 1.02, Santali: 0.116
            "AndhraPradesh_Chittoor": [
                "tel-Telu",
                "hin-Deva",
                "tam-Taml",
                "ben-Beng",
                "sat-Olck",
            ],
            # Telugu: 188.831, Hindi: 17.125, Bengali: 0.532, Marathi: 0.851, Tamil: 0.394, English: 0.074, Urdu: 9.723
            "AndhraPradesh_Guntur": [
                "tel-Telu",
                "hin-Deva",
                "ben-Beng",
                "mar-Deva",
                "tam-Taml",
                "eng-Latn",
                "urd-Arab",
            ],
            # Hindi: 19.702, Telugu: 155.909, Marathi: 0.18, Bengali: 0.297, Tamil: 0.515, English: 0.296
            "AndhraPradesh_Krishna": [
                "hin-Deva",
                "tel-Telu",
                "mar-Deva",
                "ben-Beng",
                "tam-Taml",
                "eng-Latn",
            ],
            # Marathi: 0.173, Telugu: 221.554, Hindi: 11.606, Kannada: 0.293, Bhojpuri: 0.219
            "AndhraPradesh_Srikakulam": [
                "mar-Deva",
                "tel-Telu",
                "hin-Deva",
                "kan-Knda",
                "bho-Deva",
            ],
            # Telugu: 164.229, Hindi: 11.466, Marathi: 0.314, Tamil: 0.28, Bengali: 0.278, Kannada: 0.28, Maithili: 0.101
            "AndhraPradesh_Vishakapattanam": [
                "tel-Telu",
                "hin-Deva",
                "mar-Deva",
                "tam-Taml",
                "ben-Beng",
                "kan-Knda",
                "mai-Deva",
            ],
            # Hindi: 167.755, Maithili: 22.732, Angika: 1.678, Bengali: 0.254, Bhojpuri: 0.216, Telugu: 0.096, Urdu: 0.612
            "Bihar_Araria": [
                "hin-Deva",
                "mai-Deva",
                "anp-Beng",
                "bho-Deva",
                "tel-Telu",
                "urd-Arab",
            ],
            # Angika: 12.869, Hindi: 131.54, Maithili: 32.628, Bhojpuri: 1.217, Telugu: 0.12, Magahi: 0.082, Urdu: 15.825
            "Bihar_Begusarai": [
                "anp-Beng",
                "hin-Deva",
                "mai-Deva",
                "bho-Deva",
                "tel-Telu",
                "mag-Deva",
                "urd-Arab",
            ],
            # Angika: 45.029, Maithili: 9.685, Hindi: 141.861, Marathi: 0.647, Bhojpuri: 0.157, Magahi: 0.275, Bengali: 0.582, Telugu: 0.193, Urdu: 0.284
            "Bihar_Bhagalpur": [
                "anp-Beng",
                "mai-Deva",
                "hin-Deva",
                "mar-Deva",
                "bho-Deva",
                "mag-Deva",
                "ben-Beng",
                "tel-Telu",
                "urd-Arab",
            ],
            # Hindi: 176.606, Maithili: 20.816, Marathi: 0.301, Kannada: 0.147, Urdu: 1.314
            "Bihar_Darbhanga": [
                "hin-Deva",
                "mai-Deva",
                "mar-Deva",
                "kan-Knda",
                "urd-Arab",
            ],
            # Bhojpuri: 39.012, Hindi: 168.973, Bengali: 1.15, Maithili: 2.491, Marathi: 0.708, Urdu: 0.539
            "Bihar_EastChamparan": [
                "bho-Deva",
                "hin-Deva",
                "ben-Beng",
                "mai-Deva",
                "mar-Deva",
                "urd-Arab",
            ],
            # Hindi: 187.242, Bhojpuri: 0.343, Magahi: 35.393, Maithili: 3.048, Marwari: 0.039, Marathi: 1.142, Bengali: 0.126
            "Bihar_Gaya": [
                "hin-Deva",
                "bho-Deva",
                "mag-Deva",
                "mai-Deva",
                "mwr-Deva",
                "mar-Deva",
                "ben-Beng",
            ],
            # Hindi: 192.245, Bhojpuri: 35.267, Maithili: 2.226, Bengali: 0.201, Marathi: 0.46
            "Bihar_Gopalganj": [
                "hin-Deva",
                "bho-Deva",
                "mai-Deva",
                "ben-Beng",
                "mar-Deva",
            ],
            # Hindi: 184.454, Magahi: 12.912, Bengali: 0.651, Maithili: 2.89, Marathi: 0.77, Urdu: 0.761
            "Bihar_Jahanabad": [
                "hin-Deva",
                "mag-Deva",
                "ben-Beng",
                "mai-Deva",
                "mar-Deva",
                "urd-Arab",
            ],
            # Bengali: 0.275, Hindi: 186.895, Magahi: 6.467, Maithili: 4.571, Angika: 10.953, Tamil: 0.155
            "Bihar_Jamui": [
                "ben-Beng",
                "hin-Deva",
                "mag-Deva",
                "mai-Deva",
                "anp-Deva",
                "tam-Taml",
            ],
            # Hindi: 165.249, Marathi: 0.199, Bengali: 3.529, Maithili: 4.121, Magahi: 0.242, Surjapuri: 25.919, Urdu: 3.238
            "Bihar_Kishanganj": [
                "hin-Deva",
                "mar-Deva",
                "ben-Beng",
                "mai-Deva",
                "mag-Deva",
                "sjp-Deva",
                "urd-Arab",
            ],
            # Hindi: 159.678, Marathi: 0.383, Bhojpuri: 1.157, Maithili: 3.331, Magahi: 28.746, Konkani: 0.245, Telugu: 0.117
            "Bihar_Lakhisarai": [
                "hin-Deva",
                "mar-Deva",
                "bho-Deva",
                "mai-Deva",
                "mag-Deva",
                "kok-Deva",
                "tel-Telu",
            ],
            # Angika: 3.113, Maithili: 77.053, Magahi: 0.24, Hindi: 112.652, Khortha: 0.239, Bengali: 0.673, Telugu: 0.176, Marathi: 0.187
            "Bihar_Madhepura": [
                "anp-Beng",
                "mai-Deva",
                "mag-Deva",
                "hin-Deva",
                "mag-Deva",
                "ben-Beng",
                "tel-Telu",
                "mar-Deva",
            ],
            # Hindi: 146.714, Maithili: 8.447, Kurmali: 0.99, Angika: 0.143, Bhojpuri: 28.504, Bengali: 0.147, Bajjika: 12.282
            "Bihar_Muzaffarpur": [
                "hin-Deva",
                "mai-Deva",
                "kyw-Deva",
                "anp-Beng",
                "bho-Deva",
                "ben-Beng",
                "vjk-Deva",
            ],
            # Maithili: 19.789, Hindi: 175.595, Chhattisgarhi: 0.065, Angika: 9.669, Urdu: 0.496
            "Bihar_Purnia": [
                "mai-Deva",
                "hin-Deva",
                "hne-Deva",
                "anp-Beng",
                "urd-Arab",
            ],
            # Angika: 2.03, Hindi: 129.425, Maithili: 37.877, Magahi: 0.269, Chhattisgarhi: 0.193, Bengali: 0.135, Kannada: 0.243, Urdu: 19.717
            "Bihar_Saharsa": [
                "anp-Beng",
                "hin-Deva",
                "mai-Deva",
                "mag-Deva",
                "hne-Deva",
                "ben-Beng",
                "kan-Knda",
                "urd-Arab",
            ],
            # Maithili: 91.061, Hindi: 111.313, Magahi: 15.456, Angika: 2.87, Bhojpuri: 0.832, Bajjika: 0.116, Urdu: 0.205
            "Bihar_Samastipur": [
                "mai-Deva",
                "hin-Deva",
                "mag-Deva",
                "anp-Beng",
                "bho-Deva",
                "vjk-Deva",
                "urd-Arab",
            ],
            # Hindi: 162.94, Bhojpuri: 61.655, Maithili: 3.343, Marathi: 0.823
            "Bihar_Saran": ["hin-Deva", "bho-Deva", "mai-Deva", "mar-Deva"],
            # Hindi: 208.209, Bajjika: 1.145, Maithili: 4.256, Bhojpuri: 1.928, Urdu: 3.936, Tamil: 0.182, Angika: 0.405, Sadri: 0.284, Khortha: 0.391, Bengali: 0.016
            "Bihar_Sitamarhi": [
                "hin-Deva",
                "vjk-Deva",
                "mai-Deva",
                "bho-Deva",
                "urd-Arab",
                "tam-Taml",
                "anp-Deva",
                "sck-Deva",
                "ben-Beng",
                "mag-Deva",
            ],
            # Maithili: 71.558, Angika: 1.932, Hindi: 111.46, Bhojpuri: 1.498, Urdu: 0.281
            "Bihar_Supaul": [
                "mai-Deva",
                "anp-Beng",
                "hin-Deva",
                "bho-Deva",
                "urd-Arab",
            ],
            # Hindi: 127.011, Maithili: 13.264, Bajjika: 58.804, Bengali: 0.226, Marathi: 0.393, Bhojpuri: 1.484, Nepali: 0.182
            "Bihar_Vaishali": [
                "hin-Deva",
                "mai-Deva",
                "vjk-Deva",
                "ben-Beng",
                "mar-Deva",
                "bho-Deva",
                "nep-Deva",
            ],
            # Hindi: 172.646, Chhattisgarhi: 8.294, Awadhi: 5.753
            "Chhattisgarh_Balrampur": ["hin-Deva", "hne-Deva", "awa-Deva"],
            # Halbi: 41.791, Hindi: 130.032, Chhattisgarhi: 24.205, Bhatri: 1.029, Oriya: 0.554, Bengali: 0.241, Gondi: 0.101
            "Chhattisgarh_Bastar": [
                "hlb-Orya",
                "hin-Deva",
                "hne-Deva",
                "bgw-Orya",
                "ori-Orya",
                "ben-Beng",
                "gon-Gonm",
            ],
            # Hindi: 191.55, Chhattisgarhi: 12.491, Marathi: 0.567, Bengali: 0.837, Maithili: 0.304
            "Chhattisgarh_Bilaspur": [
                "hin-Deva",
                "hne-Deva",
                "mar-Deva",
                "ben-Beng",
                "mai-Deva",
            ],
            # Chhattisgarhi: 55.097, Hindi: 94.596, Sadri: 25.616, Bengali: 0.428, Kurukh: 7.439, Oriya: 0.149, Agariya: 0.063
            "Chhattisgarh_Jashpur": [
                "hne-Deva",
                "hin-Deva",
                "sck-Deva",
                "ben-Beng",
                "kru-Deva",
                "ori-Orya",
                "agi-Deva",
            ],
            # Hindi: 174.052, Chhattisgarhi: 25.098, Marathi: 0.185, Maithili: 0.257
            "Chhattisgarh_Kabirdham": ["hin-Deva", "hne-Deva", "mar-Deva", "mai-Deva"],
            # Hindi: 162.584, Chhattisgarhi: 32.501
            "Chhattisgarh_Korba": ["hin-Deva", "hne-Deva"],
            # Hindi: 156.895, Chhattisgarhi: 29.899, Oriya: 0.565
            "Chhattisgarh_Raigarh": ["hin-Deva", "hne-Deva", "ori-Orya"],
            # Hindi: 113.163, Kannada: 0.131, Chhattisgarhi: 74.216, Bengali: 0.272, Marathi: 0.252, English: 0.099
            "Chhattisgarh_Rajnandgaon": [
                "hin-Deva",
                "kan-Knda",
                "hne-Deva",
                "ben-Beng",
                "mar-Deva",
                "eng-Latn",
            ],
            # Hindi: 100.768, Chhattisgarhi: 58.539, Bengali: 0.247, Kurukh: 3.961, Surgujia: 16.684
            "Chhattisgarh_Sarguja": [
                "hin-Deva",
                "hne-Deva",
                "ben-Beng",
                "kru-Deva",
                "sck-Deva",
            ],
            # Hindi: 97.468, Oriya: 0.478, Bengali: 0.311, Gondi: 3.495, Duruwa: 0.521, Dorli: 1.363, Chhattisgarhi: 32.299
            "Chhattisgarh_Sukma": [
                "hin-Deva",
                "ori-Orya",
                "ben-Beng",
                "gon-Gonm",
                "pci-Deva",
                "kff-Telu",
                "hne-Deva",
            ],
            # Hindi: 97.382, Konkani: 68.399, Bengali: 0.601, Gujarati: 0.245, Marathi: 22.651, Kannada: 0.316, English: 0.113
            "Goa_NorthSouthGoa": [
                "hin-Deva",
                "kok-Deva",
                "ben-Beng",
                "guj-Gujr",
                "mar-Deva",
                "kan-Knda",
                "eng-Latn",
            ],
            # Hindi: 165.005, Marathi: 0.572, Bhojpuri: 0.26, Maithili: 0.458, Bengali: 21.924, Khortha: 22.436
            "Jharkhand_Jamtara": [
                "hin-Deva",
                "mar-Deva",
                "bho-Deva",
                "mai-Deva",
                "ben-Beng",
                "mag-Deva",
            ],
            # Hindi: 162.87, Angika: 0.769, Marathi: 0.55, Bhojpuri: 1.171, Magahi: 0.489, Santali: 5.687, Chhattisgarhi: 0.14, Kurmali: 0.157, English: 0.046, Bengali: 20.25, Khortha: 3.524
            "Jharkhand_Sahebganj": [
                "hin-Deva",
                "anp-Deva",
                "mar-Deva",
                "bho-Deva",
                "mag-Deva",
                "sat-Olck",
                "hne-Deva",
                "kyw-Deva",
                "eng-Latn",
                "ben-Beng",
                "mag-Deva",
            ],
            # Hindi: 12.865, Kannada: 143.791, Marathi: 34.809, Malayalam: 0.273, Telugu: 3.582, Urdu: 0.204
            "Karnataka_Belgaum": [
                "hin-Deva",
                "kan-Knda",
                "mar-Deva",
                "mal-Mlym",
                "tel-Telu",
                "urd-Arab",
            ],
            # Kannada: 153.043, Hindi: 8.392, Urdu: 2.593, Telugu: 44.414, Tamil: 0.716, English: 0.388, Bearybashe: 0.489, Malayalam: 0.165
            "Karnataka_Bellary": [
                "kan-Knda",
                "hin-Deva",
                "urd-Arab",
                "tel-Telu",
                "tam-Taml",
                "eng-Latn",
                "mal-Mlym",
            ],
            # Kannada: 173.44, Hindi: 10.735, Malayalam: 0.283, Telugu: 0.288, Urdu: 3.193, Bengali: 1.393, English: 0.146
            "Karnataka_Bijapur": [
                "kan-Knda",
                "hin-Deva",
                "mal-Mlym",
                "tel-Telu",
                "urd-Arab",
                "ben-Beng",
                "eng-Latn",
            ],
            # Kannada: 181.695, Hindi: 5.65, Malayalam: 0.322, Telugu: 0.213, Tamil: 4.423
            "Karnataka_Chamrajnagar": [
                "kan-Knda",
                "hin-Deva",
                "mal-Mlym",
                "tel-Telu",
                "tam-Taml",
            ],
            # Kannada: 116.32, Hindi: 13.423, Bearybashe: 6.472, Urdu: 6.674, Telugu: 0.212, Tulu: 39.943, Malayalam: 0.475
            "Karnataka_DakshinKannada": [
                "kan-Knda",
                "hin-Deva",
                "urd-Arab",
                "tel-Telu",
                "tcy-Tutg",
                "mal-Mlym",
            ],
            # Kannada: 178.79, Hindi: 8.95, Bhojpuri: 0.291, Malayalam: 0.279, Telugu: 9.98, Urdu: 0.453
            "Karnataka_Dharwad": [
                "kan-Knda",
                "hin-Deva",
                "bho-Deva",
                "mal-Mlym",
                "tel-Telu",
                "urd-Arab",
            ],
            # Kannada: 130.248, Hindi: 17.424, Telugu: 15.542, Malayalam: 0.237, Urdu: 20.037
            "Karnataka_Gulbarga": [
                "kan-Knda",
                "hin-Deva",
                "tel-Telu",
                "mal-Mlym",
                "urd-Arab",
            ],
            # Hindi: 12.808, Kannada: 188.37, Malayalam: 0.215, Bengali: 0.498, Tamil: 0.068, Urdu: 4.342, Lambani: 0.226, Telugu: 0.243
            "Karnataka_Mysore": [
                "hin-Deva",
                "kan-Knda",
                "mal-Mlym",
                "ben-Beng",
                "tam-Taml",
                "urd-Arab",
                "lmn-Deva",
                "tel-Telu",
            ],
            # Kannada: 162.703, Hindi: 8.855, Telugu: 21.205, Bhojpuri: 0.277, Malayalam: 0.241, Urdu: 1.804
            "Karnataka_Raichur": [
                "kan-Knda",
                "hin-Deva",
                "tel-Telu",
                "bho-Deva",
                "mal-Mlym",
                "urd-Arab",
            ],
            # Kannada: 188.526, Hindi: 5.931, Telugu: 0.153, Malayalam: 0.262, Tamil: 0.302, Bengali: 0.054, Urdu: 1.118
            "Karnataka_Shimoga": [
                "kan-Knda",
                "hin-Deva",
                "tel-Telu",
                "mal-Mlym",
                "tam-Taml",
                "ben-Beng",
                "urd-Arab",
            ],
            # Marathi: 106.157, Hindi: 79.5, Bengali: 0.132, Telugu: 0.14
            "Maharashtra_Aurangabad": ["mar-Deva", "hin-Deva", "ben-Beng", "tel-Telu"],
            # Hindi: 80.426, Marathi: 105.643, Malvani: 0.313
            "Maharashtra_Chandrapur": ["hin-Deva", "mar-Deva", "kok-Deva"],
            # Hindi: 52.372, Marathi: 132.888, Bhili: 2.879, Khandeshi: 5.731
            "Maharashtra_Dhule": ["hin-Deva", "mar-Deva", "bhb-Deva", "khn-Deva"],
            # Hindi: 84.323, Marathi: 108.616, Malvani: 0.291, Chhattisgarhi: 0.193, Gujarati: 0.123
            "Maharashtra_Nagpur": [
                "hin-Deva",
                "mar-Deva",
                "kok-Deva",
                "hne-Deva",
                "guj-Gujr",
            ],
            # Marathi: 163.223, Hindi: 37.32, Maithili: 0.502, Urdu: 0.049
            "Maharashtra_Pune": ["mar-Deva", "hin-Deva", "mai-Deva", "urd-Arab"],
            # Hindi: 48.225, Marathi: 122.63, Malvani: 14.973
            "Maharashtra_Sindhudurga": ["hin-Deva", "mar-Deva", "kok-Deva"],
            # Marathi: 157.331, Hindi: 49.983, Malvani: 0.133, Maithili: 0.156, Bengali: 0.19, Kannada: 0.248
            "Maharashtra_Solapur": [
                "mar-Deva",
                "hin-Deva",
                "kok-Deva",
                "mai-Deva",
                "ben-Beng",
                "kan-Knda",
            ],
            # Hindi: 63.517, Shekhawati: 0.934, Wagdi: 0.366, Marwari: 36.964, Bagri: 0.579, Mewari: 0.318, Bengali: 0.461, Harauti: 0.29, Jaipuri: 0.803, Gujarati: 0.238, English: 0.168, Rajasthani: 89.46, Mewati: 0.016
            "Rajasthan_Churu": [
                "hin-Deva",
                "swv-Deva",
                "wbr-Deva",
                "mwr-Deva",
                "bgq-Deva",
                "mtr-Deva",
                "ben-Beng",
                "hoj-Deva",
                "dhd-Deva",
                "guj-Gujr",
                "eng-Latn",
                "raj-Deva",
                "wtm-Deva",
            ],
            # Hindi: 57.511, Marwari: 64.981, Bengali: 0.217, Jaipuri: 1.44, Marathi: 0.063, Mewari: 0.288, Rajasthani: 71.975
            "Rajasthan_Nagaur": [
                "hin-Deva",
                "mwr-Deva",
                "ben-Beng",
                "dhd-Deva",
                "mar-Deva",
                "mtr-Deva",
                "raj-Deva",
            ],
            # Telugu: 215.153, Hindi: 6.151, English: 0.683, Bengali: 0.526, Urdu: 1.999
            "Telangana_Karimnagar": [
                "tel-Telu",
                "hin-Deva",
                "eng-Latn",
                "ben-Beng",
                "urd-Arab",
            ],
            # Telugu: 209.335, Hindi: 6.183, Bengali: 0.245, Malayalam: 0.172, Lambani: 0.442
            "Telangana_Nalgonda": [
                "tel-Telu",
                "hin-Deva",
                "ben-Beng",
                "mal-Mlym",
                "lmn-Deva",
            ],
            # Hindi: 192.439, Bundeli: 2.898, Khariboli: 4.662, Awadhi: 0.331, Marathi: 0.343, Bengali: 0.248, Urdu: 1.784
            "UttarPradesh_Budaun": [
                "hin-Deva",
                "bns-Deva",
                "awa-Deva",
                "mar-Deva",
                "ben-Beng",
                "urd-Arab",
            ],
            # Bhojpuri: 89.671, Hindi: 102.05, Awadhi: 0.112, Maithili: 0.294, Khariboli: 0.228, Marathi: 0.356
            "UttarPradesh_Deoria": [
                "bho-Deva",
                "hin-Deva",
                "awa-Deva",
                "mai-Deva",
                "mar-Deva",
            ],
            # Hindi: 193.835, Khariboli: 2.844, English: 0.088, Marathi: 0.442
            "UttarPradesh_Etah": ["hin-Deva", "eng-Latn", "mar-Deva"],
            # Bhojpuri: 75.199, Hindi: 112.38, Bengali: 0.254, Awadhi: 0.34, Marathi: 0.473, Chhattisgarhi: 0.271, Khariboli: 0.36, Urdu: 0.284, Tamil: 0.286
            "UttarPradesh_Ghazipur": [
                "bho-Deva",
                "hin-Deva",
                "ben-Beng",
                "awa-Deva",
                "mar-Deva",
                "hne-Deva",
                "urd-Arab",
                "tam-Taml",
            ],
            # Bhojpuri: 73.951, Hindi: 118.869, Marathi: 1.061, Khariboli: 0.829
            "UttarPradesh_Gorakhpur": ["bho-Deva", "hin-Deva", "mar-Deva"],
            # Hindi: 200.863, Urdu: 0.049, Khariboli: 1.567, Bundeli: 3.65, Marathi: 0.586, Awadhi: 0.059, Bengali: 0.187
            "UttarPradesh_Hamirpur": [
                "hin-Deva",
                "urd-Arab",
                "mar-Deva",
                "bns-Deva",
                "awa-Deva",
                "ben-Beng",
            ],
            # Hindi: 222.448, Bundeli: 2.468, Awadhi: 0.179, Khariboli: 1.122, Marathi: 0.619, Assamese: 0.233, Gujarati: 0.094
            "UttarPradesh_Jalaun": [
                "hin-Deva",
                "bns-Deva",
                "awa-Deva",
                "mar-Deva",
                "asm-Beng",
                "guj-Gujr",
            ],
            # Hindi: 191.227, Khariboli: 1.813, Marathi: 0.623, English: 0.126, Urdu: 1.447
            "UttarPradesh_JyotibaPhuleNagar": ["hin-Deva", "mar-Deva", "urd-Arab"],
            # Hindi: 193.405, Khariboli: 3.668, Marathi: 1.12, Kannada: 0.229, Awadhi: 0.128, Urdu: 0.411
            "UttarPradesh_Muzzaffarnagar": [
                "hin-Deva",
                "mar-Deva",
                "kan-Knda",
                "awa-Deva",
                "urd-Arab",
            ],
            # Hindi: 163.8, Bhojpuri: 25.842, Marathi: 0.649, Awadhi: 0.948, Khariboli: 0.632, Bengali: 0.232
            "UttarPradesh_Varanasi": [
                "hin-Deva",
                "bho-Deva",
                "mar-Deva",
                "awa-Deva",
                "ben-Beng",
            ],
            # Garhwali: 49.979, Hindi: 143.682, Bengali: 0.559, Kumaoni: 12.974, Marathi: 0.277
            "Uttarakhand_TehriGarhwal": [
                "gbm-Deva",
                "hin-Deva",
                "ben-Beng",
                "kfy-Deva",
                "mar-Deva",
            ],
            # Hindi: 95.773, Garhwali: 85.368, Bengali: 0.261, Kumaoni: 13.735, Maithili: 0.647, Marathi: 0.365
            "Uttarakhand_Uttarkashi": [
                "hin-Deva",
                "gbm-Deva",
                "ben-Beng",
                "kfy-Deva",
                "mai-Deva",
                "mar-Deva",
            ],
            # Hindi: 10.241, Bengali: 194.158
            "WestBengal_DakshinDinajpur": ["hin-Deva", "ben-Beng"],
            # Bengali: 190.021, Marathi: 1.149, Hindi: 7.207, Sadri: 0.585
            "WestBengal_Jalpaiguri": ["ben-Beng", "mar-Deva", "hin-Deva", "sck-Deva"],
            # Hindi: 3.611, Marathi: 0.653, Bengali: 177.872, Bhojpuri: 0.261
            "WestBengal_Jhargram": ["hin-Deva", "mar-Deva", "ben-Beng", "bho-Deva"],
            # Bengali: 201.036, Hindi: 10.829
            "WestBengal_Kolkata": ["ben-Beng", "hin-Deva"],
            # Bengali: 199.205, Marathi: 0.193, Hindi: 7.75
            "WestBengal_Malda": ["ben-Beng", "mar-Deva", "hin-Deva"],
            # Bengali: 210.305, Hindi: 3.5, Marathi: 0.193
            "WestBengal_North24Parganas": ["ben-Beng", "hin-Deva", "mar-Deva"],
            # Bengali: 197.688, Hindi: 2.122
            "WestBengal_PaschimMedinipur": ["ben-Beng", "hin-Deva"],
            # Bengali: 178.953, Hindi: 19.097, Rajbanshi: 0.191, Santali: 0.223
            "WestBengal_Purulia": ["ben-Beng", "hin-Deva", "rjs-Beng", "sat-Olck"],
        },
        main_score="accuracy",
        date=(
            "2024-12-01",
            "2025-01-18",
        ),
        domains=["Academic", "Spoken", "Scene", "Speech"],
        task_subtypes=["Language identification", "Spoken Language Identification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation="""@misc{vaani2025,
        author       = {VAANI Team},
        title        = {VAANI: Capturing the Language Landscape for an Inclusive Digital India (Phase 1)},
        howpublished = {{https://vaani.iisc.ac.in/}},
        year         = {2025}
        }
        """,
    )

    audio_column_name: str = "audio"
    label_column_name: str = "languagesKnown"
    samples_per_label: int = 32

    def dataset_transform(self):
        for subset in self.hf_subsets:
            self.dataset[subset] = self.dataset[subset].cast_column(
                self.audio_column_name, Audio(16_000)
            )
            self.dataset[subset] = self.dataset[subset].map(
                lambda x: {
                    self.label_column_name: literal_eval(x[self.label_column_name])
                }
            )
            self.dataset[subset] = self.stratified_subsampling(
                self.dataset[subset],
                seed=self.seed,
                splits=["train"],
                label=self.label_column_name,
                n_samples=self.samples_per_label,
            )
