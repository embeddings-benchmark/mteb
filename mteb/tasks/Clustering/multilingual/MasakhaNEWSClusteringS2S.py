from mteb.abstasks.AbsTaskAnyClustering import AbsTaskAnyClustering
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "amh": ["amh-Ethi"],
    "eng": ["eng-Latn"],
    "fra": ["fra-Latn"],
    "hau": ["hau-Latn"],
    "ibo": ["ibo-Latn"],
    "lin": ["lin-Latn"],
    "lug": ["lug-Latn"],
    "orm": ["orm-Ethi"],
    "pcm": ["pcm-Latn"],
    "run": ["run-Latn"],
    "sna": ["sna-Latn"],
    "som": ["som-Latn"],
    "swa": ["swa-Latn"],
    "tir": ["tir-Ethi"],
    "xho": ["xho-Latn"],
    "yor": ["yor-Latn"],
}


class MasakhaNEWSClusteringS2S(AbsTaskAnyClustering):
    metadata = TaskMetadata(
        name="MasakhaNEWSClusteringS2S",
        dataset={
            "path": "mteb/MasakhaNEWSClusteringS2S",
            "revision": "b3b12a9f4a43adef3b5f4055eb41a40f91209066",
        },
        description=(
            "Clustering of news article headlines from MasakhaNEWS dataset. Clustering of 10 sets on the news article label."
        ),
        reference="https://huggingface.co/datasets/masakhane/masakhanews",
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="v_measure",
        date=("2023-04-21", "2023-05-26"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="afl-3.0",
        annotations_creators="human-annotated",
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@article{adelani2023masakhanews,
  author = {David Ifeoluwa Adelani and  Marek Masiak and  Israel Abebe Azime and  Jesujoba Oluwadara Alabi and  Atnafu Lambebo Tonja and  Christine Mwase and  Odunayo Ogundepo and  Bonaventure F. P. Dossou and  Akintunde Oladipo and  Doreen Nixdorf and  Chris Chinenye Emezue and  Sana Sabah al-azzawi and  Blessing K. Sibanda and  Davis David and  Lolwethu Ndolela and  Jonathan Mukiibi and  Tunde Oluwaseyi Ajayi and  Tatiana Moteu Ngoli and  Brian Odhiambo and  Abraham Toluwase Owodunni and  Nnaemeka C. Obiefuna and  Shamsuddeen Hassan Muhammad and  Saheed Salahudeen Abdullahi and  Mesay Gemeda Yigezu and  Tajuddeen Gwadabe and  Idris Abdulmumin and  Mahlet Taye Bame and  Oluwabusayo Olufunke Awoyomi and  Iyanuoluwa Shode and  Tolulope Anu Adelani and  Habiba Abdulganiy Kailani and  Abdul-Hakeem Omotayo and  Adetola Adeeko and  Afolabi Abeeb and  Anuoluwapo Aremu and  Olanrewaju Samuel and  Clemencia Siro and  Wangari Kimotho and  Onyekachi Raphael Ogbu and  Chinedu E. Mbonu and  Chiamaka I. Chukwuneke and  Samuel Fanijo and  Jessica Ojo and  Oyinkansola F. Awosan and  Tadesse Kebede Guge and  Sakayo Toadoum Sari and  Pamela Nyatsine and  Freedmore Sidume and  Oreen Yousuf and  Mardiyyah Oduwole and  Ussen Kimanuka and  Kanda Patrick Tshinu and  Thina Diko and  Siyanda Nxakama and   Abdulmejid Tuni Johar and  Sinodos Gebre and  Muhidin Mohamed and  Shafie Abdi Mohamed and  Fuad Mire Hassan and  Moges Ahmed Mehamed and  Evrard Ngabire and  and Pontus Stenetorp},
  journal = {ArXiv},
  title = {MasakhaNEWS: News Topic Classification for African languages},
  volume = {},
  year = {2023},
}
""",
    )
