from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
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


class MasakhaNEWSClusteringP2P(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="MasakhaNEWSClusteringP2P",
        description="Clustering of news article headlines and texts from MasakhaNEWS dataset. Clustering of 10 sets on the news article label.",
        reference="https://huggingface.co/datasets/masakhane/masakhanews",
        dataset={
            "path": "mteb/MasakhaNEWSClusteringP2P",
            "revision": "be3f8610c052991db1e4f0209d91681f321e0dd8",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="v_measure",
        date=(
            "2023-01-01",
            "2023-12-31",
        ),  # best guess (not found in paper, dataset or datasheet)
        domains=["News", "Written", "Non-fiction"],
        task_subtypes=["Thematic clustering"],
        license="afl-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{adelani2023masakhanews,
  address = {Nusa Dua, Bali},
  author = {Adelani, David Ifeoluwa and Masiak, Marek and Azime, Israel Abebe and Alabi, Jesujoba and Tonja, Atnafu Lambebo and Mwase, Christine and Ogundepo, Odunayo and Dossou, Bonaventure F. P. and Oladipo, Akintunde and Nixdorf, Doreen and Emezue, Chris Chinenye and Al-azzawi, Sana and Sibanda, Blessing and David, Davis and Ndolela, Lolwethu and Mukiibi, Jonathan and Ajayi, Tunde and Moteu, Tatiana and Odhiambo, Brian and Owodunni, Abraham and Obiefuna, Nnaemeka and Mohamed, Muhidin and Muhammad, Shamsuddeen Hassan and Ababu, Teshome Mulugeta and Salahudeen, Saheed Abdullahi and Yigezu, Mesay Gemeda and Gwadabe, Tajuddeen and Abdulmumin, Idris and Taye, Mahlet and Awoyomi, Oluwabusayo and Shode, Iyanuoluwa and Adelani, Tolulope and Abdulganiyu, Habiba and Omotayo, Abdul-Hakeem and Adeeko, Adetola and Afolabi, Abeeb and Aremu, Anuoluwapo and Samuel, Olanrewaju and Siro, Clemencia and Kimotho, Wangari and Ogbu, Onyekachi and Mbonu, Chinedu and Chukwuneke, Chiamaka and Fanijo, Samuel and Ojo, Jessica and Awosan, Oyinkansola and Kebede, Tadesse and Sakayo, Toadoum Sari and Nyatsine, Pamela and Sidume, Freedmore and Yousuf, Oreen and Oduwole, Mardiyyah and Tshinu, Kanda and Kimanuka, Ussen and Diko, Thina and Nxakama, Siyanda and Nigusse, Sinodos and Johar, Abdulmejid and Mohamed, Shafie and Hassan, Fuad Mire and Mehamed, Moges Ahmed and Ngabire, Evrard and Jules, Jules and Ssenkungu, Ivan and Stenetorp, Pontus},
  booktitle = {Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the {A}sia-Pacific Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/2023.ijcnlp-main.10},
  editor = {Park, Jong C. and Arase, Yuki and Hu, Baotian and Lu, Wei and Wijaya, Derry and Purwarianti, Ayu and Krisnadhi, Adila Alfa},
  month = {nov},
  pages = {144--159},
  publisher = {Association for Computational Linguistics},
  title = {{M}asakha{NEWS}: News Topic Classification for {A}frican languages},
  url = {https://aclanthology.org/2023.ijcnlp-main.10/},
  year = {2023},
}
""",
    )
