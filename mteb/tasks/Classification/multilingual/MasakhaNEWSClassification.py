from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification, MultilingualTask

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


class MasakhaNEWSClassification(AbsTaskClassification, MultilingualTask):
    metadata = TaskMetadata(
        name="MasakhaNEWSClassification",
        dataset={
            "path": "mteb/masakhanews",
            "revision": "18193f187b92da67168c655c9973a165ed9593dd",
        },
        description="MasakhaNEWS is the largest publicly available dataset for news topic classification in 16 languages widely spoken in Africa. The train/validation/test sets are available for all the 16 languages.",
        reference="https://arxiv.org/abs/2304.09972",
        category="s2s",
        type="Classification",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@misc{adelani2023masakhanews,
      title={MasakhaNEWS: News Topic Classification for African languages}, 
      author={David Ifeoluwa Adelani and Marek Masiak and Israel Abebe Azime and Jesujoba Alabi and Atnafu Lambebo Tonja and Christine Mwase and Odunayo Ogundepo and Bonaventure F. P. Dossou and Akintunde Oladipo and Doreen Nixdorf and Chris Chinenye Emezue and sana al-azzawi and Blessing Sibanda and Davis David and Lolwethu Ndolela and Jonathan Mukiibi and Tunde Ajayi and Tatiana Moteu and Brian Odhiambo and Abraham Owodunni and Nnaemeka Obiefuna and Muhidin Mohamed and Shamsuddeen Hassan Muhammad and Teshome Mulugeta Ababu and Saheed Abdullahi Salahudeen and Mesay Gemeda Yigezu and Tajuddeen Gwadabe and Idris Abdulmumin and Mahlet Taye and Oluwabusayo Awoyomi and Iyanuoluwa Shode and Tolulope Adelani and Habiba Abdulganiyu and Abdul-Hakeem Omotayo and Adetola Adeeko and Abeeb Afolabi and Anuoluwapo Aremu and Olanrewaju Samuel and Clemencia Siro and Wangari Kimotho and Onyekachi Ogbu and Chinedu Mbonu and Chiamaka Chukwuneke and Samuel Fanijo and Jessica Ojo and Oyinkansola Awosan and Tadesse Kebede and Toadoum Sari Sakayo and Pamela Nyatsine and Freedmore Sidume and Oreen Yousuf and Mardiyyah Oduwole and Tshinu Tshinu and Ussen Kimanuka and Thina Diko and Siyanda Nxakama and Sinodos Nigusse and Abdulmejid Johar and Shafie Mohamed and Fuad Mire Hassan and Moges Ahmed Mehamed and Evrard Ngabire and Jules Jules and Ivan Ssenkungu and Pontus Stenetorp},
      year={2023},
      eprint={2304.09972},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        n_samples={"test": 422},
        avg_character_length={"test": 5116.6},
    )

    def dataset_transform(self):
        for lang in self.dataset.keys():
            self.dataset[lang] = self.dataset[lang].rename_columns(
                {"category": "label"}
            )
