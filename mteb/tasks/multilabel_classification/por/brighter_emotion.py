from mteb.abstasks.multilabel_classification import AbsTaskMultilabelClassification
from mteb.abstasks.task_metadata import TaskMetadata


class BrighterEmotionMultilabelClassification(AbsTaskMultilabelClassification):
    """Multi-label emotion presence (6 emotions) on native PT-BR social-media texts."""

    metadata = TaskMetadata(
        name="BrighterEmotionMultilabelClassification",
        description=(
            "Multi-label emotion recognition on native Brazilian-Portuguese "
            "social-media texts (BRIGHTER, SemEval-2025 Task 11). Each text is "
            "labelled with the subset of 6 emotions present (anger, disgust, "
            "fear, joy, sadness, surprise); a text may have several or none."
        ),
        reference="https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories",
        dataset={
            "path": "MTEB-BR/BrighterEmotionMultilabelClassification",
            "revision": "839707a2191b521031519288b5f2669c128349c7",
        },
        type="MultilabelClassification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="accuracy",
        date=("2024-01-01", "2025-01-01"),
        domains=["Social", "Written"],
        task_subtypes=["Emotion classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=["brazilian"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{muhammad-etal-2025-brighter,
  address = {Vienna, Austria},
  author = {Muhammad, Shamsuddeen Hassan and Ousidhoum, Nedjma and Abdulmumin, Idris and Wahle, Jan Philip and Ruas, Terry and Beloucif, Meriem and de Kock, Christine and Surange, Nirmal and Teodorescu, Daniela and Ahmad, Ibrahim Said and Adelani, David Ifeoluwa and Aji, Alham Fikri and Ali, Felermino D. M. A. and Alimova, Ilseyar and Araujo, Vladimir and Babakov, Nikolay and Baes, Naomi and Bucur, Ana-Maria and Bukula, Andiswa and Cao, Guanqun and Tufi{\~n}o, Rodrigo and Chevi, Rendi and Chukwuneke, Chiamaka Ijeoma and Ciobotaru, Alexandra and Dementieva, Daryna and Gadanya, Murja Sani and Geislinger, Robert and Gipp, Bela and Hourrane, Oumaima and Ignat, Oana and Lawan, Falalu Ibrahim and Mabuya, Rooweither and Mahendra, Rahmad and Marivate, Vukosi and Panchenko, Alexander and Piper, Andrew and Ferreira, Charles Henrique Porto and Protasov, Vitaly and Rutunda, Samuel and Shrivastava, Manish and Udrea, Aura Cristina and Wanzare, Lilian Diana Awuor and Wu, Sophie and Wunderlich, Florian Valentin and Zhafran, Hanif Muhammad and Zhang, Tianhui and Zhou, Yi and Mohammad, Saif M.},
  booktitle = {Proceedings of the 63rd Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/2025.acl-long.436},
  pages = {8895--8916},
  publisher = {Association for Computational Linguistics},
  title = {{BRIGHTER}: {BRI}dging the Gap in Human-Annotated Textual Emotion Recognition Datasets for 28 Languages},
  url = {https://aclanthology.org/2025.acl-long.436/},
  year = {2025},
}
""",
    )

    input_column_name = "text"
    label_column_name = "label"
