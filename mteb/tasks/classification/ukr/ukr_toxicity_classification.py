from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

class UkrTweetToxicityBinaryClassification(AbsTaskClassification):
    input_column_name = "text"
    label_column_name = "toxic"
    train_split = 'train'
    metadata = TaskMetadata(
        name="UkrTweetToxicityBinaryClassification",
        description="Filtered Ukrainian Tweets dataset with Toloka.ai platform crowdsourcing task. It contains of 2.5k toxic and 2.5k non-toxic texts.",
        reference="https://huggingface.co/datasets/ukr-detect/ukr-toxicity-dataset",
        type="Classification",
        category='t2c',
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ukr-Cyrl"],
        main_score="accuracy",
        dataset={
            "path": "mteb/UkrTweetToxicityBinaryClassification_ukr-detect_ukr-toxicity-dataset",
            "revision": "d2638cc529d5d50cecc91b7ae21735fab7b34743",
        },
        date=("2019-01-01", "2025-12-03"),
        domains=["Blog", "Social", "Web"],
        task_subtypes=['Sentiment/Hate speech'],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
@inproceedings{dementieva-etal-2024-toxicity,
    title = "Toxicity Classification in {U}krainian",
    author = "Dementieva, Daryna  and
      Khylenko, Valeriia  and
      Babakov, Nikolay  and
      Groh, Georg",
    editor = {Chung, Yi-Ling  and
      Talat, Zeerak  and
      Nozza, Debora  and
      Plaza-del-Arco, Flor Miriam  and
      R{\"o}ttger, Paul  and
      Mostafazadeh Davani, Aida  and
      Calabrese, Agostina},
    booktitle = "Proceedings of the 8th Workshop on Online Abuse and Harms (WOAH 2024)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.woah-1.19",
    doi = "10.18653/v1/2024.woah-1.19",
    pages = "244--255",
}
"""
)