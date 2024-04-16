from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

class EstonianValenceSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="EstonianValenceSentimentClassification",
        description="An Estonian dataset for sentiment classification.",
        reference="https://figshare.com/articles/dataset/Estonian_Valence_Corpus_Eesti_valentsikorpus/24517054",
        dataset={
            "path": "kardosdrur/estonian-valence",
            "revision": "9157397f05a127b3ac93b93dd88abf1bdf710c22",
        },
        type="Classification",
        category="s2s",
        date=["2016-06-01", "2016-06-01"],
        eval_splits=["test"],
        eval_langs=["est-Latn"],
        main_score="accuracy",
        form=["written"],
        domains=["News"],
        task_subtypes=["Sentiment/Hate speech"],
        license="CC-BY-4.0",
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=[],
        text_creation="found",
        bibtex_citation="""@ARTICLE{Pajupuu2016-ci,
  title     = "Identifying polarity in different text types",
  author    = "Pajupuu, Hille and Altrov, Rene and Pajupuu, Jaan",
  abstract  = "While Sentiment Analysis aims to identify the writer's attitude
               toward individuals, events or topics, our aim is to predict the
               possible effect of a written text on the reader. For this
               purpose, we created an automatic identifier of the polarity of
               Estonian texts, which is independent of domain and of text type.
               Depending on the approach chosen -- lexicon-based or machine
               learning -- the identifier uses either a lexicon of words with a
               positive or negative connotation, or a text corpus where
               orthographic paragraphs have been annotated as positive,
               negative, neutral or mixed. Both approaches worked well,
               resulting in a nearly 75\% accuracy on average. It was found
               that in some cases the results depend on the text type, notably,
               with sports texts the lexicon-based approach yielded a maximum
               accuracy of 80.3\%, while over 88\% was gained for opinion
               stories approached by machine learning.",
  journal   = "Folk. Electron. J. Folk.",
  publisher = "Estonian Literary Museum of Scholarly Press",
  volume    =  64,
  pages     = "125--142",
  month     =  jun,
  year      =  2016
}
""",
        n_samples={"test": 818},
        avg_character_length={"test": 231.5},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("paragraph", "text")

        ## There are 4 unique labels.
        def map_valence_to_label_num(example):
            mapping = {'negatiivne': 1, 'neutraalne': 2, 'positiivne': 3, 'vastuoluline': 4}
            example['valence'] = mapping.get(example['valence'])

        self.dataset = self.dataset.map(map_valence_to_label_num)
        self.dataset = self.dataset.rename_column("valence", "label")
