from __future__ import annotations

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class VoyageMMarcoReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="VoyageMMarcoReranking",
        description="mMARCO is a multilingual version of the MS MARCO passage ranking dataset",
        reference="https://github.com/unicamp-dl/mMARCO",
        dataset={
            "path": "C-MTEB/Mmarco-reranking",
            "revision": "8e0c766dbe9e16e1d221116a3f36795fbade07f6",
        },
        type="Reranking",
        category="s2s",
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="map",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@misc{bonifacio2021mmarco,
      title={mMARCO: A Multilingual Version of MS MARCO Passage Ranking Dataset}, 
      author={Luiz Henrique Bonifacio and Vitor Jeronymo and Hugo Queiroz Abonizio and Israel Campiotti and Marzieh Fadaee and  and Roberto Lotufo and Rodrigo Nogueira},
      year={2021},
      eprint={2108.13897},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        n_samples=None,
        avg_character_length=None,
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )
        self.dataset = self.dataset.rename_column(
            "positives", "positive"
        ).rename_column("negatives", "negative")
        self.dataset["test"] = self.dataset.pop("train")
