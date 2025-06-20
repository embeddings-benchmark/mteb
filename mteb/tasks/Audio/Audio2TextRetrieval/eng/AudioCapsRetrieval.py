from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from mteb.abstasks.Audio.AbsTaskAudio2TextRetrieval import AbsTaskAudio2TextRetrieval
from mteb.models.wrapper import Wrapper

class AudioCapsRetrieval(AbsTaskAudio2TextRetrieval):
    metadata = TaskMetadata(
        name="AudioCapsRetrieval",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on ARC-Challenge.",
        reference="https://allenai.org/data/arc",
        dataset={
            "path": "TwinkStart/AudioCaps",
            "revision": "8fc8b151149af779517aedfbf8c536160822bd70",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test[:%10]"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{xiao2024rar,
  title={RAR-b: Reasoning as Retrieval Benchmark},
  author={Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2404.06347},
  year={2024}
}
@article{clark2018think,
  title={Think you have solved question answering? try arc, the ai2 reasoning challenge},
  author={Clark, Peter and Cowhey, Isaac and Etzioni, Oren and Khot, Tushar and Sabharwal, Ashish and Schoenick, Carissa and Tafjord, Oyvind},
  journal={arXiv preprint arXiv:1803.05457},
  year={2018}
}
""",
        # prompt={"query": "Retrieve the answer to the question."},
    )

    audio_column_name: str = 'audio'
    text_column_name: str = 'caption'
    id_column_name: str = 'audiocap_id'

    def dataset_transform(self):
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split][:10]
