from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining


class NorwegianCourtsBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="NorwegianCourtsBitextMining",
        dataset={
            "path": "mteb/NorwegianCourtsBitextMining",
            "revision": "ddf68e49e8d393964288ad5b498f1d61cca7b23f",
        },
        description="Nynorsk and Bokmål parallel corpus from Norwegian courts. Norwegian courts have two standardised written languages. Bokmål is a variant closer to Danish, while Nynorsk was created to resemble regional dialects of Norwegian.",
        reference="https://opus.nlpl.eu/index.php",
        type="BitextMining",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nob-Latn", "nno-Latn"],
        main_score="f1",
        date=("2020-01-01", "2020-12-31"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{opus4,
  author = {Tiedemann, J{\"o}rg and Thottingal, Santhosh},
  booktitle = {Proceedings of the 22nd Annual Conference of the European Association for Machine Translation (EAMT)},
  title = {OPUS-MT — Building open translation services for the World},
  year = {2020},
}
""",
        prompt="Retrieve parallel sentences in Norwegian Bokmål and Nynorsk",
    )
