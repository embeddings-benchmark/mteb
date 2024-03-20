from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class QuoraPLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Quora-PL",
        description="QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.",
        reference="https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs",
        hf_hub_name="clarin-knext/quora-pl",
        type="Retrieval",
        category="s2s",
        eval_splits=["validation", "test"],  # validation for new DataLoader
        eval_langs=["pl"],
        main_score="ndcg_at_10",
        revision="0be27e93455051e531182b85e85e425aba12e9d4",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
