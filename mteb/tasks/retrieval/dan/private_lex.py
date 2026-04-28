import mteb
from mteb.abstasks.retrieval import AbsTaskRetrieval


class GatedSyntheticLEXRetrieval(AbsTaskRetrieval):
    metadata = mteb.TaskMetadata(  # minimal metadata
        name="GatedSyntheticLEXRetrieval",
        description="Synthetically generated dataset for LEX retrieval task.",
        category="t2t",
        reference="https://huggingface.co/datasets/chcaa/lex_synth_retrieval_eval_v2",
        main_score="ndcg_at_10",
        eval_langs=["dan-Latn"],
        eval_splits=["test"],
        modalities=["text"],
        type="Retrieval",
        domains=["Non-fiction", "Encyclopaedic", "Constructed"],
        license="not specified",
        date=("2009-01-28", "2025-08-12"),
        dataset={
            "path": "mteb/GatedSyntheticLEXRetrieval",
            "revision": "b5b22dee3a30bb32ac579c14afe9eb152e8c70c7",
        },
        annotations_creators="LM-generated",
        dialect=[],
        prompt="Given a question in Danish, retrieve the  documents that can answer the question.",
        task_subtypes=["Article retrieval"],
        sample_creation="LM-generated and verified",
        bibtex_citation=r"""@misc{private-lex,
  author = {Aarhus University},
  title = {Private Synthetic LEX Retrieval Evaluation Dataset},
  year = {2026},
}
""",
    )
