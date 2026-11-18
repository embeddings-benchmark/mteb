from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata

_REPO = "MTEB-BR/juristcu-clustering"
_REVISION = "f933f2e7b8186ecafe65b79b246099e63d33eecb"


class JurisTCUClusteringP2P(AbsTaskClustering):
    """Cluster TCU jurisprudence excerpts (EXCERTO) into 10 legal AREA classes."""

    metadata = TaskMetadata(
        name="JurisTCUClusteringP2P",
        description=(
            "Cluster Brazilian Federal Court of Accounts (TCU) jurisprudence "
            "excerpts into 10 legal areas (Pessoal, Licitação, Responsabilidade, "
            "Direito Processual, Contrato Administrativo, Convênio, Competência "
            "do TCU, Finanças Públicas, Gestão Administrativa, Desestatização). "
            "Documents are the EXCERTO field of the JurisTCU corpus, labelled by "
            "TCU's own AREA taxonomy."
        ),
        reference="https://huggingface.co/datasets/MTEB-BR/juristcu-clustering",
        dataset={
            "path": _REPO,
            "revision": _REVISION,
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="v_measure",
        date=("2026-06-24", "2026-06-24"),
        domains=["Legal", "Government", "Written"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="found",
        adapted_from=["JurisTCU"],
        bibtex_citation=r"""
@article{juristcu2026,
  author = {Fernandes, Leandro Car{\'i}sio and
                 Ribeiro, Leandro dos Santos and
                 de Castro, Marcos Vin{\'i}cius Borela and
                 da Silva Pacheco, Leonardo Augusto and
                 de Oliveira Sandes, Edans Fl{\'a}vius},
  title = {{JurisTCU: a Brazilian Portuguese information retrieval dataset with query relevance judgments}},
  journal = {Language Resources and Evaluation},
  year = {2026},
  volume = {60},
  number = {1},
  doi = {10.1007/s10579-025-09881-w},
  url = {https://doi.org/10.1007/s10579-025-09881-w},
  issn = {1574-0218},
}
""",
    )

    input_column_name = "sentences"
    label_column_name = "label"
    # the datasets are small (<= ~5k documents): embed all of them instead of the default 4% sample
    max_fraction_of_documents_to_embed = None
