from typing import Any

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

HF_REPO = "mteb/DAPFAM_patent"
REFERENCE = "https://arxiv.org/abs/2506.22141"
BIBTEX = r"""@article{ayaou2025dapfamdomainawarefamilyleveldataset,
  author = {Ayaou, Iliass and Cavallucci, Denis and Chibane, Hicham},
  doi = {10.1016/j.array.2026.100720},
  issn = {2590-0056},
  journal = {Array},
  month = {Mar},
  pages = {100720},
  publisher = {Elsevier BV},
  title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
  url = {https://doi.org/10.1016/j.array.2026.100720},
  volume = {29},
  year = {2026},
}"""

_SHARED_METADATA = dict(
    dataset={"path": HF_REPO, "revision": "8200e3c1a58522df3592ace8d015c3b1e9a0ee0e"},
    reference=REFERENCE,
    type="Retrieval",
    modalities=["text"],
    category="t2t",
    task_subtypes=["Article retrieval", "Patent retrieval"],
    eval_splits=["train"],
    eval_langs=["eng-Latn"],
    main_score="ndcg_at_100",
    date=("1964-06-26", "2023-06-20"),
    domains=["Engineering", "Chemistry", "Legal"],
    license="cc-by-nc-sa-4.0",
    annotations_creators="derived",
    sample_creation="created",
    dialect=[],
    bibtex_citation=BIBTEX,
    is_public=True,
)

# text-field dictionaries
_QUERY_FIELDS = {
    "TitleAbstract": ["title_en", "abstract_en"],
    "TitleAbstractClaims": ["title_en", "abstract_en", "claims_text"],
}
_CORPUS_FIELDS = {
    "TitleAbstract": ["title_en", "abstract_en"],
    "TitleAbstractClaims": ["title_en", "abstract_en", "claims_text"],
    "TitleAbstractClaimsDescription": [
        "title_en",
        "abstract_en",
        "claims_text",
        "description_en",
    ],
}


def _format_patent(
    row: dict[str, Any], fields: list[str], id_column: str, is_corpus: bool
) -> dict[str, Any]:
    if is_corpus:
        return {
            "id": row[id_column],
            "title": row["title_en"],
            "text": "\n".join(
                str(row[field])
                for field in fields
                if row.get(field) and field != "title_en"
            ),
        }
    return {
        "id": row[id_column],
        "text": "\n".join(str(row[field]) for field in fields if row.get(field)),
    }


# MIX-IN with shared logic
class _DAPFAMMixin:
    # class-level attributes are filled in each concrete subclass
    domain_filter: str | None = None
    query_fields: list[str] = []  # noqa: RUF012
    corpus_fields: list[str] = []  # noqa: RUF012

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        ds_c = load_dataset(
            self.metadata.dataset["path"],
            "corpus",
            split=self.metadata.eval_splits[0],
            revision=self.metadata.dataset["revision"],
        )
        ds_q = load_dataset(
            self.metadata.dataset["path"],
            "queries",
            split=self.metadata.eval_splits[0],
            revision=self.metadata.dataset["revision"],
        )
        ds_r = load_dataset(
            self.metadata.dataset["path"],
            "relations",
            split=self.metadata.eval_splits[0],
            revision=self.metadata.dataset["revision"],
        )

        # Deterministic maps reuse the same Arrow cache for equal field selections
        # across the All/In/Out tasks, without eighteen separate Hub uploads.
        corpus = ds_c.select_columns(["relevant_id", *self.corpus_fields])
        corpus = corpus.map(
            _format_patent,
            fn_kwargs={
                "fields": self.corpus_fields,
                "id_column": "relevant_id",
                "is_corpus": True,
            },
            remove_columns=corpus.column_names,
            num_proc=num_proc,
        )
        queries = ds_q.select_columns(["query_id", *self.query_fields])
        queries = queries.map(
            _format_patent,
            fn_kwargs={
                "fields": self.query_fields,
                "id_column": "query_id",
                "is_corpus": False,
            },
            remove_columns=queries.column_names,
            num_proc=num_proc,
        )

        raw: dict[str, dict[str, tuple[float, str]]] = {}
        for r in ds_r:
            qid = r["query_id"]
            pid = r["relevant_id"]
            raw.setdefault(qid, {})[pid] = (
                float(r["relevance_score"]),
                r["domain_rel"],
            )
        self._qrels_raw = {"train": raw}

        qrels_int: dict[str, dict[str, int]] = {}
        for qid, pairs in raw.items():
            if self.domain_filter is None:
                pos = {pid: 1 for pid, (s, dom) in pairs.items() if s > 0.0}
            else:
                pos = {
                    pid: 1
                    for pid, (s, dom) in pairs.items()
                    if s > 0.0 and dom == self.domain_filter
                }
            if pos:
                qrels_int[qid] = pos

        self.dataset = {
            "default": {
                "train": RetrievalSplitData(
                    corpus=corpus,
                    queries=queries,
                    relevant_docs=qrels_int,
                    top_ranked=None,
                )
            }
        }
        self.data_loaded = True


# ───────────────────────────────────────────────────
# DAPFAM Patent Family Retrieval Tasks


class DAPFAMAllTitlAbsToTitlAbsRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = None
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstract"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMAllTitlAbsToTitlAbsRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title and Abstract, "
            "and target patent families are represented by Title and Abstract. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, no International Patent Classification-based filtering is applied. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to retrieve citation-linked patent families using query and target patent family representations of Title and Abstract across all technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMAllTitlAbsToTitlAbsClmRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = None
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaims"]
    in_paper = True
    metadata = TaskMetadata(
        name="DAPFAMAllTitlAbsToTitlAbsClmRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title and Abstract, "
            "and target patent families are represented by Title, Abstract, and Claims. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, no International Patent Classification-based filtering is applied. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to assess how adding Claims text to target patent family representations improves retrieval of citation-linked patent families across all technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMAllTitlAbsToFullTextRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = None
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaimsDescription"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMAllTitlAbsToFullTextRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title and Abstract, "
            "and target patent families are represented by Title, Abstract, Claims, and Description. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, no International Patent Classification-based filtering is applied. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to evaluate retrieval performance using Title and Abstract query patent family representations and full-text target patent family representations across all technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMAllTitlAbsClmToTitlAbsRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = None
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstract"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMAllTitlAbsClmToTitlAbsRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, "
            "and target patent families are represented by Title and Abstract. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, no International Patent Classification-based filtering is applied. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to measure the effect of Claims-augmented query patent family representations when targets are limited to Title and Abstract across all technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMAllTitlAbsClmToTitlAbsClmRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = None
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaims"]
    in_paper = True
    metadata = TaskMetadata(
        name="DAPFAMAllTitlAbsClmToTitlAbsClmRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, "
            "and target patent families are represented by Title, Abstract, and Claims. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, no International Patent Classification-based filtering is applied. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to evaluate retrieval when both query and target patent families use Claims-augmented representations across all technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMAllTitlAbsClmToFullTextRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = None
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaimsDescription"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMAllTitlAbsClmToFullTextRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, "
            "and target patent families are represented by Title, Abstract, Claims, and Description. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, no International Patent Classification-based filtering is applied. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to evaluate retrieval performance using Claims-augmented query patent family representations full-text target patent family representations across all technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMInTitlAbsToTitlAbsRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "IN"
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstract"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMInTitlAbsToTitlAbsRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title and Abstract, "
            "and target patent families are represented by Title and Abstract. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to retrieve citation-linked patent families using query and target patent family representations of Title and Abstract within the same technical domain."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMInTitlAbsToTitlAbsClmRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "IN"
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaims"]
    in_paper = True
    metadata = TaskMetadata(
        name="DAPFAMInTitlAbsToTitlAbsClmRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title and Abstract, "
            "and target patent families are represented by Title, Abstract, and Claims. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to assess how adding Claims text to target patent family representations improves retrieval of citation-linked patent families within the same technical domain."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMInTitlAbsToFullTextRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "IN"
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaimsDescription"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMInTitlAbsToFullTextRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title and Abstract, "
            "and target patent families are represented by Title, Abstract, Claims, and Description. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to evaluate retrieval performance using Title and Abstract query patent family representations and full-text target patent family representations within the same technical domain."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMInTitlAbsClmToTitlAbsRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "IN"
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstract"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMInTitlAbsClmToTitlAbsRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, "
            "and target patent families are represented by Title and Abstract. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to measure the effect of Claims-augmented query patent family representations when targets are limited to Title and Abstract within the same technical domain."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMInTitlAbsClmToTitlAbsClmRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "IN"
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaims"]
    in_paper = True
    metadata = TaskMetadata(
        name="DAPFAMInTitlAbsClmToTitlAbsClmRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, "
            "and target patent families are represented by Title, Abstract, and Claims. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to evaluate retrieval when both query and target patent families use Claims-augmented representations within the same technical domain."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMInTitlAbsClmToFullTextRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "IN"
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaimsDescription"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMInTitlAbsClmToFullTextRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, "
            "and target patent families are represented by Title, Abstract, Claims, and Description. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing at least one three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to evaluate retrieval performance using Claims-augmented query patent family representations full-text target patent family representations within the same technical domain."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMOutTitlAbsToTitlAbsRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "OUT"
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstract"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMOutTitlAbsToTitlAbsRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title and Abstract, "
            "and target patent families are represented by Title and Abstract. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing no three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to retrieve citation-linked patent families using query and target patent family representations of Title and Abstract across different technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMOutTitlAbsToTitlAbsClmRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "OUT"
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaims"]
    in_paper = True
    metadata = TaskMetadata(
        name="DAPFAMOutTitlAbsToTitlAbsClmRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title and Abstract, "
            "and target patent families are represented by Title, Abstract, and Claims. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing no three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to assess how adding Claims text to target patent family representations improves retrieval of citation-linked patent families across different technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMOutTitlAbsToFullTextRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "OUT"
    query_fields = _QUERY_FIELDS["TitleAbstract"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaimsDescription"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMOutTitlAbsToFullTextRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title and Abstract, "
            "and target patent families are represented by Title, Abstract, Claims, and Description. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing no three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to evaluate retrieval performance using Title and Abstract query patent family representations and full-text target patent family representations across different technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMOutTitlAbsClmToTitlAbsRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "OUT"
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstract"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMOutTitlAbsClmToTitlAbsRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, "
            "and target patent families are represented by Title and Abstract. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing no three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to measure the effect of Claims-augmented query patent family representations when targets are limited to Title and Abstract across different technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMOutTitlAbsClmToTitlAbsClmRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "OUT"
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaims"]
    in_paper = True
    metadata = TaskMetadata(
        name="DAPFAMOutTitlAbsClmToTitlAbsClmRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, "
            "and target patent families are represented by Title, Abstract, and Claims. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing no three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to evaluate retrieval when both query and target patent families use Claims-augmented representations across different technical domains."
        ),
        **_SHARED_METADATA,
    )


class DAPFAMOutTitlAbsClmToFullTextRetrieval(_DAPFAMMixin, AbsTaskRetrieval):
    domain_filter = "OUT"
    query_fields = _QUERY_FIELDS["TitleAbstractClaims"]
    corpus_fields = _CORPUS_FIELDS["TitleAbstractClaimsDescription"]
    in_paper = False
    metadata = TaskMetadata(
        name="DAPFAMOutTitlAbsClmToFullTextRetrieval",
        description=(
            "In this patent family retrieval task, query patent families are represented by Title, Abstract, and Claims, "
            "and target patent families are represented by Title, Abstract, Claims, and Description. "
            "Relevant target families have a citation link (cited or citing) with the query family. "
            "Additionally, only targets sharing no three-character International Patent Classification code with the query family. "
            "Relevance and labelling scheme are described in detail in Section 3.4 and 3.5 of Ayaou et al. (2025), arXiv:2506.22141."
            "Patents are aggregated and represented at the family level to reduce redundancy across jurisdictions. "
            "The goal of the task is to evaluate retrieval performance using Claims-augmented query patent family representations full-text target patent family representations across different technical domains."
        ),
        **_SHARED_METADATA,
    )
