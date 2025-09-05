from __future__ import annotations

from datasets import load_dataset

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ....abstasks.TaskMetadata import TaskMetadata

HF_REPO = "datalyes/DAPFAM_patent"
REFERENCE = "https://arxiv.org/abs/2506.22141"
BIBTEX = r"""
@misc{ayaou2025dapfamdomainawarefamilyleveldataset,
      title = {DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval},
      author = {Iliass Ayaou and Denis Cavallucci and Hicham Chibane},
      year = {2025},
      eprint = {2506.22141},
      archivePrefix = {arXiv},
      primaryClass = {cs.CL},
      url = {https://arxiv.org/abs/2506.22141},
}
"""

_SHARED_METADATA = dict(
    dataset={"path": HF_REPO, "revision": "3ad6eab6ed9b5fb1c0609b4dbf40e391ebb5a544"},
    reference=REFERENCE,
    type="Retrieval",
    category="p2p",
    task_subtypes=["Article retrieval", "Patent retrieval"],
    eval_splits=["test"],
    eval_langs=["eng-Latn"],
    main_score="ndcg_at_10",
    date=("1964-06-26", "2023-06-20"),
    domains=["Engineering", "Chemistry", "Legal"],
    license="cc-by-nc-sa-4.0",
    annotations_creators="derived",
    sample_creation="created",
    bibtex_citation=BIBTEX,
    judged_docs_only_flag=True,
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

# Text representations used in the paper
_IN_PAPER = {
    ("TitleAbstract", "TitleAbstractClaims"),
    ("TitleAbstractClaims", "TitleAbstractClaims"),
}


# MIX-IN with shared logic + metric implementation
class _DAPFAMMixin:
    # class-level attributes are filled in each concrete subclass
    domain_filter: str | None = None
    query_fields: list[str] = []
    corpus_fields: list[str] = []
    in_paper: bool = False

    def load_data(self, **_) -> tuple[dict, dict, dict]:
        ds_c = load_dataset(
            HF_REPO,
            "corpus",
            split="train",
            revision="3ad6eab6ed9b5fb1c0609b4dbf40e391ebb5a544",
        )
        ds_q = load_dataset(
            HF_REPO,
            "queries",
            split="train",
            revision="3ad6eab6ed9b5fb1c0609b4dbf40e391ebb5a544",
        )
        ds_r = load_dataset(
            HF_REPO,
            "relations",
            split="train",
            revision="3ad6eab6ed9b5fb1c0609b4dbf40e391ebb5a544",
        )

        self.corpus = {
            "test": {
                r["relevant_id"]: "\n".join(
                    str(r[f]) for f in self.corpus_fields if r.get(f)
                )
                for r in ds_c
            }
        }
        self.queries = {
            "test": {
                r["query_id"]: "\n".join(
                    str(r[f]) for f in self.query_fields if r.get(f)
                )
                for r in ds_q
            }
        }

        raw: dict[str, dict[str, tuple[float, str]]] = {}
        for r in ds_r:
            qid = r["query_id"]
            pid = r["relevant_id"]
            raw.setdefault(qid, {})[pid] = (
                float(r["relevance_score"]),
                r["domain_rel"],
            )
        self._qrels_raw = {"test": raw}

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

        self.relevant_docs = {"test": qrels_int}
        self.data_loaded = True
        return self.corpus, self.queries, self.relevant_docs


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
