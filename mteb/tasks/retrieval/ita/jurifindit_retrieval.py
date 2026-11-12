from __future__ import annotations

from typing import TYPE_CHECKING, Any

import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
    from mteb.timing import TimingStack

# The published corpus keeps one placeholder row per source act to anchor the
# document hierarchy. It carries no legal text and is never a relevant document,
# so it is dropped: removing the 159 placeholders leaves exactly the 23,458
# statutory articles reported in the paper.
_STRUCTURAL_PLACEHOLDER = "<< file root node >>"


class JuriFindITRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JuriFindITRetrieval",
        description="JuriFindIT is a native Italian statutory article retrieval dataset. The corpus holds statutory"
        " articles from Italian and EU legislative acts in Akoma Ntoso format, covering civil law, criminal law,"
        " anti-money laundering and counter-terrorism, and data protection. The queries are the 895 legal questions"
        " written by four Italian legal professionals (one per macro-area), each mapped to every article of the corpus"
        " that answers it. The dataset also ships 169,301 LLM-generated questions, which are deliberately excluded here"
        " so the task stays fully human-authored. Exact-duplicate articles (boilerplate provisions such as entry-into-force"
        " clauses repeated across acts) are collapsed onto a single document id so that retrieving an identical twin of a"
        " relevant article is not scored as an error.",
        reference="https://aclanthology.org/2026.findings-eacl.221/",
        dataset={
            "path": "jurifindit/JuriFindIT",
            "revision": "d765a61998a0dc962a9c06394ab546b5a5ee0aa5",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ita-Latn"],
        main_score="ndcg_at_10",
        # oldest legislative act in the corpus -- release of the dataset on the Hugging Face hub
        date=("1930-10-19", "2025-10-08"),
        domains=["Legal", "Government", "Written"],
        task_subtypes=["Article retrieval", "Question answering"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        # the corpus is found (published legislation), while the queries were written from
        # scratch by legal professionals
        sample_creation="created",
        prompt={
            "query": "Given a legal question in Italian, retrieve the statutory articles that answer it."
        },
        bibtex_citation=r"""
@inproceedings{noce-etal-2026-jurifindit,
  address = {Rabat, Morocco},
  author = {Noce, Niko Dalla and Colla, Davide and Doust, Sina Farhang and De Mattei, Lorenzo and Bacciu, Davide},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2026},
  doi = {10.18653/v1/2026.findings-eacl.221},
  editor = {Demberg, Vera and Inui, Kentaro and Marquez, Llu{\'i}s},
  isbn = {979-8-89176-386-9},
  month = mar,
  pages = {4223--4241},
  publisher = {Association for Computational Linguistics},
  title = {{JuriFindIT}: an {I}talian legal retrieval dataset},
  url = {https://aclanthology.org/2026.findings-eacl.221/},
  year = {2026},
}
""",
    )

    def load_data(
        self,
        num_proc: int | None = None,
        *,
        timer: TimingStack | None = None,
        **kwargs: Any,
    ) -> None:
        """Reshape the JuriFindIT corpus and expert questions into a retrieval task.

        The upstream dataset ships the corpus and the questions as separate configs, and
        the questions carry no query id, so ids are derived from the origin split. The
        single `test` split holds all 895 expert-written questions, the upstream `train`
        and `validation` splits merged: `mteb` evaluation is zero-shot, so nothing is fit
        on the upstream `train` questions, and merging them gives a query set large enough
        for stable nDCG estimates. The 179 questions the paper evaluates on keep their
        `validation-` id prefix, so its numbers can be reproduced from this split alone.
        """
        if self.data_loaded:
            return

        corpus_raw = datasets.load_dataset(
            name="corpus", split="corpus", **self.metadata.dataset
        )
        questions_raw = datasets.load_dataset(name="questions", **self.metadata.dataset)

        corpus, canonical_doc_ids = self._build_corpus(corpus_raw)

        queries: list[dict[str, str]] = []
        relevant_docs: dict[str, dict[str, int]] = {}
        for origin in ("train", "validation"):
            for index, row in enumerate(questions_raw[origin]):
                query_id = f"{origin}-{index}"
                queries.append({"id": query_id, "text": row["question"]})
                relevant_docs[query_id] = {
                    canonical_doc_ids[doc_id]: 1 for doc_id in row["relevant_doc_ids"]
                }
        split_data: RetrievalSplitData = {
            "corpus": corpus,
            "queries": datasets.Dataset.from_list(queries),
            "relevant_docs": relevant_docs,
            "top_ranked": None,
        }
        self.dataset = {"default": {"test": split_data}}

        self.data_loaded = True

    @staticmethod
    def _build_corpus(
        corpus_raw: datasets.Dataset,
    ) -> tuple[datasets.Dataset, dict[int, str]]:
        """Drop structural placeholders and collapse exact-duplicate articles.

        Returns the deduplicated corpus together with a mapping from every upstream
        article id to the document id that represents it, so that the qrels still
        resolve after deduplication.
        """
        documents: list[dict[str, str]] = []
        first_id_for_text: dict[str, str] = {}
        canonical_doc_ids: dict[int, str] = {}

        for doc_id, content in zip(
            corpus_raw["id"], corpus_raw["content"], strict=True
        ):
            text = content.strip()
            if text == _STRUCTURAL_PLACEHOLDER:
                continue
            canonical_id = first_id_for_text.get(text)
            if canonical_id is None:
                canonical_id = str(doc_id)
                first_id_for_text[text] = canonical_id
                documents.append({"id": canonical_id, "text": text})
            canonical_doc_ids[doc_id] = canonical_id

        return datasets.Dataset.from_list(documents), canonical_doc_ids
