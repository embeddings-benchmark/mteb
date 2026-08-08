from __future__ import annotations

from collections import defaultdict

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class NSynthInstrumentFamilyA2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NSynthInstrumentFamilyA2ARetrieval",
        description=(
            "Audio-to-audio instrument family retrieval built from the NSynth test "
            "split: given a single musical note (query), retrieve notes played by "
            "instruments of the same instrument family (bass, brass, flute, guitar, "
            "keyboard, mallet, organ, reed, string, vocal). Queries and corpus are "
            "instrument-disjoint: within each family the individual instruments are "
            "sorted by name and assigned alternately to the query side and the corpus "
            "side, so no instrument contributes clips to both sides. A model therefore "
            "cannot succeed by recognising the exact same instrument and must "
            "generalise across timbres within a family."
        ),
        reference="https://arxiv.org/abs/1704.01279",
        dataset={
            "path": "mteb/nsynth-mini",
            "revision": "e32dfe9b65e121e64229a821fe1ff177e8962635",
        },
        type="Any2AnyRetrieval",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2017-04-05", "2017-04-05"),
        domains=["Music"],
        task_subtypes=["Music Instrument Recognition"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        adapted_from=["NSynth"],
        is_beta=True,
        prompt={
            "query": "Retrieve notes played by instruments of the same instrument family as the given audio clip."
        },
        bibtex_citation=r"""
@misc{engel2017neuralaudiosynthesismusical,
  archiveprefix = {arXiv},
  author = {Jesse Engel and Cinjon Resnick and Adam Roberts and Sander Dieleman and Douglas Eck and Karen Simonyan and Mohammad Norouzi},
  eprint = {1704.01279},
  primaryclass = {cs.LG},
  title = {Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders},
  url = {https://arxiv.org/abs/1704.01279},
  year = {2017},
}
""",
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return

        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            ds = load_dataset(**self.metadata.dataset, split=split)

            # Group the row indices by instrument family, then by the individual
            # instrument. Only the label columns are touched, the audio column is
            # never materialised here.
            grouped: dict[str, dict[str, list[int]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for idx, (family, instrument) in enumerate(
                zip(ds["instrument_family_str"], ds["instrument_str"])
            ):
                grouped[family][instrument].append(idx)

            # Deterministic instrument-disjoint split. Within every family the
            # instruments are sorted lexicographically (NSynth ids are zero-padded,
            # so this matches numeric order) and assigned alternately: even
            # position -> query side, odd position -> corpus side. Whole
            # instruments are assigned, so the notes of one instrument never end
            # up on both sides. No randomness is involved.
            note_ids = ds["note_str"]
            query_indices: list[int] = []
            corpus_indices: list[int] = []
            query_ids_per_family: dict[str, list[str]] = defaultdict(list)
            corpus_ids_per_family: dict[str, list[str]] = defaultdict(list)
            for family in sorted(grouped):
                for position, instrument in enumerate(sorted(grouped[family])):
                    indices = grouped[family][instrument]
                    if position % 2 == 0:
                        query_indices.extend(indices)
                        query_ids_per_family[family].extend(
                            note_ids[i] for i in indices
                        )
                    else:
                        corpus_indices.extend(indices)
                        corpus_ids_per_family[family].extend(
                            note_ids[i] for i in indices
                        )

            # Relevance is instrument-family equality, so each query is relevant to
            # every corpus clip of its own family (full cross-product, score 1).
            relevant_docs = {
                query_id: dict.fromkeys(corpus_ids_per_family[family], 1)
                for family, query_ids in query_ids_per_family.items()
                for query_id in query_ids
            }

            # note_str is NSynth's unique per-clip identifier.
            ds = ds.rename_column("note_str", "id")
            self.dataset["default"][split] = RetrievalSplitData(
                queries=ds.select(query_indices).select_columns(["id", "audio"]),
                corpus=ds.select(corpus_indices).select_columns(["id", "audio"]),
                relevant_docs=relevant_docs,
                top_ranked=None,
            )

        self.data_loaded = True
