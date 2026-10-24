from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class InsAVE80KVT2VRetrieval(AbsTaskRetrieval):
    """Instruction-conditioned video+text to video retrieval derived from InsAVE-80K."""

    metadata = TaskMetadata(
        name="InsAVE80KVT2VRetrieval",
        description=(
            "Instruction-conditioned composed audio-video retrieval built from the "
            "held-out evaluation split of InsAVE-80K, the audio-video editing dataset "
            "released with InstructAV2AV. Each query pairs a source clip with the "
            "natural-language editing instruction that was applied to it, and the "
            "relevant document is the corresponding edited clip. The candidate pool is "
            "the complete set of evaluation clips, so every query's own source clip "
            "stays in the pool as a hard negative and the instruction is required to "
            "prefer the edited clip over the unedited one. Clips carry synchronised "
            "audio, and instructions may contain <S>/<E> markers delimiting intended "
            "spoken content. This is a new MOEB retrieval construction derived from the "
            "official 1,000-pair generation-evaluation split: InstructAV2AV scores "
            "generation quality on these pairs and defines no retrieval evaluation, so "
            "scores here are not comparable to any number reported in the paper."
        ),
        reference="https://arxiv.org/abs/2605.18467",
        dataset={
            "path": "myang333/InsAVE80KVT2VRetrieval",
            "revision": "8bf85ebf9183bde4217688c3cc1ba0e8de4b95b7",
        },
        type="Any2AnyRetrieval",
        category="vt2v",
        modalities=["video", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2026-05-18", "2026-07-28"),
        domains=["Scene", "Spoken", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="automatic-and-reviewed",
        dialect=[],
        sample_creation="multiple",
        bibtex_citation=r"""
@article{zheng2026instructav2av,
  author = {Zheng, Haojie and Yang, Yixin and Yang, Siqi and Weng, Shuchen and Shi, Boxin},
  journal = {arXiv preprint arXiv:2605.18467},
  title = {InstructAV2AV: Instruction-Guided Audio-Video Joint Editing},
  year = {2026},
}
""",
        prompt={
            "query": "Given a source video and an editing instruction, retrieve the video that results from applying the instruction to the source video."
        },
        is_beta=True,
    )
