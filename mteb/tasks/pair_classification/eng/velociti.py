from __future__ import annotations

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class VELOCITIPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="VELOCITIPairClassification",
        description=(
            "Pair classification on VELOCITI: determining whether a text caption "
            "correctly entails the events in a video or is a compositional negative "
            "constructed via in-video negation (swapping agents/actions occurring in "
            "the same video) or text-inspired negation (LLM-generated contradictions). "
            "Tests fine-grained video-language compositional binding of agents to actions "
            "across events."
        ),
        reference="https://arxiv.org/abs/2406.10889",
        dataset={
            "path": "yaswanth169/VELOCITI-PC",
            "revision": "509144fa4727a3f9b92602d491725154c32a958f",
        },
        type="VideoPairClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2024-06-16", "2024-06-16"),
        domains=["Activity", "Web"],
        task_subtypes=["Caption Pairing"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="LM-generated and verified",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{saravanan2025velociti,
  author = {Saravanan, Darshana and Gupta, Varun and Singh, Darshan and Khan, Zeeshan and Gandhi, Vineet and Tapaswi, Makarand},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  title = {VELOCITI: Benchmarking Video-Language Compositional Reasoning with Strict Entailment},
  year = {2025},
}
""",
    )

    input1_column_name = {"video": "video"}
    input2_column_name = {"text": "text"}
    label_column_name: str = "label"
