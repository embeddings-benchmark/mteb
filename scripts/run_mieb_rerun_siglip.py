from __future__ import annotations

import mteb

for model_name in [
    "google/siglip-so400m-patch14-384",
    "google/siglip-base-patch16-256-multilingual",
    "google/siglip-base-patch16-256",
    "google/siglip-base-patch16-512",
    "google/siglip-base-patch16-384",
    "google/siglip-base-patch16-224",
    "google/siglip-large-patch16-256",
    "google/siglip-large-patch16-384",
]:
    model = mteb.get_model(model_name)
    tasks = mteb.get_tasks(
        task_types=[
            "Any2AnyRetrieval",
            "Any2AnyMultiChoice",
            "VisionCentricQA",
            "ImageClustering",
            "ImageClassification",
            "ImageMultilabelClassification",
            "Compositionality",
            # "VisualSTS",  # visual sts does not need rerun as will be the same after fixed.
            "ZeroShotClassification",
            "DocumentUnderstanding",
        ]
    )
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder="results-mieb-final/siglip_rerun")
