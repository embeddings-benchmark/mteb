from __future__ import annotations

import mteb

for model_name in [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-large-patch14",
    "royokong/e5-v",
    "BAAI/bge-visualized-base",
    "BAAI/bge-visualized-m3",
    "kakaobrain/align-base",
    "jinaai/jina-clip-v1",
    "nomic-ai/nomic-embed-vision-v1.5",
    "Salesforce/blip-image-captioning-large",
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip2-opt-2.7b",
    "Salesforce/blip2-opt-6.7b-coco",
    "facebook/dinov2-small",
    "facebook/dinov2-base",
    "facebook/dinov2-large",
    "facebook/dinov2-giant",
    "laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",
    "laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    "nyu-visionx/moco-v3-vit-b",
    "nyu-visionx/moco-v3-vit-l",
    # "google/siglip-so400m-patch14-384",# haven't pushed
]:
    model = mteb.get_model(model_name)
    tasks = mteb.get_tasks(
        task_types=[
            "Any2AnyRetrieval",
            "AbsTaskAny2AnyMultiChoice",
            "Any2TextMutipleChoice",
            "ImageClustering",
            "ImageClassification",
            "ImageMultilabelClassification",
            "ImageTextPairClassification",
            "VisualSTS",
            "ZeroShotClassification",
        ]
    )
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder="results-mieb-final")
