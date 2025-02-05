from __future__ import annotations

import mteb

for model_name in [
    # key ones for this ablation (different types of models)
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-large-patch14",
    "royokong/e5-v",
    "facebook/dinov2-small",
    "facebook/dinov2-base",
    "facebook/dinov2-large",
    "facebook/dinov2-giant",
    # more insights
    "BAAI/bge-visualized-base",
    "BAAI/bge-visualized-m3",
    "google/siglip-so400m-patch14-384",
    "google/siglip-base-patch16-256-multilingual",
    "google/siglip-base-patch16-256",
    "google/siglip-base-patch16-512",
    "google/siglip-base-patch16-384",
    "google/siglip-base-patch16-224",
    "google/siglip-large-patch16-256",
    "google/siglip-large-patch16-384",
    "nyu-visionx/moco-v3-vit-b",
    "nyu-visionx/moco-v3-vit-l",
    "laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",
    "laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    "laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",
    "laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "laion/CLIP-ViT-g-14-laion2B-s34B-b88K",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "EVA02-CLIP-B-16",
    "EVA02-CLIP-L-14",
    "EVA02-CLIP-bigE-14",
    "EVA02-CLIP-bigE-14-plus",
    "TIGER-Lab/VLM2Vec-LoRA",
    "TIGER-Lab/VLM2Vec-Full",
    # run if enough compute:
    # "Salesforce/blip-itm-base-coco",
    # "Salesforce/blip-itm-large-coco",
    # "Salesforce/blip-itm-base-flickr",
    # "Salesforce/blip-itm-large-flickr",
    # "kakaobrain/align-base",
    # "jinaai/jina-clip-v1",
    # "nomic-ai/nomic-embed-vision-v1.5",
    # "Salesforce/blip2-opt-2.7b",
    # "Salesforce/blip2-opt-6.7b-coco",
    # "embed-english-v3.0-v",  # not feasible to run due to the 40 images/min constraint
]:
    # 16 by default already

    for k_shot in [8, 32, 64, 128, 256]:
        model = mteb.get_model(model_name)
        tasks = mteb.get_tasks(
            task_types=[
                "ImageClassification",
            ]
        )
        for task in tasks:
            task.samples_per_label = k_shot
        evaluation = mteb.MTEB(tasks=tasks)
        results = evaluation.run(
            model, output_folder=f"results-mieb-final/linear_probe_{k_shot}"
        )
