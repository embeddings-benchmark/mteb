from __future__ import annotations

import mteb

# missing
model = mteb.get_model("royokong/e5-v")
tasks = mteb.get_tasks(tasks=["EDIST2ITRetrieval", "Winoground"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder="results-mieb-final")

model = mteb.get_model("nyu-visionx/moco-v3-vit-b")
tasks = mteb.get_tasks(
    tasks=[
        "CIFAR10Clustering",
        "ImageNetDog15Clustering",
        "TinyImageNetClustering",
        "UCF101",
        "VOC2007",
    ]
)
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder="results-mieb-final")

model = mteb.get_model("nyu-visionx/moco-v3-vit-l")
tasks = mteb.get_tasks(
    tasks=[
        "SketchyI2IRetrieval",
        "CIFAR10Clustering",
        "ImageNetDog15Clustering",
        "TinyImageNetClustering",
        "UCF101",
        "VOC2007",
    ]
)
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder="results-mieb-final")

model = mteb.get_model("BAAI/bge-visualized-base")
tasks = mteb.get_tasks(
    tasks=[
        "SciMMIRI2TRetrieval",
        "SciMMIRT2IRetrieval",
        "VisualNewsI2TRetrieval",
        "WITT2IRetrieval",
    ]
)
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder="results-mieb-final")

model = mteb.get_model("TIGER-Lab/VLM2Vec-Full")
tasks = mteb.get_tasks(
    tasks=["CVBenchCount", "CVBenchDepth", "CVBenchDistance", "CVBenchRelation"]
)
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder="results-mieb-final")

model = mteb.get_model("TIGER-Lab/VLM2Vec-LoRA")
tasks = mteb.get_tasks(
    task_types=[
        "Any2AnyRetrieval",
        "Any2AnyMultiChoice",
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

# rerun
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
    "google/siglip-so400m-patch14-384",
    "google/siglip-base-patch16-256-multilingual",
    "google/siglip-base-patch16-256",
    "google/siglip-base-patch16-512",
    "google/siglip-base-patch16-384",
    "google/siglip-base-patch16-224",
    "google/siglip-large-patch16-256",
    "google/siglip-large-patch16-384",
    "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    "laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",
    "laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "laion/CLIP-ViT-g-14-laion2B-s34B-b88K",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "TIGER-Lab/VLM2Vec-LoRA",
    "TIGER-Lab/VLM2Vec-Full",
    "Salesforce/blip-itm-base-coco",
    "Salesforce/blip-itm-large-coco",
    "Salesforce/blip-itm-base-flickr",
    "Salesforce/blip-itm-large-flickr",
    "EVA02-CLIP-B-16",
    "EVA02-CLIP-L-14",
    "EVA02-CLIP-bigE-14",
    "EVA02-CLIP-bigE-14-plus",
]:
    model = mteb.get_model(model_name)
    tasks = mteb.get_tasks(
        tasks=[
            "ROxfordEasyI2IRetrieval",
            "ROxfordHardI2IRetrieval",
            "ROxfordMediumI2IRetrieval",
            "RParisEasyI2IRetrieval",
            "RParisHardI2IRetrieval",
            "RParisMediumI2IRetrieval",
        ]
    )
    # get i-only tasks for i-only models.
    if ("moco" in model_name) or ("dinov2" in model_name):
        tasks = [task for task in tasks if "t" not in task.metadata.category]

    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder="results-mieb-rerun")
