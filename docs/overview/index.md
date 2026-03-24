---
title: "Overview"
---

# Overview

This section provides an overview of available tasks, models and benchmarks in MTEB.


<div class="grid cards" style="grid-template-columns: repeat(3, 1fr);" markdown>

-   :lucide-square-stack: __Benchmarks__

    ---

    All the popular benchmarks for evaluating embeddings in one place

    [:lucide-corner-down-right: See benchmarks](./available_benchmarks.md)



-   :lucide-bot: __Models__

    ---

    Reproducle models implementations, for any modality and language.

    [:lucide-corner-down-right: See models](./available_models/text.md)

-   :lucide-square: __Tasks__

    ---

    Our comprehensive collection of tasks for evaluating embeddings

    [:lucide-corner-down-right: See tasks](./available_tasks/index.md)



</div>

# Models

<div class="grid cards" style="grid-template-columns: repeat(3, 1fr);" markdown>

-   :lucide-type: __Text__

    ---

    Models that only encode text into embeddings.

    [:lucide-corner-down-right: See text models](./available_models/text.md)

-   :lucide-image: __Image__

    ---

    Models that only encode images into embeddings.

    [:lucide-corner-down-right: See image models](./available_models/image.md)

-   :lucide-layout: __Image Text__

    ---

    Models that jointly encode images and text.

    [:lucide-corner-down-right: See image text models](./available_models/image_text.md)

-   :lucide-audio-lines: __Audio__

    ---

    Models that only encode audio into embeddings.

    [:lucide-corner-down-right: See audio models](./available_models/audio.md)

-   :lucide-layers: __Audio Text__

    ---

    Models that jointly encode audio and text.

    [:lucide-corner-down-right: See audio text models](./available_models/audio_text.md)

-   :lucide-combine: __Multimodal__

    ---

    Models that encode more than two modalities.

    [:lucide-corner-down-right: See multimodal models](./available_models/multimodal.md)

</div>


# Tasks

While MTEB covers multiple task types, we categorize them into 5 broad categories based on the type of evaluation they require, these categories are not mutually exclusive but provide a useful way to navigate the large collection of tasks in MTEB.

<div class="grid cards" style="grid-template-columns: repeat(3, 1fr);" markdown>

-   :lucide-shapes: __Classification__

    ---

    Embeddings linearly separable by category, evaluated using a classifier or regression probe.

    [:lucide-corner-down-right: See tasks](./available_tasks/classification.md)

-   :lucide-chart-network: __Clustering__

    ---

    Globally coherent embeddings where distances reflect semantic grouping.

    [:lucide-corner-down-right: See tasks](./available_tasks/clustering.md)

-   :lucide-git-compare: __Pair classification__

    ---

    Embeddings capturing relationships between item pairs, such as entailment or paraphrase.

    [:lucide-corner-down-right: See tasks](./available_tasks/pair_classification.md)

-   :lucide-search: __Retrieval__

    ---

    Asymmetric matching between queries and a corpus across different embedding regions.

    [:lucide-corner-down-right: See tasks](./available_tasks/retrieval.md)

-   :lucide-spline: __Semantic similarity__

    ---

    Fine-grained similarity between item pairs, where cosine similarity reflects human judgments.

    [:lucide-corner-down-right: See tasks](./available_tasks/semantic_similarity.md)

</div>