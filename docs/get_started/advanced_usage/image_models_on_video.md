---
title: "Image models on video tasks"
icon: lucide/film
---

## Evaluating Image Models on Video Tasks

Image-text models such as CLIP or SigLIP are commonly reported on video benchmarks even though they have no notion of time. However, a common approach is to sample a fixed number of frames uniformly from each clip and then encode each frame and mean-pool those into a single video embedding. This is the protocol used for CLIP-style baselines in CLIP4Clip, X-CLIP and ChinaOpen, among others.

By default, evaluating a model that supports `image` but not `video` on a video task raises an error. However, MTEB supplies a [`Video2ImagesWrapper`][mteb.models.video_wrappers.video2images_wrapper.Video2ImagesWrapper] to make it easier to quickly establish such baselines (see [Define the Model](../usage/defining_the_model.md) and [Run the Evaluation](../usage/running_the_evaluation.md) for the general workflow):

```python
import mteb
from mteb.models import Video2ImagesWrapper

task = mteb.get_task("MSRVTTT2V")
model = mteb.get_model("openai/clip-vit-base-patch32")

video_model = Video2ImagesWrapper(model, num_frames=8)
results = mteb.evaluate(video_model, tasks=[task])
```

`num_frames` controls how many frames are sampled uniformly from each video; alternatively sample at a rate with `fps` (optionally capped by `max_frames`). Text inputs (for example the queries of a text-to-video task) are passed through to the model unchanged.

### Caching frame embeddings

To cache the frame embeddings (see [Cache embeddings](cache_embeddings.md) for the cache wrapper), wrap the cache inside the video wrapper:

```python
from mteb.models import Video2ImagesWrapper
from mteb.models.cache_wrappers import CachedEmbeddingWrapper

video_model = Video2ImagesWrapper(
    CachedEmbeddingWrapper(model, "path/to/cache"), num_frames=8
)
```

The other order, `CachedEmbeddingWrapper(Video2ImagesWrapper(...))`, does not work because the cache can hash text and images but not video.

### Provenance

Sampling 8 frames uniformly and mean-pooling is the default protocol, the same frame count X-CLIP, ViCLIP, LanguageBind and InternVideo2 use by default in MTEB.

The sampling is always recorded in the model's `experiment_kwargs` (for example `video_frame_pooling` and `video_num_frames`), so the results are stored under an `experiments/` folder next to the model's regular results, exactly as loader overrides such as `mteb.get_model("microsoft/xclip-base-patch32", num_frames=16)` are handled for native video models. This keeps an image model evaluated on video distinguishable from a native video model, and runs with different sampling never overwrite each other.
