---
title: "Image models on video tasks"
icon: lucide/film
---

## Evaluating Image Models on Video Tasks

Image-text models such as CLIP or SigLIP are commonly reported on video benchmarks even though they have no notion of time: a fixed number of frames is sampled uniformly from each clip, every frame is encoded as an image, and the frame embeddings are mean-pooled into a single video embedding. This is the protocol used for CLIP-style baselines in CLIP4Clip, X-CLIP and ChinaOpen, among others.

MTEB applies this protocol automatically whenever a model that supports `image` but not `video` is evaluated on a video task. No change to the model or its `ModelMeta` is needed:

```bash
mteb run -m openai/clip-vit-base-patch32 -t MSRVTTT2V --video-frames 8
```

```python
import mteb

task = mteb.get_task("MSRVTTT2V")
model = mteb.get_model("openai/clip-vit-base-patch32")

results = mteb.evaluate(model, tasks=[task], video_frames=8)
```

`video_frames` controls how many frames are sampled uniformly from each video. If it is not set, MTEB falls back to 8 frames and emits a warning, so a frame count is always a deliberate part of a reported result. Text inputs (for example the queries of a text-to-video task) are passed through to the model unchanged.

### Provenance

Sampling 8 frames uniformly and mean-pooling is the default protocol, the same frame count X-CLIP, ViCLIP, LanguageBind and InternVideo2 use by default in MTEB. Results produced with it are stored as the model's regular results and appear on the leaderboard like those of native video models.

Any other sampling (`video_frames=16`, or `fps` / `max_frames` through the wrapper) is recorded in the model's `experiment_kwargs` and stored under an `experiments/` folder next to the regular results, exactly as loader overrides such as `mteb.get_model("microsoft/xclip-base-patch32", num_frames=16)` are handled for native video models. Those results never overwrite the default-protocol ones and are not shown on the leaderboard.

### Using the wrapper directly

Under the hood `mteb.evaluate` wraps the model in [`Video2ImagesWrapper`][mteb.models.video_wrappers.video2images_wrapper.Video2ImagesWrapper]. You can do the same yourself for custom pipelines, or to sample frames at a rate (`fps`, optionally capped by `max_frames`) instead of a fixed count, as the native video models in MTEB do:

```python
from mteb.models import Video2ImagesWrapper

video_model = Video2ImagesWrapper(model, num_frames=8)
# or: Video2ImagesWrapper(model, fps=2.0, max_frames=64)
embeddings = video_model.encode(
    dataloader, task_metadata=task.metadata, hf_split="test", hf_subset="default"
)
```

### Limitations

- Only video-only inputs are pooled. Tasks whose rows combine video with text or audio in a single input (e.g. `vt2t`) are not supported and raise `NotImplementedError`.
- Decoding video requires the `video` extra (`pip install "mteb[video]"`).
