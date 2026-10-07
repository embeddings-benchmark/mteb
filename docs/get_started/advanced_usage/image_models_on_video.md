---
title: "Image models on video tasks"
icon: lucide/film
---

## Evaluating Image Models on Video Tasks

Image-text models such as CLIP or SigLIP are commonly reported on video benchmarks even though they have no notion of time: a fixed number of frames is sampled uniformly from each clip, every frame is encoded as an image, and the frame embeddings are mean-pooled into a single video embedding. This is the protocol used for CLIP-style baselines in CLIP4Clip, X-CLIP and ChinaOpen, among others.

MTEB never applies this protocol implicitly. Evaluating a model that supports `image` but not `video` on a video task raises an error pointing here, so a frame count is always a deliberate part of a reported result. To opt in, wrap the model in [`Video2ImagesWrapper`][mteb.models.video_wrappers.video2images_wrapper.Video2ImagesWrapper]:

```python
import mteb
from mteb.models import Video2ImagesWrapper

task = mteb.get_task("MSRVTTT2V")
model = mteb.get_model("openai/clip-vit-base-patch32")

video_model = Video2ImagesWrapper(model, num_frames=8)
results = mteb.evaluate(video_model, tasks=[task])
```

`num_frames` controls how many frames are sampled uniformly from each video; alternatively sample at a rate with `fps` (optionally capped by `max_frames`). Text inputs (for example the queries of a text-to-video task) are passed through to the model unchanged. Wrapping is not available from the CLI.

For tasks that mix video with other modalities the model does support, `mteb.evaluate` only warns about the partial overlap, so images can still be evaluated without the wrapper.

### Provenance

Sampling 8 frames uniformly and mean-pooling is the default protocol, the same frame count X-CLIP, ViCLIP, LanguageBind and InternVideo2 use by default in MTEB. Results produced with it are stored as the model's regular results and appear on the leaderboard like those of native video models.

Any other sampling (`num_frames=16`, or `fps` / `max_frames`) is recorded in the model's `experiment_kwargs` and stored under an `experiments/` folder next to the regular results, exactly as loader overrides such as `mteb.get_model("microsoft/xclip-base-patch32", num_frames=16)` are handled for native video models. Those results never overwrite the default-protocol ones and are not shown on the leaderboard.

### Limitations

- Only video-only inputs are pooled. Tasks whose rows combine video with text or audio in a single input (e.g. `vt2t`) are not supported and raise `NotImplementedError`.
- Decoding video requires the `video` extra (`pip install "mteb[video]"`).
