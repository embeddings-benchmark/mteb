# EVVE construction notes

EVVE is a video-to-video specific-event retrieval benchmark introduced at
CVPR 2013. The published core protocol contains 620 query videos, 2,375
database videos, 13 events, and 135,213 positive query/database judgments. Its
optional large-scale track adds 100,000 distractor videos.

The original project page releases annotations, evaluation software, and
descriptors and identifies the media by YouTube ID. Its legal note says that
the software is BSD-licensed, descriptors are free, and copyright details for
original videos remain on their YouTube pages. No dataset-wide video license is
specified. The page does not state an explicit prohibition on redistribution.
The constructor records this provenance without claiming that redistribution
is authorized by EVVE or that video copyright was transferred.

`s2vs-2023-video-ids.txt` freezes the video IDs present in the public S2VS EVVE
feature artifact. Of those 2,410 IDs, 2,110 videos were reproducibly obtainable
from public sources: 2,042 directly from their original YouTube IDs, 67 from
exact Wayback Machine captures, and one from an exact Internet Archive item.
`surviving-public-media-video-ids.txt` freezes that packaged set. Intersecting it
with the checksum-pinned annotation file produces the fixed evaluation protocol:

- 466 query videos;
- 1,644 database videos;
- all 13 events;
- 86,925 binary relevance judgments.

Filtering the S2VS layer removes 300 unavailable IDs: 38 queries and 262 database
videos (81 positive and 181 other candidates), eliminating 13,864 qrels. The
final protocol retains all 13 events, 958 positive database videos, 686 other
candidates, zero query/corpus ID overlap, and 86,925 positive qrels. Every one
of the 466 retained queries has at least one positive. `protocol-summary.json`
records event-by-event original, S2VS-before-filter, and packaged-after-filter
coverage.

This is a reproducible surviving-public-media subset of EVVE, not the complete
original benchmark. It has 154 fewer queries and 731 fewer database videos than
the published core, and it omits the original 100,000 distractors. Published
scores are therefore not directly comparable.

Run a metadata-only integrity audit with:

```bash
uv run python scripts/data/evve/create_data.py --work-dir /tmp/evve-mteb
```

Download the exact frozen set reproducibly with yt-dlp. Downloads are resumable,
limited to video at up to 360p, and never silently remove failures:

```bash
uv run python scripts/data/evve/create_data.py \
  --work-dir /tmp/evve-mteb \
  --download \
  --workers 8
```

YouTube may require proof-of-origin tokens even for public, logged-out media.
When that happens, install a guest PO-token provider alongside yt-dlp and pass
its client/runtime settings without using account cookies. For example, with a
compatible provider listening on its default local port:

```bash
uv run \
  --with 'yt-dlp[default]==2026.7.4' \
  --with 'bgutil-ytdlp-pot-provider==1.3.1' \
  python scripts/data/evve/create_data.py \
  --work-dir /tmp/evve-mteb \
  --download \
  --workers 4 \
  --yt-dlp-arg=--js-runtimes \
  --yt-dlp-arg=node \
  --yt-dlp-arg=--extractor-args \
  --yt-dlp-arg=youtube:player_client=mweb
```

`--yt-dlp-arg` is repeatable so current extractor requirements can be supplied
explicitly and recorded with a construction run. Authentication cookies are
optional and are never read unless `--cookies-from-browser` is provided.
For a distributed or focused resume, `--download-id-file` accepts a newline-
separated subset of the frozen IDs. It changes only the acquisition queue;
packaging still validates the complete 2,110-video evaluation manifest.

The constructor writes `download-failures.jsonl` and fails until all 2,110
videos are present and contain decodable frames. Four YouTube IDs whose
progressive format currently contains MP4 headers but no media payload are
downloaded from separate video-only and audio streams and merged without
transcoding. The complete frozen tree was also audited with the same TorchCodec
10-frame uniform sampler that MTEB applies during evaluation; all 2,110 videos
passed. On a resumed run it validates existing media
only once, tries previously unattempted IDs before earlier transient and hard
failures, and stops the pass immediately if YouTube starts returning its
bot-confirmation gate. This preserves completed files and prevents temporary
rate limits from being recorded as dataset attrition. `media-manifest.jsonl`
then freezes each
filename, original YouTube URL, actual acquisition URL, byte count, and SHA-256
digest. `media-source-overrides.json` pins exact public archive captures and
their checksums for source IDs that are no longer directly downloadable. To use
an existing media tree, pass `--media-root`; files may be named
`<youtube-id>.*` or placed beneath a directory named `<youtube-id>`.

After validating the complete frozen set, push the standard MTEB `corpus`,
`queries`, and `qrels` configurations plus the dataset card and manifests:

```bash
uv run python scripts/data/evve/create_data.py \
  --work-dir /tmp/evve-mteb \
  --repo-id Cerru02/EVVE \
  --push
```

The Hub upload uses the currently authenticated Hugging Face identity and writes
the immutable final revision to `hub_revision.txt`.

Sources:

- Original paper: https://openaccess.thecvf.com/content_cvpr_2013/html/Revaud_Event_Retrieval_in_2013_CVPR_paper.html
- Archived project page: https://web.archive.org/web/20241102150758id_/http://pascal.inrialpes.fr/data2/evve/index.html
- S2VS source revision: https://github.com/gkordo/s2vs/tree/03eb0c9a9a7455d132c4210b797fa6ef563b52ea
- S2VS feature artifact: https://mever.iti.gr/s2vs/features/evve.hdf5
- Surviving evaluator copy: https://github.com/fyang93/BURST/blob/c0f59d49ffe6b4069809794697554c8a1ce969d9/eval_evve.py

The original evaluator uses trapezoidal AP and removes event-specific `null`
videos from each ranking. It reports query-weighted `overall mAP` and the mean
of 13 event-level mAP values as `avg-mAP`. Table 3 identifies the latter as
`Event avg-mAP`. The MTEB task therefore retrieves all 1,644 database videos,
preserves the per-query ignored-ID field (empty in the pinned annotation
snapshot), and exposes event-balanced `evve_avg_map` as `main_score`;
`evve_overall_map` is also reported. Ordinary pytrec
`map_at_k` remains useful but is not identical to the source score.

The original 100,000-video distractor collection is not publicly available and
is not included. Together with core attrition, that makes scores on this frozen
reduced corpus not directly comparable to source-paper results.
