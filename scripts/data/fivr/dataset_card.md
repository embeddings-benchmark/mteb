---
license: apache-2.0
pretty_name: FIVR-5K MTEB Metadata
task_categories:
  - feature-extraction
tags:
  - mteb
  - moeb
  - video-retrieval
  - fivr
configs:
  - config_name: corpus
    data_files:
      - split: test
        path: corpus/test-*
  - config_name: queries
    data_files:
      - split: test
        path: queries/test-*
  - config_name: dsvr-qrels
    data_files:
      - split: test
        path: dsvr-qrels/test-*
  - config_name: csvr-qrels
    data_files:
      - split: test
        path: csvr-qrels/test-*
  - config_name: isvr-qrels
    data_files:
      - split: test
        path: isvr-qrels/test-*
  - config_name: availability
    data_files:
      - split: test
        path: availability/test-*
  - config_name: positive-losses
    data_files:
      - split: test
        path: positive-losses/test-*
  - config_name: query-decisions
    data_files:
      - split: test
        path: query-decisions/test-*
---

# FIVR-5K for MTEB

This repository packages the metadata needed for the MTEB/MOEB version of
fine-grained incident video retrieval. It contains no video bytes.

## Relationship to the source benchmarks

[FIVR-200K](https://github.com/MKLab-ITI/FIVR-200K) is the authoritative
source of the YouTube IDs and human ND/DS/CS/IS annotations. The canonical
FIVR-5K protocol published with
[ViSiL](https://github.com/MKLab-ITI/visil) selects the 50 most difficult DSVR
queries (using iMAC to measure difficulty), randomly retains 30% of annotated
videos per label category for each query, and publishes a 5,000-video database
from FIVR-200K. The released ViSiL metadata is used to validate membership; the
MTEB construction does not rerun that historical random selection.

[VideoEval](https://huggingface.co/datasets/lixinhao/VideoEval) freezes a
downloadable subset of that established protocol: 31 of the canonical queries,
3,415 canonical database rows, and 3,445 unique IDs (one query is also a
database row). Its annotations are identical to the official FIVR-200K JSON.
The MTEB construction validates all of these relationships instead of sampling
a new subset.

The older ViSiL pickle and current FIVR/VideoEval annotations differ for two
manifest items, both historical IS positives that are absent from the current
official annotation. The construction records and validates this drift, then
uses the current official annotation bundled identically by VideoEval.

The MTEB freeze re-audited all VideoEval IDs on 2026-08-10. It observed
{available} available and {unavailable} unavailable/restricted IDs. After
removing unavailable media, excluding the single query self-match from the
corpus, and retaining only available queries with a surviving DSVR positive,
the evaluation has {queries} queries and {corpus} corpus videos.
Across all 31 VideoEval source queries, unavailable corpus media removed
{all_nd_lost} ND, {all_ds_lost} DS, {all_cs_lost} CS, and {all_is_lost} IS
positive assignments. For the {queries} retained evaluation queries, the
corresponding losses are {retained_nd_lost} ND, {retained_ds_lost} DS,
{retained_cs_lost} CS, and {retained_is_lost} IS. The per-query records remain
available in `positive-losses` rather than being silently discarded.

## Retrieval definitions

All qrels are binary, exactly following the official evaluator:

- DSVR: ND + DS ({dsvr_qrels} qrels)
- CSVR: ND + DS + CS ({csvr_qrels} qrels)
- ISVR: ND + DS + CS + IS ({isvr_qrels} qrels)

The source metric is full-ranking mean average precision. MTEB evaluates
`map_at_<corpus size>`, which is equivalent because every corpus item is ranked.

## Media and licensing

The FIVR and ViSiL repositories, the VideoEval metadata repository, and this
metadata-only derivative are Apache-2.0. That license does not grant rights to
redistribute the underlying third-party YouTube videos. VideoEval likewise
instructs users to download original media separately because of potential
copyright issues.

Consequently, this repository publishes only IDs, original-source URLs,
availability decisions, and qrels. The MTEB task does not download media.
Prepare a complete local video directory separately with the reproducible
construction script, then set `MTEB_FIVR_VIDEO_DIR` or pass `fivr_video_dir` to
the task. Missing files are reported as an error; the benchmark never silently
changes its frozen corpus.

Users are responsible for complying with the original platforms' terms and
applicable law. Individual media rights remain with their respective owners.

## Configurations

- `corpus`, `queries`: identifiers, source URLs, frozen availability status,
  and duration metadata;
- `dsvr-qrels`, `csvr-qrels`, `isvr-qrels`: official binary relevance unions;
- `availability`: every ID from VideoEval's frozen manifest, including missing
  items;
- `positive-losses`: lost positives per query and ND/DS/CS/IS label;
- `query-decisions`: retained/dropped query decisions.

The `audit/` directory contains the human-readable construction summary and
the full frozen availability/loss records.

## Citations

```bibtex
@article{kordopatis2019fivr,
  author = {Kordopatis-Zilos, Giorgos and Papadopoulos, Symeon and Patras, Ioannis and Kompatsiaris, Ioannis},
  journal = {IEEE Transactions on Multimedia},
  title = {FIVR: Fine-grained Incident Video Retrieval},
  year = {2019}
}

@inproceedings{kordopatis2019visil,
  author = {Kordopatis-Zilos, Giorgos and Papadopoulos, Symeon and Patras, Ioannis and Kompatsiaris, Ioannis},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  title = {ViSiL: Fine-grained Spatio-Temporal Video Similarity Learning},
  year = {2019}
}

@article{li2024videoeval,
  author = {Li, Xinhao and Huang, Zhenpeng and Wang, Jing and Li, Kunchang and Wang, Limin},
  journal = {arXiv preprint arXiv:2407.06491},
  title = {VideoEval: Comprehensive Benchmark Suite for Low-cost Evaluation of Video Foundation Model},
  year = {2024}
}
```
