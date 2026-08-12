# Mars-VL-Pairs construction

This directory freezes Task 1 of
[MarsRetrieval](https://github.com/ml-stat-Sustech/MarsRetrieval), replacing its
mutable web image URLs with validated image bytes while preserving the original
one-to-one order and provenance.

The builder pins `SUSTech/Mars-VL-Pairs`, downloads every image with retries,
validates it with Pillow, records redirects and failures, computes byte/pixel
hashes and dimensions, checks URL/caption/media duplicates, and writes a JSON
audit report. Both MTEB directions are derived from the same frozen pair table.

```bash
/path/to/python scripts/data/mars_vl_pairs/create_data.py \
  --work-dir /tmp/mars_vl_pairs_mteb \
  --archive-recovery

# Upload only after reviewing audit_summary.json and audit_rows.jsonl.
/path/to/python scripts/data/mars_vl_pairs/create_data.py \
  --work-dir /tmp/mars_vl_pairs_mteb \
  --archive-recovery \
  --allow-missing \
  --repo-id Cerru02/Mars-VL-Pairs-MTEB \
  --push
```

If direct downloads fail, `--archive-recovery` also upgrades HTTP URLs to HTTPS
and attempts exact-URL Wayback snapshots, including captures stored with an
incorrect MIME type. HTTPS certificate verification is always enabled.
Recovered rows retain the original URL and recovery method. The script stops
on missing or exact duplicate media by default; `--allow-missing` and
`--allow-duplicate-images` require an explicit, documented decision.

Manual review of every image pair with dHash distance at most two found two
resize-equivalent groups: source rows 460/907 and 1446/2098. The builder
deterministically excludes the lower-resolution rows 907 and 2098 because their
different captions would otherwise create ambiguous one-positive qrels. The
other near-hash alerts are visibly different scenes.

The source dataset declares CC-BY-4.0. Its images were selected from web-scale
corpora and come from many external domains, so the builder preserves the
source URL and does not treat the dataset-level declaration as a substitute for
the original source's rights information.
