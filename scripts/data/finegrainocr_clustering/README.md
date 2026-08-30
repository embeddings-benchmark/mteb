# FineGrainOCR image+text clustering exploration

## Recommendation

FineGrainOCR is a strong candidate for the missing non-audio/video cross-modal
clustering coverage in MOEB. It has native image+OCR pairs, 256 fine-grained
grocery-product classes, barcode-grounded rather than subjective labels, and a
CC0-1.0 source license. The source paper reports that image and OCR features are
complementary and that multimodal fusion outperforms either unimodal model.

Proposed MTEB task:

- Name: `FineGrainOCRITClustering`
- Type: `ImageClustering`
- Category: `it2c` (new metadata category)
- Modalities: `image`, `text`
- Input columns: `("image", "text")`
- Label: the source product-class directory (a barcode/GTIN identifier)
- Evaluation data: a deterministic class-capped subset of the source validation
  split
- Main score: V-measure
- License: `cc0-1.0`

Do not add the task definition until a maintainer agrees with the formulation.
The dataset proposal should call out that OCR is derived from the product image,
and that long numeric sequences will be redacted to prevent the class barcode
from becoming a shortcut.

## Why not the other candidates?

- Memotion: the maintainer flagged the subjective labels and low inter-annotator
  agreement reported by the paper.
- N24News: the official builder scrapes New York Times text and images, and the
  source repository does not declare a dataset license.
- CUB-200 plus descriptions: the species labels are strong, but the official
  site restricts image use to non-commercial research and warns about ImageNet
  overlap. The human descriptions also come from a separate release.
- GLAMI-1M: a very good fit, but another contributor already opened MTEB PR
  #5331 for its multimodal classification task.

FineGrainOCR was not mentioned by any existing MTEB issue or PR when checked on
2026-08-30.

## Provenance

- Source repository: <https://github.com/Tubbias/finegrainocr>
- Inspected source commit: `9ce19719123fd33a994b103b6e91c37a640ce92b`
- Paper: <https://doi.org/10.1007/s00138-024-01549-9>
- License: CC0-1.0 in the source repository
- Source archive: the Dropbox URL in the source README
- Observed archive size: 53,728,458,454 bytes

The paper explains that an automated checkout's barcode scanners register the
product class while a camera captures the product from unconstrained customer
placements. Google Vision API OCR is stored alongside each image. This makes the
class identity substantially less ambiguous than Memotion's subjective labels.

## Archive audit

The 50 GB archive supports HTTP byte ranges, so its ZIP index and a selected
subset can be fetched without downloading the entire file.

The inspected ZIP64 central directory has:

- byte range: `53701780340-53728458355`
- size: 26,678,016 bytes
- SHA-256: `242eaa1c31c37a47957269ce598e11b1414dbfe1d154c72977952e2314cbbb8a`
- entries: 184,620

It can be downloaded and audited with:

```bash
curl -L \
  --range 53701780340-53728458355 \
  'https://www.dropbox.com/scl/fi/jraqxgrg0z7carmj7anxs/FineGrainOCR.zip?rlkey=qq9p7orig0csxo7s1vq1htc5r&dl=1' \
  -o /tmp/finegrainocr-central-directory.bin

python scripts/data/finegrainocr_clustering/analyze_archive.py \
  /tmp/finegrainocr-central-directory.bin \
  --cap-per-class 20 \
  --seed 42 \
  --manifest-out /tmp/finegrainocr-validation-manifest.json
```

Observed source layout:

| Split | Image members | OCR members | Paired samples | Image count per class |
|---|---:|---:|---:|---:|
| train | 73,378 | 73,378 | 73,378 | 23–587 (median 315) |
| validation | 18,416 | 18,416 | 18,416 | 6–147 (median 79) |

There are 256 classes and no image-only or text-only samples.

## Proposed evaluation subset

Rank validation samples within each class using SHA-256 of
`seed + class_id + sample_stem`, retain up to 20 per class, and skip empty OCR.
This yields:

- 4,919 image+text pairs across all 256 classes
- 6–20 examples per class (median 20)
- 2,866,365,935 compressed source bytes for the selected ZIP members
- 13 selected OCR strings containing the exact class ID, which must be redacted
- no selected empty OCR strings

Use the first Google Vision result's `description` field. This matches the
paper's representation: all detected words concatenated in the API's default
order. Preserve the full string in the hosted dataset and let model token limits
handle truncation.

Resize images so the longest edge is 512 pixels and encode as optimized JPEG at
quality 90. The paper explicitly evaluates 512-pixel images. In the extraction
smoke test, one 2,592×1,944 source JPEG shrank from 633,106 to 21,718 bytes, so
the hosted subset should be on the order of 100–200 MB rather than gigabytes.

Redact all OCR digit sequences containing 8–14 digits, allowing spaces or
hyphens between digits. This removes exact and alternate barcode identifiers
while retaining product names, ingredients, and other useful package text.

## OCR quality audit

All validation OCR files occupy one contiguous archive range,
`197817161-244164911` (46,347,751 bytes). Its inspected SHA-256 is
`aef86966828d1e3daec104363007da1858b6f13656ef8e907bcc6e01b878b9ff`.
Passing that span to `analyze_archive.py` produced:

- 27 empty descriptions in all 18,416 validation rows
- median 257 characters; 95th percentile 1,438; maximum 3,636
- 11,861 first detections marked English and 6,008 marked French; the remainder
  are mostly noisy low-count language guesses
- 59 descriptions containing their exact class ID
- 5,130 descriptions containing at least one barcode-like 8–14 digit sequence
- 121 exact duplicate rows, with 12 duplicate strings crossing class boundaries

Audit command:

```bash
curl -L \
  --range 197817161-244164911 \
  'https://www.dropbox.com/scl/fi/jraqxgrg0z7carmj7anxs/FineGrainOCR.zip?rlkey=qq9p7orig0csxo7s1vq1htc5r&dl=1' \
  -o /tmp/finegrainocr-validation-text-span.bin

python scripts/data/finegrainocr_clustering/analyze_archive.py \
  /tmp/finegrainocr-central-directory.bin \
  --cap-per-class 20 \
  --seed 42 \
  --validation-text-span /tmp/finegrainocr-validation-text-span.bin
```

## Selective extraction proof

A selected image and its OCR JSON were fetched independently from the remote ZIP
using their central-directory offsets, decompressed with raw DEFLATE, and checked
against their central-directory CRC-32 and uncompressed sizes. The decoded image
was a valid 2,592×1,944 JPEG and the matching OCR JSON had the expected Google
Vision schema. This verifies that a construction script can download only the
selected members.

## Remaining work

1. Ask the MOEB maintainer to approve FineGrainOCR as an `IT -> class`
   clustering task, explicitly disclosing that the text is OCR-derived.
2. Turn the range-extraction proof into a resumable, parallel construction
   script with CRC checks, barcode redaction, image resizing, and a dataset card.
3. Upload the processed 4,919-row test set to the MTEB Hugging Face organization
   and pin its revision.
4. Add `it2c` metadata support, the task definition, registration, tests, and
   descriptive statistics.
5. Run one genuinely joint image+text embedding baseline plus image-only and
   text-only diagnostic ablations. Check that the joint task is neither random
   nor saturated and that both modalities add signal.

## Draft tracker comment

> For the missing cross-modal clustering task, I propose FineGrainOCR as an
> image+text (`IT -> class`) clustering benchmark. It contains 91,794 paired
> product images and Google Vision OCR texts across 256 barcode-registered
> grocery-product classes and is released under CC0-1.0. Unlike Memotion, the
> cluster labels are scanner-grounded product identities rather than subjective
> judgments. The paper reports that image and OCR features complement each other
> and multimodal fusion outperforms unimodal models. I audited the official
> archive and found complete image/OCR pairing. To keep evaluation practical, I
> would use a deterministic, class-capped subset of the source validation split:
> 4,919 non-empty pairs, 6–20 per class, with images resized to the paper-tested
> 512-pixel setting. I would redact 8–14 digit OCR sequences so printed barcodes
> cannot leak the class identifier. Would this formulation fit the intended
> cross-modal clustering gap?
