"""Reference pointer for the mteb/kinetics-600 dataset.

REFERENCE ONLY. The processing follows the same pattern as
kinetics_400. See [../kinetics_400/create_data.py](../kinetics_400/create_data.py)
for the canonical implementation; the kinetics-600 run uses the source
URLs and parameters below.

Source
------
Kinetics-600: Carreira et al., "A Short Note about Kinetics-600" (2018).
    https://arxiv.org/abs/1808.01340

Mirror used:
    https://github.com/cvdfoundation/kinetics-dataset
    https://s3.amazonaws.com/kinetics/600/test/k600_test_path.txt
    https://s3.amazonaws.com/kinetics/600/annotations/test.csv

MVEB-specific processing
------------------------
Same shape as kinetics_400 with two differences:
- Only the official `test` split is downloaded and published (no `train`).
- Per-class cap raised from 10 to ~16.

The 600 sorted action classes are joined via the annotation CSV, audio
is extracted with ffmpeg, and the schema is `{video, audio, label}`
where `label` is a ClassLabel over the 600 class names.

Final size: ~9,576 rows in the single `test` split.
"""
