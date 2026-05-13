"""Reference pointer for the mteb/kinetics-700-2020 dataset.

REFERENCE ONLY. The processing follows the same pattern as
kinetics_400. See [../kinetics_400/create_data.py](../kinetics_400/create_data.py)
for the canonical implementation; the kinetics-700-2020 run uses the
source URLs and parameters below.

Source
------
Kinetics-700-2020: Smaira et al., "A Short Note on the Kinetics-700-2020
Human Action Dataset" (2020). https://arxiv.org/abs/2010.10864

Mirror used:
    https://github.com/cvdfoundation/kinetics-dataset
    https://s3.amazonaws.com/kinetics/700_2020/test/k700_2020_test_path.txt
    https://s3.amazonaws.com/kinetics/700_2020/annotations/test.csv

MVEB-specific processing
------------------------
Same shape as kinetics_400 with two differences:
- Only the official `test` split is downloaded and published (no `train`).
- Per-class cap raised from 10 to ~16.

The 700 sorted action classes (2020 vocabulary) are joined via the
annotation CSV, audio is extracted with ffmpeg, and the schema is
`{video, audio, label}` where `label` is a ClassLabel over the 700
class names.

Final size: ~11,190 rows in the single `test` split.
"""
