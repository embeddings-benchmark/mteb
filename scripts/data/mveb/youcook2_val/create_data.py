"""Reference stub for the mteb/YouCook2_val dataset.

REFERENCE ONLY — SOURCE / PIPELINE LOST. Pipeline not recovered; source
archive likely `YouCookIIVideos.zip` (UMich).

Source
------
YouCook2: Zhou et al., "Towards Automatic Learning of Procedures from Web
Instructional Videos" (AAAI 2018).
    http://youcook2.eecs.umich.edu/

Published artifact schema (from the HuggingFace dataset card)
-------------------------------------------------------------
Single `test` split with 3,104 rows and the following columns:
    video    : string (clip path / identifier)
    audio    : string (extracted track path / identifier)
    sentence : string (cooking step description)

~1k validation videos × ~3 steps each = 3,104 rows.

How to reproduce
----------------
1. Obtain `YouCookIIVideos.zip` (~30 GB) from the official UMich
   distribution or a HuggingFace mirror.
2. Parse the official YouCook2 validation annotation JSON to extract
   per-step (video_id, start_time, end_time, sentence) records.
3. Cut each step segment from the parent video with ffmpeg, extract
   mono 16 kHz audio, and yield {video, audio, sentence}.
"""
