"""Implementation of Sentence Transformers model validated in MTEB."""

from mteb.model_meta import ModelMeta

all_MiniLM_L6_v2 = ModelMeta(
    name="sentence-transformers/all-MiniLM-L6-v2",
    languages=["eng-Latn"],
    open_source=True,
    revision="e4ce9877abf3edfe10b0d82785e83bdcb973e22e",
    release_date="2021-08-30",
)

paraphrase_multilingual_MiniLM_L12_v2 = ModelMeta(
    name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    languages=[], # TODO:  ar, bg, ca, cs, da, de, el, en, es, et, fa, fi, fr, fr-ca, gl, gu, he, hi, hr, hu, hy, id, it, ja, ka, ko, ku, lt, lv, mk, mn, mr, ms, my, nb, nl, pl, pt, pt-br, ro, ru, sk, sl, sq, sr, sv, th, tr, uk, ur, vi, zh-cn, zh-tw
    open_source=True,
    revision="e4ce9877abf3edfe10b0d82785e83bdcb973e22e",  # TODO: update revision
    release_date="2019-11-01", # release date of paper 
)
