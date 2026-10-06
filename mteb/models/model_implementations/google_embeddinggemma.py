from __future__ import annotations

import logging

from mteb.models import SentenceTransformerEncoderWrapper
from mteb.models.model_implementations.google_gemini import GECKO_TRAINING_DATA
from mteb.models.model_meta import ModelMeta

logger = logging.getLogger(__name__)


MULTILINGUAL_EVALUATED_LANGUAGES = [
    "arb-Arab",
    "ben-Beng",
    "eng-Latn",
    "spa-Latn",
    "deu-Latn",
    "pes-Arab",
    "fin-Latn",
    "fra-Latn",
    "hin-Deva",
    "ind-Latn",
    "jpn-Jpan",
    "kor-Hang",
    "rus-Cyrl",
    "swh-Latn",
    "tel-Telu",
    "tha-Thai",
    "yor-Latn",
    "zho-Hant",
    "zho-Hans",
]


embedding_gemma_300m = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,  # type: ignore[call-arg]
    name="google/embeddinggemma-300m",
    model_type=["dense"],
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    open_weights=True,
    revision="64614b0b8b64f0c6c1e52b07e4e9a4e8fe4d2da2",
    release_date="2025-09-04",
    n_parameters=307_581_696,
    n_embedding_parameters=201_326_592,
    embed_dim=768,
    max_tokens=2048,
    license="gemma",
    reference="https://ai.google.dev/gemma/docs/embeddinggemma/model_card",
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
    similarity_fn_name="cosine",
    memory_usage_mb=1155,
    citation="""
@misc{vera2025embeddinggemmapowerfullightweighttext,
      title={EmbeddingGemma: Powerful and Lightweight Text Representations},
      author={Henrique Schechter Vera and Sahil Dua and Biao Zhang and Daniel Salz and Ryan Mullins and Sindhu Raghuram Panyam and Sara Smoot and Iftekhar Naim and Joe Zou and Feiyang Chen and Daniel Cer and Alice Lisak and Min Choi and Lucas Gonzalez and Omar Sanseviero and Glenn Cameron and Ian Ballantyne and Kat Black and Kaifeng Chen and Weiyi Wang and Zhe Li and Gus Martins and Jinhyuk Lee and Mark Sherwood and Juyeong Ji and Renjie Wu and Jingxiao Zheng and Jyotinder Singh and Abheesht Sharma and Divyashree Sreepathihalli and Aashi Jain and Adham Elarabawy and AJ Co and Andreas Doumanoglou and Babak Samari and Ben Hora and Brian Potetz and Dahun Kim and Enrique Alfonseca and Fedor Moiseev and Feng Han and Frank Palma Gomez and Gustavo Hernández Ábrego and Hesen Zhang and Hui Hui and Jay Han and Karan Gill and Ke Chen and Koert Chen and Madhuri Shanbhogue and Michael Boratko and Paul Suganthan and Sai Meher Karthik Duddu and Sandeep Mariserla and Setareh Ariafar and Shanfeng Zhang and Shijie Zhang and Simon Baumgartner and Sonam Goenka and Steve Qiu and Tanmaya Dabral and Trevor Walker and Vikram Rao and Waleed Khawaja and Wenlei Zhou and Xiaoqi Ren and Ye Xia and Yichang Chen and Yi-Ting Chen and Zhe Dong and Zhongli Ding and Francesco Visin and Gaël Liu and Jiageng Zhang and Kathleen Kenealy and Michelle Casbon and Ravin Kumar and Thomas Mesnard and Zach Gleicher and Cormac Brick and Olivier Lacombe and Adam Roberts and Qin Yin and Yunhsuan Sung and Raphael Hoffmann and Tris Warkentin and Armand Joulin and Tom Duerig and Mojtaba Seyedhosseini},
      year={2025},
      eprint={2509.20354},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.20354},
}""",
    extra_requirements_groups=["embeddinggemma"],
)


EMBEDDING_GEMMA_2_PLACEHOLDER_CITATION = """
@article{embedding_gemma_2025,
    title={EmbeddingGemma: Powerful and Lightweight Text Representations},
    author={Schechter Vera, Henrique* and Dua, Sahil* and Zhang, Biao and Salz, Daniel and Mullins, Ryan and Raghuram Panyam, Sindhu and Smoot, Sara and Naim, Iftekhar and Zou, Joe and Chen, Feiyang and Cer, Daniel and Lisak, Alice and Choi, Min and Gonzalez, Lucas and Sanseviero, Omar and Cameron, Glenn and Ballantyne, Ian and Black, Kat and Chen, Kaifeng and Wang, Weiyi and Li, Zhe and Martins, Gus and Lee, Jinhyuk and Sherwood, Mark and Ji, Juyeong and Wu, Renjie and Zheng, Jingxiao and Singh, Jyotinder and Sharma, Abheesht and Sreepat, Divya and Jain, Aashi and Elarabawy, Adham and Co, AJ and Doumanoglou, Andreas and Samari, Babak and Hora, Ben and Potetz, Brian and Kim, Dahun and Alfonseca, Enrique and Moiseev, Fedor and Han, Feng and Palma Gomez, Frank and Hernández Ábrego, Gustavo and Zhang, Hesen and Hui, Hui and Han, Jay and Gill, Karan and Chen, Ke and Chen, Koert and Shanbhogue, Madhuri and Boratko, Michael and Suganthan, Paul and Duddu, Sai Meher Karthik and Mariserla, Sandeep and Ariafar, Setareh and Zhang, Shanfeng and Zhang, Shijie and Baumgartner, Simon and Goenka, Sonam and Qiu, Steve and Dabral, Tanmaya and Walker, Trevor and Rao, Vikram and Khawaja, Waleed and Zhou, Wenlei and Ren, Xiaoqi and Xia, Ye and Chen, Yichang and Chen, Yi-Ting and Dong, Zhe and Ding, Zhongli and Visin, Francesco and Liu, Gaël and Zhang, Jiageng and Kenealy, Kathleen and Casbon, Michelle and Kumar, Ravin and Mesnard, Thomas and Gleicher, Zach and Brick, Cormac and Lacombe, Olivier and Roberts, Adam and Sung, Yunhsuan and Hoffmann, Raphael and Warkentin, Tris and Joulin, Armand and Duerig, Tom and Seyedhosseini, Mojtaba},
    publisher={Google DeepMind},
    year={2025},
    url={https://arxiv.org/abs/2509.20354}
}"""


embedding_gemma_2_text = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,  # type: ignore[call-arg]
    name="google/embeddinggemma-2-text",
    model_type=["dense"],
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    open_weights=True,
    revision="914f7f89142e33e77833254d9c9b90c3cef7303b",
    release_date="2026-10-06",
    n_parameters=271_002_624,
    n_embedding_parameters=134_217_728,
    embed_dim=768,
    max_tokens=8192,
    license="gemma",
    reference="https://huggingface.co/google/embeddinggemma-2",
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
    similarity_fn_name="cosine",
    memory_usage_mb=1034,
    modalities=["text"],
    citation=EMBEDDING_GEMMA_2_PLACEHOLDER_CITATION,
    contacts=["hschechter"],
    extra_requirements_groups=["embeddinggemma"],
)


embedding_gemma_2_text_vision = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,  # type: ignore[call-arg]
    name="google/embeddinggemma-2-text-vision",
    model_type=["dense"],
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    open_weights=True,
    revision="914f7f89142e33e77833254d9c9b90c3cef7303b",
    release_date="2026-10-06",
    n_parameters=438_760_448,
    n_embedding_parameters=134_217_728,
    embed_dim=768,
    max_tokens=8192,
    license="gemma",
    reference="https://huggingface.co/google/embeddinggemma-2",
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
    similarity_fn_name="cosine",
    memory_usage_mb=1674,
    modalities=["text", "image"],
    citation=EMBEDDING_GEMMA_2_PLACEHOLDER_CITATION,
    contacts=["hschechter"],
    extra_requirements_groups=["embeddinggemma"],
)


embedding_gemma_2_text_audio = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,  # type: ignore[call-arg]
    name="google/embeddinggemma-2-text-audio",
    model_type=["dense"],
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    open_weights=True,
    revision="914f7f89142e33e77833254d9c9b90c3cef7303b",
    release_date="2026-10-06",
    n_parameters=576_613_664,
    n_embedding_parameters=134_217_728,
    embed_dim=768,
    max_tokens=8192,
    license="gemma",
    reference="https://huggingface.co/google/embeddinggemma-2",
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
    similarity_fn_name="cosine",
    memory_usage_mb=2200,
    modalities=["text", "audio"],
    citation=EMBEDDING_GEMMA_2_PLACEHOLDER_CITATION,
    contacts=["hschechter"],
    extra_requirements_groups=["embeddinggemma"],
)
