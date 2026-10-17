"""Audio and video embedding benchmarks (MAEB, MVEB)."""

from mteb.benchmarks.benchmark import (
    Benchmark,
    BenchmarkAggregation,
    CustomGroup,
    CustomGrouping,
)
from mteb.get_tasks import MTEBTasks, get_task, get_tasks

_MAEB_CITATION = """@misc{assadi2026maebmassiveaudioembedding,
  archiveprefix = {arXiv},
  author = {Adnan El Assadi and Isaac Chung and Chenghao Xiao and Roman Solomatin and Animesh Jha and Rahul Chand and Silky Singh and Kaitlyn Wang and Ali Sartaz Khan and Marc Moussa Nasser and Sufen Fong and Pengfei He and Alan Xiao and Ayush Sunil Munot and Aditya Shrivastava and Artem Gazizov and Niklas Muennighoff and Kenneth Enevoldsen},
  eprint = {2602.16008},
  primaryclass = {cs.SD},
  title = {MAEB: Massive Audio Embedding Benchmark},
  url = {https://arxiv.org/abs/2602.16008},
  year = {2026},
}"""

_MAEB_AUDIO_ONLY_TASKS = [
    # Any2AnyRetrieval (1)
    "JamAltArtistA2ARetrieval",
    # AudioClassification (11)
    "BeijingOpera",
    "BirdCLEF",
    "CREMA_D",
    "CommonLanguageAgeDetection",
    "GTZANGenre",
    "IEMOCAPGender",
    "MInDS14",
    "MridinghamTonic",
    "SIBFLEURS",
    "VoxCelebSA",
    "VoxPopuliLanguageID",
    # AudioClustering (3)
    "CREMA_DClustering",
    "VehicleSoundClustering",
    "VoxPopuliGenderClustering",
    # AudioPairClassification (3)
    "CREMADPairClassification",
    "NMSQAPairClassification",
    "VoxPopuliAccentPairClassification",
    # AudioReranking (1)
    "GTZANAudioReranking",
]

MAEB_AUDIO = Benchmark(
    name="MAEB(v1, audio-only)",
    aliases=["MAEB(beta, audio-only)", "MAEB(audio-only)"],
    display_name="MAEB Audio-Only",
    icon="https://raw.githubusercontent.com/DennisSuitters/LibreICONS/master/svg/libre-gui-activity.svg",
    tasks=get_tasks(tasks=_MAEB_AUDIO_ONLY_TASKS),
    description="Audio-only embedding quality across classification, clustering, pair classification, reranking, and retrieval tasks.",
    reference=None,
    citation=_MAEB_CITATION,
    contacts=["AdnanElAssadi56", "isaac-chung", "KennethEnevoldsen", "Samoed"],
)

MAEB = Benchmark(
    name="MAEB(v1)",
    aliases=["MAEB(beta)", "MAEB"],
    display_name="MAEB",
    icon="https://raw.githubusercontent.com/DennisSuitters/LibreICONS/master/svg/libre-gui-activity.svg",
    tasks=MTEBTasks(
        get_tasks(
            tasks=[
                # Any2AnyRetrieval (9)
                "ClothoT2ARetrieval",
                "CommonVoiceMini21T2ARetrieval",
                "FleursT2ARetrieval",
                "GigaSpeechT2ARetrieval",
                "JamAltArtistA2ARetrieval",
                "JamAltLyricA2TRetrieval",
                "MACST2ARetrieval",
                "SpokenSQuADT2ARetrieval",
                "UrbanSound8KT2ARetrieval",
                # AudioClassification (11)
                "BeijingOpera",
                "BirdCLEF",
                "CREMA_D",
                "CommonLanguageAgeDetection",
                "GTZANGenre",
                "IEMOCAPGender",
                "MInDS14",
                "MridinghamTonic",
                "SIBFLEURS",
                "VoxCelebSA",
                "VoxPopuliLanguageID",
                # AudioClustering (3)
                "CREMA_DClustering",
                "VehicleSoundClustering",
                "VoxPopuliGenderClustering",
                # AudioPairClassification (3)
                "CREMADPairClassification",
                "NMSQAPairClassification",
                "VoxPopuliAccentPairClassification",
                # AudioReranking (1)
                "GTZANAudioReranking",
                # AudioZeroshotClassification (2)
                "RavdessZeroshot",
                "SpeechCommandsZeroshotv0.02",
            ]
        )
        + (
            # AudioMultilabelClassification (1), curated only (noisy has uncurated labels)
            get_task("FSD2019Kaggle", hf_subsets=["curated"]),
        )
    ),
    description="Audio embedding quality across both audio-only and audio-text cross-modal tasks, spanning retrieval, classification, clustering, multilabel classification, pair classification, reranking, and zero-shot classification.",
    reference=None,
    citation=_MAEB_CITATION,
    contacts=["AdnanElAssadi56", "isaac-chung", "KennethEnevoldsen", "Samoed"],
    aggregations=(
        BenchmarkAggregation.MEAN_TASK,
        BenchmarkAggregation.MEAN_TASK_TYPE,
        BenchmarkAggregation.TASK_TYPES,
        CustomGrouping(
            name="Benchmark Variant",
            groups=(
                CustomGroup(
                    label="Audio-Only",
                    tasks=get_tasks(_MAEB_AUDIO_ONLY_TASKS),
                    description="This model's score restricted to audio only tasks. It is equivalent to MAEB(v1, audio-only).",
                ),
            ),
        ),
    ),
)

MVEB = Benchmark(
    name="MVEB(v1)",
    aliases=["MVEB(beta)", "MVEB"],
    display_name="MVEB",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-video.svg",
    tasks=get_tasks(
        tasks=[
            # Any2AnyRetrieval (10)
            "AVMemeExamAT2VRetrieval",
            "ActivityNetCaptionsT2VRetrieval",
            "AudioCapsAVVA2TRetrieval",
            "AudioCapsAVVT2ARetrieval",
            "MSVDT2VRetrieval",
            "VALOR32KT2VARetrieval",
            "VATEXV2ARetrieval",
            "VATEXVA2TRetrieval",
            "VGGSoundAVA2VRetrieval",
            "YouCook2T2VARetrieval",
            # VideoCentricQA (1)
            "EgoSchemaVideoCentricQA",
            # VideoClassification (6)
            "AVEDatasetClassification",
            "AVMemeAudioVideoClassification",
            "BreakfastClassification",
            "Kinetics700VA",
            "RAVDESSAVClassification",
            "UCF101VideoAudioClassification",
            # VideoClustering (2)
            "MELDEmotionAudioVideoClustering",
            "MusicAVQACLSAudioVideoClustering",
            # VideoPairClassification (2)
            "HumanAnimalCartoonVAPairClassification",
            "MusicAVQAVAPairClassification",
            # VideoZeroshotClassification (2)
            "HMDB51ZeroShot",
            "WorldSenseAudioVideoZeroShot",
        ]
    ),
    description="Audio-visual video embedding quality across retrieval, classification, clustering, pair classification, zero-shot classification, and video-centric QA, with tasks selected to maximize coverage of audio-video joint modality inputs.",
    reference=None,
    citation=r"""
@inproceedings{anonymous2026mveb,
  author = {Adnan El Assadi and Roman Solomatin and Isaac Chung and Chenghao Xiao and Deep Shah and Shriya Sudhakar and Ayush Sunil Munot and M{\'a}rton Kardos and Niklas Muennighoff and Kenneth Enevoldsen},
  booktitle = {The 2026 Conference on Empirical Methods in Natural Language Processing},
  title = {{MVEB}: Massive Video Embedding Benchmark},
  url = {https://openreview.net/forum?id=p5KyLtjRPo},
  year = {2026},
}
""",
    contacts=["AdnanElAssadi56", "isaac-chung", "KennethEnevoldsen", "Samoed"],
)

MVEB_TEXT_VIDEO = Benchmark(
    name="MVEB(text, video, v1)",
    aliases=["MVEB(text, video, beta)", "MVEB(text, video)"],
    display_name="MVEB Video-Text",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-video.svg",
    tasks=get_tasks(
        tasks=[
            # Any2AnyRetrieval (8)
            "AVMemeExamT2VRetrieval",
            "ActivityNetCaptionsT2VRetrieval",
            "AudioCapsAVT2VRetrieval",
            "DiDeMoV2TRetrieval",
            "MSVDV2TRetrieval",
            "Panda70MT2VRetrieval",
            "VALOR32KT2VRetrieval",
            "VATEXT2VRetrieval",
            # VideoCentricQA (1)
            "OmniVideoBenchVideoCentricQA",
            # VideoClassification (4)
            "AVMemeVideoClassification",
            "BreakfastClassification",
            "Kinetics700V",
            "VGGSoundV",
            # VideoClustering (1)
            "RAVDESSVideoClustering",
            # VideoPairClassification (1)
            "HumanAnimalCartoonVPairClassification",
            # VideoZeroshotClassification (4)
            "Kinetics400ZeroShot",
            "MELDVideoZeroShot",
            "UCF101VideoZeroShotClassification",
            "WorldSenseVideoZeroShot",
        ]
    ),
    description="Text and video embedding quality across retrieval, classification, clustering, pair classification, zero-shot classification, and video-centric QA, for models without an audio encoder.",
    reference=None,
    citation=r"""
@inproceedings{anonymous2026mveb,
  author = {Adnan El Assadi and Roman Solomatin and Isaac Chung and Chenghao Xiao and Deep Shah and Shriya Sudhakar and Ayush Sunil Munot and M{\'a}rton Kardos and Niklas Muennighoff and Kenneth Enevoldsen},
  booktitle = {The 2026 Conference on Empirical Methods in Natural Language Processing},
  title = {{MVEB}: Massive Video Embedding Benchmark},
  url = {https://openreview.net/forum?id=p5KyLtjRPo},
  year = {2026},
}
""",
    contacts=["AdnanElAssadi56", "isaac-chung", "KennethEnevoldsen", "Samoed"],
)

MVEB_VIDEO = Benchmark(
    name="MVEB(video, v1)",
    aliases=["MVEB(video, beta)", "MVEB(video)"],
    display_name="MVEB Video-Only",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-video.svg",
    tasks=get_tasks(
        tasks=[
            # VideoClassification (6)
            "AVMemeVideoClassification",
            "BreakfastClassification",
            "HMDB51Classification",
            "Kinetics600V",
            "MELDVideoClassification",
            "WorldSenseVideoClassification",
            # VideoPairClassification (3)
            "HumanAnimalCartoonVPairClassification",
            "MusicAVQAVPairClassification",
            "RAVDESSAVVPairClassification",
        ]
    ),
    description="Video-only embedding quality across classification and pair classification, for encoders without a text component. Retrieval, QA, and zero-shot tasks are excluded as they require a text encoder.",
    reference=None,
    citation=r"""
@inproceedings{anonymous2026mveb,
  author = {Adnan El Assadi and Roman Solomatin and Isaac Chung and Chenghao Xiao and Deep Shah and Shriya Sudhakar and Ayush Sunil Munot and M{\'a}rton Kardos and Niklas Muennighoff and Kenneth Enevoldsen},
  booktitle = {The 2026 Conference on Empirical Methods in Natural Language Processing},
  title = {{MVEB}: Massive Video Embedding Benchmark},
  url = {https://openreview.net/forum?id=p5KyLtjRPo},
  year = {2026},
}
""",
    contacts=["AdnanElAssadi56", "isaac-chung", "KennethEnevoldsen", "Samoed"],
)
