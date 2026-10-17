"""Image and visual-document embedding benchmarks (MIEB, ViDoRe, ...)."""

from mteb.benchmarks.benchmark import (
    Benchmark,
    BenchmarkAggregation,
    CustomGroup,
    CustomGrouping,
    MIEBBenchmark,
    VidoreBenchmark,
)
from mteb.get_tasks import MTEBTasks, get_tasks

MIEB_common_tasks = [
    # Image Classification
    "Birdsnap",  # fine
    "Caltech101",  # fine
    "CIFAR10",  # coarse
    "CIFAR100",  # fine
    "Country211",  # fine
    "DTD",  # coarse
    "EuroSAT",  # coarse
    "FER2013",  # coarse
    "FGVCAircraft",  # fine
    "Food101Classification",  # fine
    "GTSRB",  # coarse
    "Imagenet1k",  # fine
    "MNIST",  # coarse
    "OxfordFlowersClassification",  # fine
    "OxfordPets",  # fine
    "PatchCamelyon",  # coarse
    "RESISC45",  # fine
    "StanfordCars",  # fine
    "STL10",  # coarse
    "SUN397",  # fine
    "UCF101",  # fine
    # MultiLabelClassification
    "VOC2007",  # coarse
    # Clustering
    "CIFAR10Clustering",
    "CIFAR100Clustering",
    "ImageNetDog15Clustering",
    "ImageNet10Clustering",
    "TinyImageNetClustering",
    # ZeroShotClassification
    "BirdsnapZeroShot",
    "Caltech101ZeroShot",
    "CIFAR10ZeroShot",
    "CIFAR100ZeroShot",
    "CLEVRZeroShot",
    "CLEVRCountZeroShot",
    "Country211ZeroShot",
    "DTDZeroShot",
    "EuroSATZeroShot",
    "FER2013ZeroShot",
    "FGVCAircraftZeroShot",
    "Food101ZeroShot",
    "GTSRBZeroShot",
    "Imagenet1kZeroShot",
    "MNISTZeroShot",
    "OxfordPetsZeroShot",
    "PatchCamelyonZeroShot",
    "RenderedSST2",
    "RESISC45ZeroShot",
    "StanfordCarsZeroShot",
    "STL10ZeroShot",
    "SUN397ZeroShot",
    "UCF101ZeroShot",
    # Any2AnyMultipleChoice
    "BLINKIT2IMultiChoice",
    "BLINKIT2TMultiChoice",
    "CVBenchCount",
    "CVBenchRelation",
    "CVBenchDepth",
    "CVBenchDistance",
    # Compositionality
    "AROCocoOrder",
    "AROFlickrOrder",
    "AROVisualAttribution",
    "AROVisualRelation",
    "SugarCrepe",
    "Winoground",
    "ImageCoDe",
    # VisualSTS
    "STS12VisualSTS",
    "STS13VisualSTS",
    "STS14VisualSTS",
    "STS15VisualSTS",
    "STS16VisualSTS",
    # Any2AnyRetrieval
    "BLINKIT2IRetrieval",
    "BLINKIT2TRetrieval",
    "CIRRIT2IRetrieval",
    "CUB200I2IRetrieval",
    "EDIST2ITRetrieval",
    "Fashion200kI2TRetrieval",
    "Fashion200kT2IRetrieval",
    "FashionIQIT2IRetrieval",
    "Flickr30kI2TRetrieval",
    "Flickr30kT2IRetrieval",
    "FORBI2IRetrieval",
    "GLDv2I2IRetrieval",
    "GLDv2I2TRetrieval",
    "HatefulMemesI2TRetrieval",
    "HatefulMemesT2IRetrieval",
    "ImageCoDeT2IRetrieval",
    "InfoSeekIT2ITRetrieval",
    "InfoSeekIT2TRetrieval",
    "MemotionI2TRetrieval",
    "MemotionT2IRetrieval",
    "METI2IRetrieval",
    "MSCOCOI2TRetrieval",
    "MSCOCOT2IRetrieval",
    "NIGHTSI2IRetrieval",
    "OVENIT2ITRetrieval",
    "OVENIT2TRetrieval",
    "ROxfordEasyI2IRetrieval",
    "ROxfordMediumI2IRetrieval",
    "ROxfordHardI2IRetrieval",
    "RP2kI2IRetrieval",
    "RParisEasyI2IRetrieval",
    "RParisMediumI2IRetrieval",
    "RParisHardI2IRetrieval",
    "SciMMIRI2TRetrieval",
    "SciMMIRT2IRetrieval",
    "SketchyI2IRetrieval",
    "SOPI2IRetrieval",
    "StanfordCarsI2IRetrieval",
    "TUBerlinT2IRetrieval",
    "VidoreArxivQARetrieval",
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTabfquadRetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreShiftProjectRetrieval",
    "VidoreSyntheticDocQAAIRetrieval",
    "VidoreSyntheticDocQAEnergyRetrieval",
    "VidoreSyntheticDocQAGovernmentReportsRetrieval",
    "VidoreSyntheticDocQAHealthcareIndustryRetrieval",
    "VisualNewsI2TRetrieval",
    "VisualNewsT2IRetrieval",
    "VizWizIT2TRetrieval",
    "VQA2IT2TRetrieval",
    "WebQAT2ITRetrieval",
    "WebQAT2TRetrieval",
]

_MIEB_LITE_TASKS = [
    # Image Classification
    "Country211",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "OxfordPets",
    "PatchCamelyon",
    "RESISC45",
    "SUN397",
    # Clustering
    "ImageNetDog15Clustering",
    "TinyImageNetClustering",
    # ZeroShotClassification
    "CIFAR100ZeroShot",
    "Country211ZeroShot",
    "FER2013ZeroShot",
    "FGVCAircraftZeroShot",
    "Food101ZeroShot",
    "OxfordPetsZeroShot",
    "StanfordCarsZeroShot",
    # Any2AnyMultipleChoice
    "BLINKIT2IMultiChoice",
    "CVBenchCount",
    "CVBenchRelation",
    "CVBenchDepth",
    "CVBenchDistance",
    # ImageTextPairClassification
    "AROCocoOrder",
    "AROFlickrOrder",
    "AROVisualAttribution",
    "AROVisualRelation",
    "Winoground",
    "ImageCoDe",
    # VisualSTS
    "STS13VisualSTS",
    "STS15VisualSTS",
    "VisualSTS17Multilingual",
    "VisualSTS-b-Multilingual",
    # Any2AnyRetrieval
    "CIRRIT2IRetrieval",
    "CUB200I2IRetrieval",
    "Fashion200kI2TRetrieval",
    "HatefulMemesI2TRetrieval",
    "InfoSeekIT2TRetrieval",
    "NIGHTSI2IRetrieval",
    "OVENIT2TRetrieval",
    "RP2kI2IRetrieval",
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTabfquadRetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreShiftProjectRetrieval",
    "VidoreSyntheticDocQAAIRetrieval",
    "VisualNewsI2TRetrieval",
    "VQA2IT2TRetrieval",
    "WebQAT2ITRetrieval",
    "WITT2IRetrieval",
    "XM3600T2IRetrieval",
]

MIEB_ENG = MIEBBenchmark(
    name="MIEB(eng)",
    display_name="Image-Text, English",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-picture.svg",
    tasks=get_tasks(
        tasks=MIEB_common_tasks
        + [
            "VisualSTS17Eng",
            "VisualSTS-b-Eng",
        ],
    ),
    description="English image embedding quality across image classification (zero-shot and linear probing), clustering, retrieval, compositionality evaluation, document understanding, visual STS, and CV-centric tasks.",
    reference="https://arxiv.org/abs/2504.10471",
    contacts=["gowitheflow-1998", "isaac-chung"],
    citation=r"""
@inproceedings{xiao2025mieb,
  author = {Xiao, Chenghao and Chung, Isaac and Kerboua, Imene and Stirling, Jamie and Zhang, Xin and Kardos, M\'arton and Solomatin, Roman and Al Moubayed, Noura and Enevoldsen, Kenneth and Muennighoff, Niklas},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month = {October},
  pages = {22187-22198},
  title = {MIEB: Massive Image Embedding Benchmark},
  year = {2025},
}
""",
)

MIEB_MULTILINGUAL = MIEBBenchmark(
    name="MIEB(Multilingual)",
    display_name="Image-Text, Multilingual",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-pictures.svg",
    tasks=get_tasks(
        tasks=MIEB_common_tasks
        + [
            "WITT2IRetrieval",
            "XFlickr30kCoT2IRetrieval",
            "XM3600T2IRetrieval",
            "VisualSTS17Eng",
            "VisualSTS-b-Eng",
            "VisualSTS17Multilingual",
            "VisualSTS-b-Multilingual",
        ],
    ),
    description="Multilingual image embedding quality across 39 languages, spanning image classification (zero-shot and linear probing), clustering, retrieval, compositionality evaluation, document understanding, visual STS, and CV-centric tasks. Extends MIEB(eng) with multilingual retrieval datasets and the multilingual portions of VisualSTS-b and VisualSTS-16.",
    reference="https://arxiv.org/abs/2504.10471",
    contacts=["gowitheflow-1998", "isaac-chung"],
    citation=r"""
@inproceedings{xiao2025mieb,
  author = {Xiao, Chenghao and Chung, Isaac and Kerboua, Imene and Stirling, Jamie and Zhang, Xin and Kardos, M\'arton and Solomatin, Roman and Al Moubayed, Noura and Enevoldsen, Kenneth and Muennighoff, Niklas},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month = {October},
  pages = {22187-22198},
  title = {MIEB: Massive Image Embedding Benchmark},
  year = {2025},
}
""",
    aggregations=(
        BenchmarkAggregation.MEAN_TASK_TYPE,
        BenchmarkAggregation.TASK_TYPES,
        CustomGrouping(
            name="MIEB(eng) Task Mean",
            groups=(
                CustomGroup(
                    label="Task Mean",
                    tasks=get_tasks(
                        MIEB_common_tasks + ["VisualSTS17Eng", "VisualSTS-b-Eng"]
                    ),
                    description="Unweighted mean across MIEB(eng)'s task set (a strict subset of MIEB(Multilingual)'s) — not the same statistic as MIEB(eng)'s own Mean(TaskType).",
                ),
            ),
        ),
        CustomGrouping(
            name="MIEB(lite) Task Mean",
            groups=(
                CustomGroup(
                    label="Task Mean",
                    tasks=get_tasks(_MIEB_LITE_TASKS),
                    description="Unweighted mean across MIEB(lite)'s task set (a strict subset of MIEB(Multilingual)'s) — not the same statistic as MIEB(lite)'s own Mean(TaskType).",
                ),
            ),
        ),
    ),
)

MIEB_LITE = MIEBBenchmark(
    name="MIEB(lite)",
    display_name="Image-Text, Lite",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-map-landscape.svg",
    tasks=get_tasks(tasks=_MIEB_LITE_TASKS),
    description="Multilingual image embedding quality across the same task types as MIEB(Multilingual), designed to be run at a fraction of the cost while maintaining relative model rankings.",
    reference="https://arxiv.org/abs/2504.10471",
    contacts=["gowitheflow-1998", "isaac-chung"],
    citation=r"""
@inproceedings{xiao2025mieb,
  author = {Xiao, Chenghao and Chung, Isaac and Kerboua, Imene and Stirling, Jamie and Zhang, Xin and Kardos, M\'arton and Solomatin, Roman and Al Moubayed, Noura and Enevoldsen, Kenneth and Muennighoff, Niklas},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month = {October},
  pages = {22187-22198},
  title = {MIEB: Massive Image Embedding Benchmark},
  year = {2025},
}
""",
)

MIEB_IMG = MIEBBenchmark(
    name="MIEB(Img)",
    display_name="Image only",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-pictures.svg",
    tasks=get_tasks(
        tasks=[
            "CUB200I2IRetrieval",
            "FORBI2IRetrieval",
            "GLDv2I2IRetrieval",
            "METI2IRetrieval",
            "NIGHTSI2IRetrieval",
            "ROxfordEasyI2IRetrieval",
            "ROxfordMediumI2IRetrieval",
            "ROxfordHardI2IRetrieval",
            "RP2kI2IRetrieval",
            "RParisEasyI2IRetrieval",
            "RParisMediumI2IRetrieval",
            "RParisHardI2IRetrieval",
            "SketchyI2IRetrieval",
            "SOPI2IRetrieval",
            "StanfordCarsI2IRetrieval",
            "Birdsnap",
            "Caltech101",
            "CIFAR10",
            "CIFAR100",
            "Country211",
            "DTD",
            "EuroSAT",
            "FER2013",
            "FGVCAircraft",
            "Food101Classification",
            "GTSRB",
            "Imagenet1k",
            "MNIST",
            "OxfordFlowersClassification",
            "OxfordPets",
            "PatchCamelyon",
            "RESISC45",
            "StanfordCars",
            "STL10",
            "SUN397",
            "UCF101",
            "CIFAR10Clustering",
            "CIFAR100Clustering",
            "ImageNetDog15Clustering",
            "ImageNet10Clustering",
            "TinyImageNetClustering",
            "VOC2007",
            "STS12VisualSTS",
            "STS13VisualSTS",
            "STS14VisualSTS",
            "STS15VisualSTS",
            "STS16VisualSTS",
            "STS17MultilingualVisualSTS",
            "STSBenchmarkMultilingualVisualSTS",
        ],
    ),
    description="Image-only embedding quality across retrieval, classification, clustering, and visual STS, excluding tasks that require a text encoder.",
    reference="https://arxiv.org/abs/2504.10471",
    citation=r"""
@inproceedings{xiao2025mieb,
  author = {Xiao, Chenghao and Chung, Isaac and Kerboua, Imene and Stirling, Jamie and Zhang, Xin and Kardos, M\'arton and Solomatin, Roman and Al Moubayed, Noura and Enevoldsen, Kenneth and Muennighoff, Niklas},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month = {October},
  pages = {22187-22198},
  title = {MIEB: Massive Image Embedding Benchmark},
  year = {2025},
}
""",
    contacts=["gowitheflow-1998", "isaac-chung"],
)

VIDORE = Benchmark(
    name="ViDoRe(v1)",
    tasks=get_tasks(
        tasks=[
            "VidoreArxivQARetrieval",
            "VidoreDocVQARetrieval",
            "VidoreInfoVQARetrieval",
            "VidoreTabfquadRetrieval",
            "VidoreTatdqaRetrieval",
            "VidoreShiftProjectRetrieval",
            "VidoreSyntheticDocQAAIRetrieval",
            "VidoreSyntheticDocQAEnergyRetrieval",
            "VidoreSyntheticDocQAGovernmentReportsRetrieval",
            "VidoreSyntheticDocQAHealthcareIndustryRetrieval",
        ],
    ),
    description="Visual document retrieval across diverse document types and domains, matching natural language queries to document page images.",
    reference="https://arxiv.org/abs/2407.01449",
    citation=r"""
@article{faysse2024colpali,
  author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal = {arXiv preprint arXiv:2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  year = {2024},
}
""",
)

VIDORE_V2 = Benchmark(
    name="ViDoRe(v2)",
    tasks=get_tasks(
        tasks=[
            "Vidore2ESGReportsRetrieval",
            "Vidore2EconomicsReportsRetrieval",
            "Vidore2BioMedicalLecturesRetrieval",
            "Vidore2ESGReportsHLRetrieval",
        ],
    ),
    description="Visual document retrieval across ESG reports, economics reports, biomedical lectures, and related enterprise document types, matching natural language queries to document page images.",
    reference="https://arxiv.org/abs/2407.01449",
    citation=r"""
@article{mace2025vidorev2,
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
)

VIDORE_V3 = VidoreBenchmark(
    name="ViDoRe(v3)",
    display_name="ViDoRe V3",
    language_view=[
        "deu-Latn",
        "eng-Latn",
        "fra-Latn",
        "ita-Latn",
        "por-Latn",
        "spa-Latn",
    ],
    icon="https://cdn-uploads.huggingface.co/production/uploads/66e16a677c2eb2da5109fb5c/x99xqw__fl2UaPbiIdC_f.png",
    tasks=get_tasks(
        tasks=[
            "Vidore3FinanceEnRetrieval",
            "Vidore3IndustrialRetrieval",
            "Vidore3ComputerScienceRetrieval",
            "Vidore3PharmaceuticalsRetrieval",
            "Vidore3HrRetrieval",
            "Vidore3FinanceFrRetrieval",
            "Vidore3PhysicsRetrieval",
            "Vidore3EnergyRetrieval",
            "Vidore3TelecomRetrieval",
            "Vidore3NuclearRetrieval",
        ]
    ),
    description="Visual document retrieval across multi-modal enterprise documents spanning finance, industrial, computer science, pharmaceutical, and other professional domains. Includes both open and closed datasets; to submit results on private tasks, please [open an issue](https://github.com/embeddings-benchmark/mteb/issues?template=eval_request.yaml).",
    reference="https://arxiv.org/abs/2601.08620",
    citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
)

VIDORE_V3_1 = VidoreBenchmark(
    name="ViDoRe(v3.1)",
    display_name="ViDoRe v3.1",
    language_view=[
        "deu-Latn",
        "eng-Latn",
        "fra-Latn",
        "ita-Latn",
        "por-Latn",
        "spa-Latn",
    ],
    icon="https://cdn-uploads.huggingface.co/production/uploads/66e16a677c2eb2da5109fb5c/x99xqw__fl2UaPbiIdC_f.png",
    tasks=get_tasks(
        tasks=[
            "Vidore3FinanceEnRetrieval.v2",
            "Vidore3IndustrialRetrieval.v2",
            "Vidore3ComputerScienceRetrieval.v2",
            "Vidore3PharmaceuticalsRetrieval.v2",
            "Vidore3HrRetrieval.v2",
            "Vidore3FinanceFrRetrieval.v2",
            "Vidore3PhysicsRetrieval.v2",
            "Vidore3EnergyRetrieval.v2",
            "Vidore3TelecomRetrieval.v2",
            "Vidore3NuclearRetrieval.v2",
        ]
    ),
    description="Visual document retrieval across multi-modal enterprise documents spanning finance, industrial, computer science, pharmaceutical, and other professional domains. Includes both open and closed datasets; to submit results on private tasks, please [open an issue](https://github.com/embeddings-benchmark/mteb/issues?template=eval_request.yaml). v3.1 adds markdown derived from OCR to support text-only and joint image-text baselines.",
    reference="https://arxiv.org/abs/2601.08620",
    citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
)

_VIDORE_V3_1_RERANKING_BM25S = [
    "Vidore3ComputerScienceBM25sReranking.v2",
    "Vidore3EnergyBM25sReranking.v2",
    "Vidore3FinanceEnBM25sReranking.v2",
    "Vidore3FinanceFrBM25sReranking.v2",
    "Vidore3HrBM25sReranking.v2",
    "Vidore3IndustrialBM25sReranking.v2",
    "Vidore3PharmaceuticalsBM25sReranking.v2",
    "Vidore3PhysicsBM25sReranking.v2",
    "Vidore3TelecomBM25sReranking.v2",
    "Vidore3NuclearBM25sReranking.v2",
]

_VIDORE_V3_1_RERANKING_BGEM3 = [
    "Vidore3ComputerScienceBGEm3Reranking.v2",
    "Vidore3EnergyBGEm3Reranking.v2",
    "Vidore3FinanceEnBGEm3Reranking.v2",
    "Vidore3FinanceFrBGEm3Reranking.v2",
    "Vidore3HrBGEm3Reranking.v2",
    "Vidore3IndustrialBGEm3Reranking.v2",
    "Vidore3PharmaceuticalsBGEm3Reranking.v2",
    "Vidore3PhysicsBGEm3Reranking.v2",
    "Vidore3TelecomBGEm3Reranking.v2",
    "Vidore3NuclearBGEm3Reranking.v2",
]

_VIDORE_V3_1_RERANKING_QWEN3_VL_EMBEDDING_2B = [
    "Vidore3ComputerScienceQwen3VLEmbedding2BReranking.v2",
    "Vidore3EnergyQwen3VLEmbedding2BReranking.v2",
    "Vidore3FinanceEnQwen3VLEmbedding2BReranking.v2",
    "Vidore3FinanceFrQwen3VLEmbedding2BReranking.v2",
    "Vidore3HrQwen3VLEmbedding2BReranking.v2",
    "Vidore3IndustrialQwen3VLEmbedding2BReranking.v2",
    "Vidore3PharmaceuticalsQwen3VLEmbedding2BReranking.v2",
    "Vidore3PhysicsQwen3VLEmbedding2BReranking.v2",
    "Vidore3TelecomQwen3VLEmbedding2BReranking.v2",
    "Vidore3NuclearQwen3VLEmbedding2BReranking.v2",
]

_VIDORE_V3_1_RERANKING_NEMOTRON_3_5_EMBED_8B = [
    "Vidore3ComputerScienceNemotron35Embed8BReranking.v2",
    "Vidore3EnergyNemotron35Embed8BReranking.v2",
    "Vidore3FinanceEnNemotron35Embed8BReranking.v2",
    "Vidore3FinanceFrNemotron35Embed8BReranking.v2",
    "Vidore3HrNemotron35Embed8BReranking.v2",
    "Vidore3IndustrialNemotron35Embed8BReranking.v2",
    "Vidore3PharmaceuticalsNemotron35Embed8BReranking.v2",
    "Vidore3PhysicsNemotron35Embed8BReranking.v2",
    "Vidore3TelecomNemotron35Embed8BReranking.v2",
    "Vidore3NuclearNemotron35Embed8BReranking.v2",
]

VIDORE_V3_1_RERANKING = VidoreBenchmark(
    name="ViDoRe(v3.1, reranking)",
    aliases=["ViDoRe(v3.1, reranking, beta)"],
    display_name="ViDoRe v3.1 Reranking",
    language_view=[
        "deu-Latn",
        "eng-Latn",
        "fra-Latn",
        "ita-Latn",
        "por-Latn",
        "spa-Latn",
    ],
    icon="https://cdn-uploads.huggingface.co/production/uploads/66e16a677c2eb2da5109fb5c/x99xqw__fl2UaPbiIdC_f.png",
    tasks=get_tasks(
        tasks=_VIDORE_V3_1_RERANKING_BM25S
        + _VIDORE_V3_1_RERANKING_BGEM3
        + _VIDORE_V3_1_RERANKING_QWEN3_VL_EMBEDDING_2B
        + _VIDORE_V3_1_RERANKING_NEMOTRON_3_5_EMBED_8B
    ),
    description="Reranking of visual documents on the ViDoRe v3.1 domains. For each domain, the candidates are the top-50 pages retrieved by one of four first-stage retrievers; BM25, bge-m3, Qwen3-VL-Embedding-2B and Nemotron-3.5-Embed-8B, so rerankers are compared on candidate lists from various model types. Includes both open and closed datasets; to submit results on private tasks, please [open an issue](https://github.com/embeddings-benchmark/mteb/issues?template=eval_request.yaml).",
    reference="https://arxiv.org/abs/2601.08620",
    citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
    aggregations=(
        BenchmarkAggregation.MEAN_TASK,
        BenchmarkAggregation.PUBLIC_PRIVATE,
        CustomGrouping(
            name="First-stage retriever",
            groups=(
                CustomGroup(
                    label="BM25s",
                    tasks=get_tasks(tasks=_VIDORE_V3_1_RERANKING_BM25S),
                    description="Mean over the domains when reranking the top-50 candidates retrieved by mteb/baseline-bm25s.",
                ),
                CustomGroup(
                    label="BGE-M3",
                    tasks=get_tasks(tasks=_VIDORE_V3_1_RERANKING_BGEM3),
                    description="Mean over the domains when reranking the top-50 candidates retrieved by BAAI/bge-m3.",
                ),
                CustomGroup(
                    label="Qwen3-VL-Embedding-2B",
                    tasks=get_tasks(tasks=_VIDORE_V3_1_RERANKING_QWEN3_VL_EMBEDDING_2B),
                    description="Mean over the domains when reranking the top-50 candidates retrieved by Qwen/Qwen3-VL-Embedding-2B.",
                ),
                CustomGroup(
                    label="Nemotron-3.5-Embed-8B",
                    tasks=get_tasks(tasks=_VIDORE_V3_1_RERANKING_NEMOTRON_3_5_EMBED_8B),
                    description="Mean over the domains when reranking the top-50 candidates retrieved by nvidia/Nemotron-3.5-Embed-8B-BF16.",
                ),
            ),
        ),
    ),
)

VISUAL_DOCUMENT_RETRIEVAL = Benchmark(
    name="ViDoRe(v1&v2)",
    aliases=["VisualDocumentRetrieval"],
    display_name="ViDoRe (V1&V2)",
    tasks=get_tasks(
        tasks=[
            # v1
            "VidoreArxivQARetrieval",
            "VidoreDocVQARetrieval",
            "VidoreInfoVQARetrieval",
            "VidoreTabfquadRetrieval",
            "VidoreTatdqaRetrieval",
            "VidoreShiftProjectRetrieval",
            "VidoreSyntheticDocQAAIRetrieval",
            "VidoreSyntheticDocQAEnergyRetrieval",
            "VidoreSyntheticDocQAGovernmentReportsRetrieval",
            "VidoreSyntheticDocQAHealthcareIndustryRetrieval",
            # v2
            "Vidore2ESGReportsRetrieval",
            "Vidore2EconomicsReportsRetrieval",
            "Vidore2BioMedicalLecturesRetrieval",
            "Vidore2ESGReportsHLRetrieval",
        ],
    ),
    description="Visual document retrieval across diverse document types and domains, combining the ViDoRe v1 and v2 task sets.",
    reference="https://arxiv.org/abs/2407.01449",
    citation=r"""
@article{mace2025vidorev2,
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
    aggregations=(BenchmarkAggregation.MEAN_TASK,),
    show_zero_shot=False,
)

JINA_VDR = Benchmark(
    name="JinaVDR",
    display_name="Jina Visual Document Retrieval",
    tasks=get_tasks(
        tasks=[
            "JinaVDRMedicalPrescriptionsRetrieval",
            "JinaVDRStanfordSlideRetrieval",
            "JinaVDRDonutVQAISynHMPRetrieval",
            "JinaVDRTableVQARetrieval",
            "JinaVDRChartQARetrieval",
            "JinaVDRTQARetrieval",
            "JinaVDROpenAINewsRetrieval",
            "JinaVDREuropeanaDeNewsRetrieval",
            "JinaVDREuropeanaEsNewsRetrieval",
            "JinaVDREuropeanaItScansRetrieval",
            "JinaVDREuropeanaNlLegalRetrieval",
            "JinaVDRHindiGovVQARetrieval",
            "JinaVDRAutomobileCatelogRetrieval",
            "JinaVDRBeveragesCatalogueRetrieval",
            "JinaVDRRamensBenchmarkRetrieval",
            "JinaVDRJDocQARetrieval",
            "JinaVDRHungarianDocQARetrieval",
            "JinaVDRArabicChartQARetrieval",
            "JinaVDRArabicInfographicsVQARetrieval",
            "JinaVDROWIDChartsRetrieval",
            "JinaVDRMPMQARetrieval",
            "JinaVDRJina2024YearlyBookRetrieval",
            "JinaVDRWikimediaCommonsMapsRetrieval",
            "JinaVDRPlotQARetrieval",
            "JinaVDRMMTabRetrieval",
            "JinaVDRCharXivOCRRetrieval",
            "JinaVDRStudentEnrollmentSyntheticRetrieval",
            "JinaVDRGitHubReadmeRetrieval",
            "JinaVDRTweetStockSyntheticsRetrieval",
            "JinaVDRAirbnbSyntheticRetrieval",
            "JinaVDRShanghaiMasterPlanRetrieval",
            "JinaVDRWikimediaCommonsDocumentsRetrieval",
            "JinaVDREuropeanaFrNewsRetrieval",
            "JinaVDRDocQAHealthcareIndustryRetrieval",
            "JinaVDRDocQAAI",
            "JinaVDRShiftProjectRetrieval",
            "JinaVDRTatQARetrieval",
            "JinaVDRInfovqaRetrieval",
            "JinaVDRDocVQARetrieval",
            "JinaVDRDocQAGovReportRetrieval",
            "JinaVDRTabFQuadRetrieval",
            "JinaVDRDocQAEnergyRetrieval",
            "JinaVDRArxivQARetrieval",
        ],
    ),
    description="Visual document retrieval across multilingual, domain-diverse, and layout-rich document types, spanning medical, legal, financial, technical, and other domains across multiple languages.",
    reference="https://arxiv.org/abs/2506.18902",
    citation=r"""@misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
  archiveprefix = {arXiv},
  author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
  eprint = {2506.18902},
  primaryclass = {cs.AI},
  title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
  url = {https://arxiv.org/abs/2506.18902},
  year = {2025},
}""",
)

KOVIDORE_V2 = Benchmark(
    name="KoViDoRe(v2)",
    display_name="KoViDoRe v2",
    tasks=get_tasks(
        tasks=[
            "KoVidore2CybersecurityRetrieval",
            "KoVidore2EconomicRetrieval",
            "KoVidore2EnergyRetrieval",
            "KoVidore2HrRetrieval",
        ]
    ),
    description="Korean visual document retrieval across enterprise document domains including cybersecurity, economics, energy, and HR.",
    reference="https://github.com/whybe-choi/kovidore-data-generator",
    citation=r"""
@inproceedings{choi-etal-2026-kovidore,
  address = {San Diego, USA},
  author = {Choi, Yongbin  and
Song, Yongwoo  and
Sung, Mujeen},
  booktitle = {Proceedings of the 2nd Workshop on Multimodal Augmented Generation via Multimodal Retrieval ({MAGM}a{R} 2026)},
  doi = {10.18653/v1/2026.magmar-main.11},
  editor = {Murray, Kenton  and
Kriz, Reno},
  isbn = {979-8-89176-425-5},
  month = jul,
  pages = {54--80},
  publisher = {Association for Computational Linguistics},
  title = {{K}o{V}i{D}o{R}e: A Benchmark for {K}orean Visual Document Retrieval},
  url = {https://aclanthology.org/2026.magmar-main.11/},
  year = {2026},
}
""",
)

VISRAG_RETRIEVAL = Benchmark(
    name="VisRAG Retrieval(v1)",
    aliases=["VisRAG", "VisRAG Retrieval", "VisRag", "VisRAG(v1)"],
    display_name="VisRAG Retrieval",
    tasks=MTEBTasks(
        get_tasks(
            tasks=[
                "VisRAGRetArxivQA",
                "VisRAGRetChartQA",
                "VisRAGRetInfoVQA",
                "VisRAGRetPlotQA",
                "VisRAGRetMPDocVQA",
                "VisRAGRetSlideVQA",
            ],
        )
    ),
    description="Visual retrieval augmented generation quality across document types including scientific papers, charts, infographics, plots, and multi-page documents.",
    reference="https://huggingface.co/collections/openbmb/visrag",
    citation=r"""
@misc{yu2025visragvisionbasedretrievalaugmentedgeneration,
  archiveprefix = {arXiv},
  author = {Shi Yu and Chaoyue Tang and Bokai Xu and Junbo Cui and Junhao Ran and Yukun Yan and Zhenghao Liu and Shuo Wang and Xu Han and Zhiyuan Liu and Maosong Sun},
  eprint = {2410.10594},
  primaryclass = {cs.IR},
  title = {VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents},
  url = {https://arxiv.org/abs/2410.10594},
  year = {2025},
}
""",
)
