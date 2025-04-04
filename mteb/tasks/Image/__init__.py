from __future__ import annotations

from .Any2AnyMultiChoice import (
    BLINKIT2IMultiChoice,
    BLINKIT2TMultiChoice,
    CVBenchCount,
    CVBenchDepth,
    CVBenchDistance,
    CVBenchRelation,
)
from .Any2AnyRetrieval import (
    CUB200I2I,
    FORBI2I,
    BLINKIT2IRetrieval,
    BLINKIT2TRetrieval,
    CIRRIT2IRetrieval,
    EDIST2ITRetrieval,
    EncyclopediaVQAIT2ITRetrieval,
    Fashion200kI2TRetrieval,
    Fashion200kT2IRetrieval,
    FashionIQIT2IRetrieval,
    Flickr30kI2TRetrieval,
    Flickr30kT2IRetrieval,
    GLDv2I2IRetrieval,
    GLDv2I2TRetrieval,
    HatefulMemesI2TRetrieval,
    HatefulMemesT2IRetrieval,
    ImageCoDeT2IRetrieval,
    InfoSeekIT2ITRetrieval,
    InfoSeekIT2TRetrieval,
    LLaVAIT2TRetrieval,
    MemotionI2TRetrieval,
    MemotionT2IRetrieval,
    METI2IRetrieval,
    MSCOCOI2TRetrieval,
    MSCOCOT2IRetrieval,
    NIGHTSI2IRetrieval,
    OKVQAIT2TRetrieval,
    OVENIT2ITRetrieval,
    OVENIT2TRetrieval,
    ReMuQIT2TRetrieval,
    ROxfordEasyI2IRetrieval,
    ROxfordHardI2IRetrieval,
    ROxfordMediumI2IRetrieval,
    RP2kI2IRetrieval,
    RParisEasyI2IRetrieval,
    RParisHardI2IRetrieval,
    RParisMediumI2IRetrieval,
    SciMMIRI2TRetrieval,
    SciMMIRT2IRetrieval,
    SketchyI2IRetrieval,
    SOPI2IRetrieval,
    StanfordCarsI2I,
    TUBerlinT2IRetrieval,
    VidoreArxivQARetrieval,
    VidoreDocVQARetrieval,
    VidoreInfoVQARetrieval,
    VidoreShiftProjectRetrieval,
    VidoreSyntheticDocQAAIRetrieval,
    VidoreSyntheticDocQAEnergyRetrieval,
    VidoreSyntheticDocQAGovernmentReportsRetrieval,
    VidoreSyntheticDocQAHealthcareIndustryRetrieval,
    VidoreTabfquadRetrieval,
    VidoreTatdqaRetrieval,
    VisualNewsI2TRetrieval,
    VisualNewsT2IRetrieval,
    VizWizIT2TRetrieval,
    VQA2IT2TRetrieval,
    WebQAT2ITRetrieval,
    WebQAT2TRetrieval,
    WITT2IRetrieval,
    XFlickr30kCoT2IRetrieval,
    XM3600T2IRetrieval,
)
from .Clustering import (
    CIFAR10Clustering,
    CIFAR100Clustering,
    ImageNet10Clustering,
    ImageNetDog15Clustering,
    TinyImageNet,
)
from .ImageClassification import (
    BirdsnapClassification,
    Caltech101Classification,
    CIFAR10Classification,
    CIFAR100Classification,
    Country211Classification,
    DTDClassification,
    EuroSATClassification,
    FER2013Classification,
    FGVCAircraftClassification,
    Food101Classification,
    GTSRBClassification,
    Imagenet1kClassification,
    MNISTClassification,
    OxfordFlowersClassification,
    OxfordPetsClassification,
    PatchCamelyonClassification,
    RESISC45Classification,
    StanfordCarsClassification,
    STL10Classification,
    SUN397Classification,
    UCF101Classification,
)
from .ImageMultilabelClassification import VOC2007Classification
from .ImageTextPairClassification import (
    AROCocoOrder,
    AROFlickrOrder,
    AROVisualAttribution,
    AROVisualRelation,
    ImageCoDe,
    SugarCrepe,
    Winoground,
)
from .VisualSTS import (
    STS12VisualSTS,
    STS13VisualSTS,
    STS14VisualSTS,
    STS15VisualSTS,
    STS16VisualSTS,
    STS17MultilingualVisualSTS,
    STSBenchmarkMultilingualVisualSTS,
)
from .ZeroShotClassification import (
    CLEVR,
    BirdsnapZeroShotClassification,
    Caltech101ZeroShotClassification,
    CIFAR10ZeroShotClassification,
    CIFAR100ZeroShotClassification,
    CLEVRCount,
    Country211ZeroShotClassification,
    DTDZeroShotClassification,
    EuroSATZeroShotClassification,
    FER2013ZeroShotClassification,
    FGVCAircraftZeroShotClassification,
    Food101ZeroShotClassification,
    GTSRBZeroShotClassification,
    Imagenet1kZeroShotClassification,
    MNISTZeroShotClassification,
    OxfordPetsZeroShotClassification,
    PatchCamelyonZeroShotClassification,
    RenderedSST2,
    RESISC45ZeroShotClassification,
    SciMMIR,
    StanfordCarsZeroShotClassification,
    STL10ZeroShotClassification,
    SUN397ZeroShotClassification,
    UCF101ZeroShotClassification,
)

__all__ = [
    "VOC2007Classification",
    "STS17MultilingualVisualSTS",
    "STSBenchmarkMultilingualVisualSTS",
    "STS13VisualSTS",
    "STS15VisualSTS",
    "STS12VisualSTS",
    "STS16VisualSTS",
    "STS14VisualSTS",
    "TinyImageNet",
    "CIFAR100Clustering",
    "CIFAR10Clustering",
    "ImageNet10Clustering",
    "ImageNetDog15Clustering",
    "OxfordPetsClassification",
    "StanfordCarsClassification",
    "SUN397Classification",
    "OxfordFlowersClassification",
    "UCF101Classification",
    "GTSRBClassification",
    "DTDClassification",
    "CIFAR100Classification",
    "CIFAR10Classification",
    "FER2013Classification",
    "Country211Classification",
    "EuroSATClassification",
    "Imagenet1kClassification",
    "STL10Classification",
    "Caltech101Classification",
    "PatchCamelyonClassification",
    "MNISTClassification",
    "Food101Classification",
    "BirdsnapClassification",
    "RESISC45Classification",
    "FGVCAircraftClassification",
    "CVBenchCount",
    "CVBenchDepth",
    "CVBenchDistance",
    "CVBenchRelation",
    "BLINKIT2IMultiChoice",
    "BLINKIT2TMultiChoice",
    "Winoground",
    "ImageCoDe",
    "AROFlickrOrder",
    "AROVisualRelation",
    "SugarCrepe",
    "AROVisualAttribution",
    "AROCocoOrder",
    "MemotionI2TRetrieval",
    "BLINKIT2TRetrieval",
    "InfoSeekIT2ITRetrieval",
    "ReMuQIT2TRetrieval",
    "VisualNewsT2IRetrieval",
    "Fashion200kI2TRetrieval",
    "CUB200I2I",
    "SciMMIRT2IRetrieval",
    "RP2kI2IRetrieval",
    "Flickr30kI2TRetrieval",
    "OVENIT2TRetrieval",
    "VizWizIT2TRetrieval",
    "BLINKIT2IRetrieval",
    "WebQAT2TRetrieval",
    "GLDv2I2IRetrieval",
    "MemotionT2IRetrieval",
    "SketchyI2IRetrieval",
    "Fashion200kT2IRetrieval",
    "ROxfordEasyI2IRetrieval",
    "ROxfordHardI2IRetrieval",
    "ROxfordMediumI2IRetrieval",
    "EncyclopediaVQAIT2ITRetrieval",
    "VidoreArxivQARetrieval",
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreShiftProjectRetrieval",
    "VidoreSyntheticDocQAAIRetrieval",
    "VidoreSyntheticDocQAEnergyRetrieval",
    "VidoreSyntheticDocQAGovernmentReportsRetrieval",
    "VidoreSyntheticDocQAHealthcareIndustryRetrieval",
    "VidoreTabfquadRetrieval",
    "VidoreTatdqaRetrieval",
    "CIRRIT2IRetrieval",
    "METI2IRetrieval",
    "StanfordCarsI2I",
    "MSCOCOT2IRetrieval",
    "VisualNewsI2TRetrieval",
    "FORBI2I",
    "Flickr30kT2IRetrieval",
    "WebQAT2ITRetrieval",
    "SOPI2IRetrieval",
    "NIGHTSI2IRetrieval",
    "EDIST2ITRetrieval",
    "LLaVAIT2TRetrieval",
    "OVENIT2ITRetrieval",
    "InfoSeekIT2TRetrieval",
    "HatefulMemesT2IRetrieval",
    "HatefulMemesI2TRetrieval",
    "TUBerlinT2IRetrieval",
    "RParisEasyI2IRetrieval",
    "RParisHardI2IRetrieval",
    "RParisMediumI2IRetrieval",
    "GLDv2I2TRetrieval",
    "MSCOCOI2TRetrieval",
    "ImageCoDeT2IRetrieval",
    "FashionIQIT2IRetrieval",
    "OKVQAIT2TRetrieval",
    "SciMMIRI2TRetrieval",
    "VQA2IT2TRetrieval",
    "XFlickr30kCoT2IRetrieval",
    "WITT2IRetrieval",
    "XM3600T2IRetrieval",
    "MNISTZeroShotClassification",
    "CLEVR",
    "CLEVRCount",
    "SciMMIR",
    "PatchCamelyonZeroShotClassification",
    "OxfordPetsZeroShotClassification",
    "EuroSATZeroShotClassification",
    "StanfordCarsZeroShotClassification",
    "CIFAR100ZeroShotClassification",
    "CIFAR10ZeroShotClassification",
    "Country211ZeroShotClassification",
    "Food101ZeroShotClassification",
    "SUN397ZeroShotClassification",
    "GTSRBZeroShotClassification",
    "Imagenet1kZeroShotClassification",
    "DTDZeroShotClassification",
    "RESISC45ZeroShotClassification",
    "STL10ZeroShotClassification",
    "Caltech101ZeroShotClassification",
    "BirdsnapZeroShotClassification",
    "RenderedSST2",
    "UCF101ZeroShotClassification",
    "FER2013ZeroShotClassification",
    "FGVCAircraftZeroShotClassification",
]
