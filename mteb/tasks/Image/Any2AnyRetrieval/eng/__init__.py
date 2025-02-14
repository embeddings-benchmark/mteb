from __future__ import annotations

from .BLINKIT2IRetrieval import BLINKIT2IRetrieval
from .BLINKIT2TRetrieval import BLINKIT2TRetrieval
from .CIRRIT2IRetrieval import CIRRIT2IRetrieval
from .CUB200I2IRetrieval import CUB200I2I
from .EDIST2ITRetrieval import EDIST2ITRetrieval
from .EncyclopediaVQAIT2ITRetrieval import EncyclopediaVQAIT2ITRetrieval
from .Fashion200kI2TRetrieval import Fashion200kI2TRetrieval
from .Fashion200kT2IRetrieval import Fashion200kT2IRetrieval
from .FashionIQIT2IRetrieval import FashionIQIT2IRetrieval
from .Flickr30kI2TRetrieval import Flickr30kI2TRetrieval
from .Flickr30kT2IRetrieval import Flickr30kT2IRetrieval
from .FORBI2IRetrieval import FORBI2I
from .GLDv2I2IRetrieval import GLDv2I2IRetrieval
from .GLDv2I2TRetrieval import GLDv2I2TRetrieval
from .HatefulMemesI2TRetrieval import HatefulMemesI2TRetrieval
from .HatefulMemesT2IRetrieval import HatefulMemesT2IRetrieval
from .ImageCoDeT2IRetrieval import ImageCoDeT2IRetrieval
from .InfoSeekIT2ITRetrieval import InfoSeekIT2ITRetrieval
from .InfoSeekIT2TRetrieval import InfoSeekIT2TRetrieval
from .LLaVAIT2TRetrieval import LLaVAIT2TRetrieval
from .MemotionI2TRetrieval import MemotionI2TRetrieval
from .MemotionT2IRetrieval import MemotionT2IRetrieval
from .METI2IRetrieval import METI2IRetrieval
from .MSCOCOI2TRetrieval import MSCOCOI2TRetrieval
from .MSCOCOT2IRetrieval import MSCOCOT2IRetrieval
from .NIGHTSI2IRetrieval import NIGHTSI2IRetrieval
from .OKVQAIT2TRetrieval import OKVQAIT2TRetrieval
from .OVENIT2ITRetrieval import OVENIT2ITRetrieval
from .OVENIT2TRetrieval import OVENIT2TRetrieval
from .ReMuQIT2TRetrieval import ReMuQIT2TRetrieval
from .ROxfordI2IRetrieval import (
    ROxfordEasyI2IRetrieval,
    ROxfordHardI2IRetrieval,
    ROxfordMediumI2IRetrieval,
)
from .RP2kI2IRetrieval import RP2kI2IRetrieval
from .RParisI2IRetrieval import (
    RParisEasyI2IRetrieval,
    RParisHardI2IRetrieval,
    RParisMediumI2IRetrieval,
)
from .SciMMIRI2TRetrieval import SciMMIRI2TRetrieval
from .SciMMIRT2IRetrieval import SciMMIRT2IRetrieval
from .SketchyI2IRetrieval import SketchyI2IRetrieval
from .SOPI2IRetrieval import SOPI2IRetrieval
from .StanfordCarsI2IRetrieval import StanfordCarsI2I
from .TUBerlinT2IRetrieval import TUBerlinT2IRetrieval
from .VidoreBenchRetrieval import (
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
)
from .VisualNewsI2TRetrieval import VisualNewsI2TRetrieval
from .VisualNewsT2IRetrieval import VisualNewsT2IRetrieval
from .VizWizIT2TRetrieval import VizWizIT2TRetrieval
from .VQA2IT2TRetrieval import VQA2IT2TRetrieval
from .WebQAT2ITRetrieval import WebQAT2ITRetrieval
from .WebQAT2TRetrieval import WebQAT2TRetrieval

__all__ = [
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
]
