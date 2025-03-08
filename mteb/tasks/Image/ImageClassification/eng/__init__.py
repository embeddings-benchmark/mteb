from __future__ import annotations

from .BirdsnapClassification import BirdsnapClassification
from .Caltech101Classification import Caltech101Classification
from .CIFAR import CIFAR10Classification, CIFAR100Classification
from .Country211Classification import Country211Classification
from .DTDClassification import DTDClassification
from .EuroSATClassification import EuroSATClassification
from .FER2013Classification import FER2013Classification
from .FGVCAircraftClassification import FGVCAircraftClassification
from .Food101Classification import Food101Classification
from .GTSRBClassification import GTSRBClassification
from .Imagenet1k import Imagenet1kClassification
from .MNISTClassification import MNISTClassification
from .OxfordFlowersClassification import OxfordFlowersClassification
from .OxfordPetsClassification import OxfordPetsClassification
from .PatchCamelyonClassification import PatchCamelyonClassification
from .RESISC45Classification import RESISC45Classification
from .StanfordCarsClassification import StanfordCarsClassification
from .STL10Classification import STL10Classification
from .SUN397Classification import SUN397Classification
from .UCF101Classification import UCF101Classification

__all__ = [
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
]
