from __future__ import annotations

from .Birdsnap import BirdsnapZeroShotClassification
from .Caltech101 import Caltech101ZeroShotClassification
from .CIFAR import CIFAR10ZeroShotClassification, CIFAR100ZeroShotClassification
from .CLEVR import CLEVR, CLEVRCount
from .Country211 import Country211ZeroShotClassification
from .DTD import DTDZeroShotClassification
from .EuroSAT import EuroSATZeroShotClassification
from .FER2013 import FER2013ZeroShotClassification
from .FGVCAircraft import FGVCAircraftZeroShotClassification
from .Food101 import Food101ZeroShotClassification
from .GTSRB import GTSRBZeroShotClassification
from .Imagenet1k import Imagenet1kZeroShotClassification
from .MNIST import MNISTZeroShotClassification
from .OxfordPets import OxfordPetsZeroShotClassification
from .PatchCamelyon import PatchCamelyonZeroShotClassification
from .RenderedSST2 import RenderedSST2
from .RESISC45 import RESISC45ZeroShotClassification
from .SciMMIR import SciMMIR
from .StanfordCars import StanfordCarsZeroShotClassification
from .STL10 import STL10ZeroShotClassification
from .SUN397 import SUN397ZeroShotClassification
from .UCF101 import UCF101ZeroShotClassification

__all__ = [
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
