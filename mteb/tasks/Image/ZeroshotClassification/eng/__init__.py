from __future__ import annotations

from .Birdsnap import BirdsnapZeroshotClassification
from .Caltech101 import Caltech101ZeroshotClassification
from .CIFAR import CIFAR10ZeroShotClassification, CIFAR100ZeroShotClassification
from .CLEVR import CLEVR, CLEVRCount
from .Country211 import Country211ZeroshotClassification
from .DTD import DTDZeroshotClassification
from .EuroSAT import EuroSATZeroshotClassification
from .FER2013 import FER2013ZeroshotClassification
from .FGVCAircraft import FGVCAircraftZeroShotClassification
from .Food101 import Food101ZeroShotClassification
from .GTSRB import GTSRBZeroshotClassification
from .Imagenet1k import Imagenet1kZeroshotClassification
from .MNIST import MNISTZeroshotClassification
from .OxfordPets import OxfordPetsZeroshotClassification
from .PatchCamelyon import PatchCamelyonZeroshotClassification
from .RenderedSST2 import RenderedSST2
from .RESISC45 import RESISC45ZeroshotClassification
from .SciMMIR import SciMMIR
from .StanfordCars import StanfordCarsZeroshotClassification
from .STL10 import STL10ZeroshotClassification
from .SUN397 import SUN397ZeroshotClassification
from .UCF101 import UCF101ZeroshotClassification

__all__ = [
    "MNISTZeroshotClassification",
    "CLEVR",
    "CLEVRCount",
    "SciMMIR",
    "PatchCamelyonZeroshotClassification",
    "OxfordPetsZeroshotClassification",
    "EuroSATZeroshotClassification",
    "StanfordCarsZeroshotClassification",
    "CIFAR100ZeroShotClassification",
    "CIFAR10ZeroShotClassification",
    "Country211ZeroshotClassification",
    "Food101ZeroShotClassification",
    "SUN397ZeroshotClassification",
    "GTSRBZeroshotClassification",
    "Imagenet1kZeroshotClassification",
    "DTDZeroshotClassification",
    "RESISC45ZeroshotClassification",
    "STL10ZeroshotClassification",
    "Caltech101ZeroshotClassification",
    "BirdsnapZeroshotClassification",
    "RenderedSST2",
    "UCF101ZeroshotClassification",
    "FER2013ZeroshotClassification",
    "FGVCAircraftZeroShotClassification",
]
