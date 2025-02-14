from __future__ import annotations

from .Birdsnap import BirdsnapClassification
from .Caltech101 import Caltech101Classification
from .CIFAR import CIFAR10ZeroShotClassification, CIFAR100ZeroShotClassification
from .CLEVR import CLEVR, CLEVRCount
from .Country211 import Country211Classification
from .DTD import DTDClassification
from .EuroSAT import EuroSATClassification
from .FER2013 import FER2013Classification
from .FGVCAircraft import FGVCAircraftClassification
from .Food101 import Food101Classification
from .GTSRB import GTSRBClassification
from .Imagenet1k import Imagenet1kClassification
from .MNIST import MNISTClassification
from .OxfordPets import OxfordPetsClassification
from .PatchCamelyon import PatchCamelyonClassification
from .RenderedSST2 import RenderedSST2
from .RESISC45 import RESISC45Classification
from .SciMMIR import SciMMIR
from .StanfordCars import StanfordCarsClassification
from .STL10 import STL10Classification
from .SUN397 import SUN397Classification
from .UCF101 import UCF101Classification

__all__ = [
    "MNISTClassification",
    "CLEVR",
    "CLEVRCount",
    "SciMMIR",
    "PatchCamelyonClassification",
    "OxfordPetsClassification",
    "EuroSATClassification",
    "StanfordCarsClassification",
    "CIFAR100ZeroShotClassification",
    "CIFAR10ZeroShotClassification",
    "Country211Classification",
    "Food101Classification",
    "SUN397Classification",
    "GTSRBClassification",
    "Imagenet1kClassification",
    "DTDClassification",
    "RESISC45Classification",
    "STL10Classification",
    "Caltech101Classification",
    "BirdsnapClassification",
    "RenderedSST2",
    "UCF101Classification",
    "FER2013Classification",
    "FGVCAircraftClassification",
]
