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
    "BirdsnapClassification",
    "CIFAR100ZeroShotClassification",
    "CIFAR10ZeroShotClassification",
    "CLEVR",
    "CLEVRCount",
    "Caltech101Classification",
    "Country211Classification",
    "DTDClassification",
    "EuroSATClassification",
    "FER2013Classification",
    "FGVCAircraftClassification",
    "Food101Classification",
    "GTSRBClassification",
    "Imagenet1kClassification",
    "MNISTClassification",
    "OxfordPetsClassification",
    "PatchCamelyonClassification",
    "RESISC45Classification",
    "RenderedSST2",
    "STL10Classification",
    "SUN397Classification",
    "SciMMIR",
    "StanfordCarsClassification",
    "UCF101Classification",
]
