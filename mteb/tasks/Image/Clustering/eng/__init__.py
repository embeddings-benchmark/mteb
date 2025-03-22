from __future__ import annotations

from .CIFAR import CIFAR10Clustering, CIFAR100Clustering
from .ImageNet import ImageNet10Clustering, ImageNetDog15Clustering
from .TinyImageNet import TinyImageNet

__all__ = [
    "TinyImageNet",
    "CIFAR100Clustering",
    "CIFAR10Clustering",
    "ImageNet10Clustering",
    "ImageNetDog15Clustering",
]
