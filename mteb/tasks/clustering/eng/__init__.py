from .arxiv_clustering_p2p import ArxivClusteringP2P, ArxivClusteringP2PFast
from .arxiv_clustering_s2s import ArxivClusteringS2S
from .arxiv_hierarchical_clustering import (
    ArXivHierarchicalClusteringP2P,
    ArXivHierarchicalClusteringS2S,
)
from .big_patent_clustering import BigPatentClustering, BigPatentClusteringFast
from .biorxiv_clustering_p2p import BiorxivClusteringP2P, BiorxivClusteringP2PFast
from .biorxiv_clustering_s2s import BiorxivClusteringS2S, BiorxivClusteringS2SFast
from .built_bench_clustering_p2p import BuiltBenchClusteringP2P
from .built_bench_clustering_s2s import BuiltBenchClusteringS2S
from .cifar import CIFAR10Clustering, CIFAR100Clustering
from .clus_trec_covid import ClusTrecCovid
from .hume_arxiv_clustering_p2p import HUMEArxivClusteringP2P
from .hume_reddit_clustering_p2p import HUMERedditClusteringP2P
from .hume_wiki_cities_clustering import HUMEWikiCitiesClustering
from .image_net import ImageNet10Clustering, ImageNetDog15Clustering
from .medrxiv_clustering_p2p import MedrxivClusteringP2P, MedrxivClusteringP2PFast
from .medrxiv_clustering_s2s import MedrxivClusteringS2S, MedrxivClusteringS2SFast
from .reddit_clustering import RedditClustering, RedditFastClusteringS2S
from .reddit_clustering_p2p import RedditClusteringP2P, RedditFastClusteringP2P
from .stack_exchange_clustering import (
    StackExchangeClustering,
    StackExchangeClusteringFast,
)
from .stack_exchange_clustering_p2p import (
    StackExchangeClusteringP2P,
    StackExchangeClusteringP2PFast,
)
from .tiny_image_net import TinyImageNet
from .twenty_newsgroups_clustering import (
    TwentyNewsgroupsClustering,
    TwentyNewsgroupsClusteringFast,
)
from .wiki_cities_clustering import WikiCitiesClustering
from .wikipedia_chemistry_specialties_clustering import (
    WikipediaChemistrySpecialtiesClustering,
)
from .wikipedia_chemistry_topics_clustering import WikipediaChemistryTopicsClustering

__all__ = [
    "ArXivHierarchicalClusteringP2P",
    "ArXivHierarchicalClusteringS2S",
    "ArxivClusteringP2P",
    "ArxivClusteringP2PFast",
    "ArxivClusteringS2S",
    "BigPatentClustering",
    "BigPatentClusteringFast",
    "BiorxivClusteringP2P",
    "BiorxivClusteringP2PFast",
    "BiorxivClusteringS2S",
    "BiorxivClusteringS2SFast",
    "BuiltBenchClusteringP2P",
    "BuiltBenchClusteringS2S",
    "CIFAR10Clustering",
    "CIFAR100Clustering",
    "ClusTrecCovid",
    "HUMEArxivClusteringP2P",
    "HUMERedditClusteringP2P",
    "HUMEWikiCitiesClustering",
    "ImageNet10Clustering",
    "ImageNetDog15Clustering",
    "MedrxivClusteringP2P",
    "MedrxivClusteringP2PFast",
    "MedrxivClusteringS2S",
    "MedrxivClusteringS2SFast",
    "RedditClustering",
    "RedditClusteringP2P",
    "RedditFastClusteringP2P",
    "RedditFastClusteringS2S",
    "StackExchangeClustering",
    "StackExchangeClusteringFast",
    "StackExchangeClusteringP2P",
    "StackExchangeClusteringP2PFast",
    "TinyImageNet",
    "TwentyNewsgroupsClustering",
    "TwentyNewsgroupsClusteringFast",
    "WikiCitiesClustering",
    "WikipediaChemistrySpecialtiesClustering",
    "WikipediaChemistryTopicsClustering",
]
