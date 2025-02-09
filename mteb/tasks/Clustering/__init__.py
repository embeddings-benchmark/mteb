from __future__ import annotations

from .deu.BlurbsClusteringP2P import *
from .deu.BlurbsClusteringS2S import *
from .deu.TenKGnadClusteringP2P import *
from .deu.TenKGnadClusteringS2S import *
from .eng.ArxivClusteringP2P import *
from .eng.ArxivClusteringS2S import *
from .eng.ArXivHierarchicalClustering import *
from .eng.BigPatentClustering import *
from .eng.BiorxivClusteringP2P import *
from .eng.BiorxivClusteringS2S import *
from .eng.ComplaintsClustering import *
from .eng.FinanceArxivP2PClustering import *
from .eng.FinanceArxivS2SClustering import *
from .eng.MedrxivClusteringP2P import *
from .eng.MedrxivClusteringS2S import *
from .eng.MInDS14EnClustering import *
from .eng.PiiClustering import *
from .eng.RedditClustering import *
from .eng.RedditClusteringP2P import *
from .eng.StackExchangeClustering import *
from .eng.StackExchangeClusteringP2P import *
from .eng.TwentyNewsgroupsClustering import *
from .eng.WikiCitiesClustering import *
from .eng.WikiCompany2IndustryClustering import *
from .fra.AlloProfClusteringP2P import *
from .fra.AlloProfClusteringS2S import *
from .fra.HALClusteringS2S import *
from .jpn.LivedoorNewsClustering import *
from .jpn.MewsC16JaClustering import *
from .multilingual.IndicReviewsClusteringP2P import *
from .multilingual.MasakhaNEWSClusteringP2P import *
from .multilingual.MasakhaNEWSClusteringS2S import *
from .multilingual.MLSUMClusteringP2P import *
from .multilingual.MLSUMClusteringS2S import *
from .multilingual.SIB200ClusteringS2S import *
from .multilingual.WikiClusteringP2P import *
from .nob.snl_clustering import *
from .nob.SNLHierarchicalClustering import *
from .nob.vg_clustering import *
from .nob.VGHierarchicalClustering import *
from .pol.PolishClustering import *
from .rom.RomaniBibleClustering import *
from .rus.GeoreviewClusteringP2P import *
from .rus.RuSciBenchGRNTIClusteringP2P import *
from .rus.RuSciBenchOECDClusteringP2P import *
from .spa.SpanishNewsClusteringP2P import *
from .swe.swedn_clustering import *
from .swe.SwednClustering import *
from .zho.CCKS2019Clustering import *
from .zho.CCKS2020Clustering import *
from .zho.CCKS2022Clustering import *
from .zho.CMTEBClustering import *
from .zho.FinNLClustering import *
from .zho.MInDS14ZhClustering import *
=======
from .deu import (
    BlurbsClusteringP2P,
    BlurbsClusteringP2PFast,
    BlurbsClusteringS2S,
    BlurbsClusteringS2SFast,
    TenKGnadClusteringP2P,
    TenKGnadClusteringP2PFast,
    TenKGnadClusteringS2S,
    TenKGnadClusteringS2SFast,
)
from .eng import (
    ArxivClusteringP2P,
    ArxivClusteringP2PFast,
    ArxivClusteringS2S,
    ArXivHierarchicalClusteringP2P,
    ArXivHierarchicalClusteringS2S,
    BigPatentClustering,
    BigPatentClusteringFast,
    BiorxivClusteringP2P,
    BiorxivClusteringP2PFast,
    BiorxivClusteringS2S,
    BiorxivClusteringS2SFast,
    MedrxivClusteringP2P,
    MedrxivClusteringP2PFast,
    MedrxivClusteringS2S,
    MedrxivClusteringS2SFast,
    RedditClustering,
    RedditClusteringP2P,
    RedditFastClusteringP2P,
    RedditFastClusteringS2S,
    StackExchangeClustering,
    StackExchangeClusteringFast,
    StackExchangeClusteringP2P,
    StackExchangeClusteringP2PFast,
    TwentyNewsgroupsClustering,
    TwentyNewsgroupsClusteringFast,
    WikiCitiesClustering,
    WikipediaChemistrySpecialtiesClustering,
    WikipediaChemistryTopicsClustering,
)
from .fas import (
    BeytooteClustering,
    DigikalamagClustering,
    HamshahriClustring,
    NLPTwitterAnalysisClustering,
    SIDClustring,
)
from .fra import (
    AlloProfClusteringP2P,
    AlloProfClusteringP2PFast,
    AlloProfClusteringS2S,
    AlloProfClusteringS2SFast,
    HALClusteringS2S,
    HALClusteringS2SFast,
)
from .jpn import LivedoorNewsClustering, LivedoorNewsClusteringv2, MewsC16JaClustering
from .multilingual import (
    IndicReviewsClusteringP2P,
    MasakhaNEWSClusteringP2P,
    MasakhaNEWSClusteringS2S,
    MLSUMClusteringP2P,
    MLSUMClusteringP2PFast,
    MLSUMClusteringS2S,
    MLSUMClusteringS2SFast,
    SIB200ClusteringFast,
    WikiClusteringFastP2P,
    WikiClusteringP2P,
)
from .nob import (
    SNLClustering,
    SNLHierarchicalClusteringP2P,
    SNLHierarchicalClusteringS2S,
    VGClustering,
    VGHierarchicalClusteringP2P,
    VGHierarchicalClusteringS2S,
)
from .pol import (
    EightTagsClustering,
    EightTagsClusteringFast,
    PlscClusteringP2P,
    PlscClusteringP2PFast,
    PlscClusteringS2S,
    PlscClusteringS2SFast,
)
from .rom import RomaniBibleClustering
from .rus import (
    GeoreviewClusteringP2P,
    RuSciBenchGRNTIClusteringP2P,
    RuSciBenchOECDClusteringP2P,
)
from .spa import SpanishNewsClusteringP2P
from .swe import SwednClustering, SwednClusteringFastS2S, SwednClusteringP2P
from .zho import (
    CLSClusteringFastP2P,
    CLSClusteringFastS2S,
    CLSClusteringP2P,
    CLSClusteringS2S,
    ThuNewsClusteringFastP2P,
    ThuNewsClusteringFastS2S,
    ThuNewsClusteringP2P,
    ThuNewsClusteringS2S,
)

__all__ = [
    "CLSClusteringFastP2P",
    "CLSClusteringFastS2S",
    "CLSClusteringP2P",
    "CLSClusteringS2S",
    "ThuNewsClusteringFastP2P",
    "ThuNewsClusteringFastS2S",
    "ThuNewsClusteringP2P",
    "ThuNewsClusteringS2S",
    "SpanishNewsClusteringP2P",
    "EightTagsClustering",
    "EightTagsClusteringFast",
    "PlscClusteringP2P",
    "PlscClusteringP2PFast",
    "PlscClusteringS2S",
    "PlscClusteringS2SFast",
    "SwednClustering",
    "SwednClusteringFastS2S",
    "SwednClusteringP2P",
    "VGClustering",
    "SNLHierarchicalClusteringP2P",
    "SNLHierarchicalClusteringS2S",
    "SNLClustering",
    "VGHierarchicalClusteringP2P",
    "VGHierarchicalClusteringS2S",
    "BlurbsClusteringS2S",
    "BlurbsClusteringS2SFast",
    "TenKGnadClusteringP2P",
    "TenKGnadClusteringP2PFast",
    "TenKGnadClusteringS2S",
    "TenKGnadClusteringS2SFast",
    "BlurbsClusteringP2P",
    "BlurbsClusteringP2PFast",
    "RomaniBibleClustering",
    "MedrxivClusteringS2S",
    "MedrxivClusteringS2SFast",
    "BiorxivClusteringS2S",
    "BiorxivClusteringS2SFast",
    "StackExchangeClustering",
    "StackExchangeClusteringFast",
    "RedditClustering",
    "RedditFastClusteringS2S",
    "ArxivClusteringS2S",
    "ArxivClusteringP2P",
    "ArxivClusteringP2PFast",
    "MedrxivClusteringP2P",
    "MedrxivClusteringP2PFast",
    "WikipediaChemistryTopicsClustering",
    "WikiCitiesClustering",
    "BiorxivClusteringP2P",
    "BiorxivClusteringP2PFast",
    "TwentyNewsgroupsClustering",
    "TwentyNewsgroupsClusteringFast",
    "ArXivHierarchicalClusteringP2P",
    "ArXivHierarchicalClusteringS2S",
    "WikipediaChemistrySpecialtiesClustering",
    "BigPatentClustering",
    "BigPatentClusteringFast",
    "StackExchangeClusteringP2P",
    "StackExchangeClusteringP2PFast",
    "RedditClusteringP2P",
    "RedditFastClusteringP2P",
    "BeytooteClustering",
    "DigikalamagClustering",
    "HamshahriClustring",
    "NLPTwitterAnalysisClustering",
    "SIDClustring",
    "LivedoorNewsClustering",
    "LivedoorNewsClusteringv2",
    "MewsC16JaClustering",
    "WikiClusteringFastP2P",
    "WikiClusteringP2P",
    "MLSUMClusteringS2S",
    "MLSUMClusteringS2SFast",
    "MasakhaNEWSClusteringS2S",
    "MLSUMClusteringP2P",
    "MLSUMClusteringP2PFast",
    "IndicReviewsClusteringP2P",
    "SIB200ClusteringFast",
    "MasakhaNEWSClusteringP2P",
    "GeoreviewClusteringP2P",
    "RuSciBenchGRNTIClusteringP2P",
    "RuSciBenchOECDClusteringP2P",
    "HALClusteringS2S",
    "HALClusteringS2SFast",
    "AlloProfClusteringS2S",
    "AlloProfClusteringS2SFast",
    "AlloProfClusteringP2P",
    "AlloProfClusteringP2PFast",
]
