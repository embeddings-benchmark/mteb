from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_REFERENCE = "https://arxiv.org/abs/2503.07920"
_BIBTEX = r"""
@inproceedings{cahyawijaya-etal-2025-crowdsource,
    title = "Crowdsource, Crawl, or Generate? Creating {SEA}-{VL}, a Multicultural Vision-Language Dataset for {S}outheast {A}sia",
    author = {Cahyawijaya, Samuel  and
      Lovenia, Holy  and
      Moniz, Joel Ruben Antony  and
      Wong, Tack Hwa  and
      Farhansyah, Mohammad Rifqi  and
      Maung, Thant Thiri  and
      Hudi, Frederikus  and
      Anugraha, David  and
      Habibi, Muhammad Ravi Shulthan  and
      Qorib, Muhammad Reza  and
      Agarwal, Amit  and
      Imperial, Joseph Marvin  and
      Patel, Hitesh Laxmichand  and
      Feliren, Vicky  and
      Nasution, Bahrul Ilmi  and
      Rufino, Manuel Antonio  and
      Winata, Genta Indra  and
      Rajagede, Rian Adam  and
      Catalan, Carlos Rafael  and
      Imam, Mohamed Fazli Mohamed  and
      Pattnayak, Priyaranjan  and
      Pranida, Salsabila Zahirah  and
      Pratama, Kevin  and
      Bangera, Yeshil  and
      Na-Thalang, Adisai  and
      Monderin, Patricia Nicole  and
      Song, Yueqi  and
      Simon, Christian  and
      Ng, Lynnette Hui Xian  and
      Sapan, Richardy Lobo  and
      Rafi, Taki Hasan  and
      Wang, Bin  and
      Supryadi  and
      Veerakanjana, Kanyakorn  and
      Ittichaiwong, Piyalitt  and
      Roque, Matthew Theodore  and
      Vincentio, Karissa  and
      Kreangphet, Takdanai  and
      Artkaew, Phakphum  and
      Palgunadi, Kadek Hendrawan  and
      Yu, Yanzhi  and
      Hastuti, Rochana Prih  and
      Nixon, William  and
      Bangera, Mithil  and
      Lim, Adrian Xuan Wei  and
      Khine, Aye Hninn  and
      Zhafran, Hanif Muhammad  and
      Ferdinan, Teddy  and
      Izzani, Audra Aurora  and
      Singh, Ayushman  and
      Evan, Evan  and
      Krito, Jauza Akbar  and
      Anugraha, Michael  and
      Ilasariya, Fenal Ashokbhai  and
      Li, Haochen  and
      Daniswara, John Amadeo  and
      Tjiaranata, Filbert Aurelian  and
      Yulianrifat, Eryawan Presma  and
      Udomcharoenchaikit, Can  and
      Ansori, Fadil Risdian  and
      Ihsani, Mahardika Krisna  and
      Nguyen, Giang  and
      Barik, Anab Maulana  and
      Velasco, Dan John  and
      Genadi, Rifo Ahmad  and
      Saha, Saptarshi  and
      Wei, Chengwei  and
      Flores, Isaiah Edri W.  and
      Han, Kenneth Chen Ko  and
      Santos, Anjela Gail D.  and
      Lim, Wan Shen  and
      Phyo, Kaung Si  and
      Santos, Tim  and
      Dwiastuti, Meisyarah  and
      Luo, Jiayun  and
      Cruz, Jan Christian Blaise  and
      Hee, Ming Shan  and
      Hanif, Ikhlasul Akmal  and
      Hakim, M.Alif Al  and
      Sya{'}ban, Muhammad Rizky  and
      Kerdthaisong, Kun  and
      Miranda, Lester James Validad  and
      Koto, Fajri  and
      Fatyanosa, Tirana Noor  and
      Aji, Alham Fikri  and
      Rosal, Jostin Jerico  and
      Kevin, Jun  and
      Wijaya, Robert  and
      Kampman, Onno P.  and
      Zhang, Ruochen  and
      Karlsson, B{\"o}rje F.  and
      Limkonchotiwat, Peerat},
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.916/",
    doi = "10.18653/v1/2025.acl-long.916",
    pages = "18685--18717",
    ISBN = "979-8-89176-251-0",
}
"""
_DESCRIPTION = (
    "SEA-VL crawling is a large-scale, Southeast Asia–focused image–caption dataset, "
    "containing culturally relevant image–text pairs from the web. The subset used in MTEB "
    "features 2048 unique images and their corresponding captions, offering an evaluation "
    "benchmark for image–text and text–image retrieval in Southeast Asian cultural contexts."
)


class SeaVLCrawlingT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SeaVLCrawlingT2IRetrieval",
        description=_DESCRIPTION
        + " Queries are captions; the corpus contains images (text→image retrieval).",
        reference=_REFERENCE,
        dataset={
            "path": "mteb/SEA-VL-Crawling-T2I",
            "revision": "761fb5ed934f053c8b94d321b7885cd5a3ad115f",
        },
        type="Any2AnyRetrieval",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2025-03-10"),
        domains=["Web", "Written"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find an image that matches the given caption."},
        is_beta=True,
    )


class SeaVLCrawlingI2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SeaVLCrawlingI2TRetrieval",
        description=_DESCRIPTION
        + " Queries are images; the corpus contains captions (image→text retrieval).",
        reference=_REFERENCE,
        dataset={
            "path": "mteb/SEA-VL-Crawling-I2T",
            "revision": "2239cc6bc852b299bd5ecb3898d81d2ef29a17c0",
        },
        type="Any2AnyRetrieval",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2025-03-10"),
        domains=["Web", "Written"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find a caption that matches the given image."},
        is_beta=True,
    )
