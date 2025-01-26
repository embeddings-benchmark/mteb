from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SynPerChatbotConvSAAnger(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSAAnger",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Anger",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-anger",
            "revision": "5cae68b7fc094cb2fa6890a464e4d836e8107f5e",
        },
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotConvSASatisfaction(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSASatisfaction",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Satisfaction",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-satisfaction",
            "revision": "50fd9d5d09edd53af89af765636be5db6f983f0e",
        },
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotConvSAFriendship(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSAFriendship",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Friendship",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-friendship",
            "revision": "9dae119101e9b4e9bb40d5b9d29ffd7a621f9942",
        },
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotConvSAFear(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSAFear",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Fear",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-fear",
            "revision": "3c22f7e6bf4e366c86d69293c9164bf9e9d80aac",
        },
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotConvSAJealousy(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSAJealousy",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Jealousy",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-jealousy",
            "revision": "0d5104ecaa109d2448afe1f40dbf860f5e4458a8",
        },
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotConvSASurprise(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSASurprise",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Surprise",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-surprise",
            "revision": "62dad66fc2837b0ac5e5175fe7c265d2d502a386",
        },
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotConvSALove(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSALove",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Love",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-love",
            "revision": "0e000b2f73e9bb74ec8fc6da10011c52725b8469",
        },
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotConvSASadness(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSASadness",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Sadness",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-sadness",
            "revision": "e9c678325565a5e4dadc43fd6693a8ccff1dd6b2",
        },
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotConvSAHappiness(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSAHappiness",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Happiness",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-happiness",
            "revision": "e60893b7a8d01c9b8c12fadfe8f0a06e9d548a63",
        },
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotConvSAToneChatbotClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSAToneChatbotClassification",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Tone Chatbot Classification",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-tone-chatbot-classification",
            "revision": "1f403cfadb85004fbf7e2480334fffc4c999b4ab",
        },
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotConvSAToneUserClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSAToneUserClassification",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Tone User",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/chatbot-conversational-sentiment-analysis-tone-user-classification",
            "revision": "dd0f76661bef69819cc38c8a455b10af86ac3571",
        },
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotSatisfactionLevelClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotSatisfactionLevelClassification",
        description="Synthetic Persian Chatbot Satisfaction Level Classification",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-satisfaction-level-classification",
            "revision": "e72db473602d750f1bcdc9f0436e1e3b967e088f",
        },
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotRAGToneChatbotClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotRAGToneChatbotClassification",
        description="Synthetic Persian Chatbot RAG Tone Chatbot Classification",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-rag-tone-chatbot-classification",
            "revision": "76f15a203fc13bd98a8f0fdddab1b68c28d7d674",
        },
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotRAGToneUserClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotRAGToneUserClassification",
        description="Synthetic Persian Chatbot RAG Tone User Classification",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-rag-tone-user-classification",
            "revision": "f1f6ad83bb135dc94fbf1ca05c3ba164f5619369",
        },
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotToneChatbotClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotToneChatbotClassification",
        description="Synthetic Persian Chatbot Tone Chatbot Classification",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-tone-chatbot-classification",
            "revision": "a5a739a036fa7bb8ae0be91bc081fdd260d4bdab",
        },
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotToneUserClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotToneUserClassification",
        description="Synthetic Persian Chatbot Tone User Classification",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-tone-user-classification",
            "revision": "780d629437f7be127c6b287a61776372f9f333b9",
        },
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class PersianTextTone(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PersianTextTone",
        description="Persian Text Tone",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/persian-text-tone",
            "revision": "7144f4c6bdd77911df0dfc5a8bd44dba17e27e3a",
        },
        type="Classification",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SIDClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SIDClassification",
        description="SID Classification",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/sid-classification",
            "revision": "29bed651bb980395f5aa473607154d93226945e1",
        },
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Academic"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class DeepSentiPers(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DeepSentiPers",
        description="Persian Sentiment Analysis Dataset",
        reference="https://github.com/JoyeBright/DeepSentiPers",
        dataset={
            "path": "PartAI/DeepSentiPers",
            "revision": "ee4f09f404051761cfe14d68127532c82de41cb3",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )
    samples_per_label = 32

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("review", "text")


class PersianTextEmotion(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PersianTextEmotion",
        description="Emotion is a Persian dataset with six basic emotions: anger, fear, joy, love, sadness, and surprise.",
        reference="https://huggingface.co/datasets/SeyedAli/Persian-Text-Emotion",
        dataset={
            "path": "SeyedAli/Persian-Text-Emotion",
            "revision": "518fcd2c8b89917c7696770672688217a2eabf88",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SentimentDKSF(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SentimentDKSF",
        description="The Sentiment DKSF (Digikala/Snappfood comments) is a dataset for sentiment analysis.",
        reference="https://github.com/hezarai/hezar",
        dataset={
            "path": "hezarai/sentiment-dksf",
            "revision": "b4d5a8dd501db610b5ad89e9aa13f863b842b395",
        },
        type="Classification",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class NLPTwitterAnalysisClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NLPTwitterAnalysisClassification",
        description="Twitter Analysis Classification",
        reference="https://huggingface.co/datasets/hamedhf/nlp_twitter_analysis/tree/main",
        dataset={
            "path": "hamedhf/nlp_twitter_analysis",
            "revision": "4ceb1312583fd2c7c73ad2d550b726124dcd39a0",
        },
        type="Classification",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )
    samples_per_label = 32

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("tweet", "text")


class DigikalamagClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DigikalamagClassification",
        description="A total of 8,515 articles scraped from Digikala Online Magazine. This dataset includes seven different classes.",
        reference="https://hooshvare.github.io/docs/datasets/tc",
        dataset={
            "path": "PNLPhub/DigiMag",
            "revision": "969b335c9f50eda5c384460be4eb2b55505c2c53",
            "trust_remote_code": True,
        },
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )
    samples_per_label = 32

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("content", "text")
