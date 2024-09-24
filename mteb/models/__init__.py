from __future__ import annotations

import logging
from typing import Any

from sentence_transformers import SentenceTransformer

from mteb.encoder_interface import Encoder, EncoderWithQueryCorpusEncode
from mteb.model_meta import ModelMeta
from mteb.models import (bge_models, bm25, cohere_models, e5_instruct,
                         e5_models, google_models, gritlm_models, gte_models,
                         llm2vec_models, mxbai_models, nomic_models,
                         openai_models, ru_sentence_models, salesforce_models,
                         sentence_transformers_models, voyage_models)
from mteb.models.overview import *

logger = logging.getLogger(__name__)