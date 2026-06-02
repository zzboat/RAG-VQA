from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    """Runtime settings for the RAG-VQA pipeline."""

    text_embedding_model: str = os.getenv(
        "RAG_VQA_TEXT_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    image_embedding_model: str = os.getenv("RAG_VQA_IMAGE_EMBEDDING_MODEL", "openai/clip-vit-base-patch32")
    caption_model: str = os.getenv("RAG_VQA_CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
    vqa_model: str = os.getenv("RAG_VQA_VQA_MODEL", "Salesforce/blip-vqa-base")
    generator_model: str = os.getenv("RAG_VQA_GENERATOR_MODEL", "google/flan-t5-base")
    device: str = os.getenv("RAG_VQA_DEVICE", "auto")
    # Default False: first run can pull BLIP/CLIP from Hugging Face. Air-gapped/offline: set RAG_VQA_VISION_LOCAL_FILES_ONLY=1.
    vision_local_files_only: bool = os.getenv("RAG_VQA_VISION_LOCAL_FILES_ONLY", "0") == "1"
    top_k: int = int(os.getenv("RAG_VQA_TOP_K", "5"))
    text_weight: float = float(os.getenv("RAG_VQA_TEXT_WEIGHT", "0.70"))
    image_weight: float = float(os.getenv("RAG_VQA_IMAGE_WEIGHT", "0.30"))
    min_evidence_score: float = float(os.getenv("RAG_VQA_MIN_SCORE", "0.10"))
    web_timeout: int = int(os.getenv("RAG_VQA_WEB_TIMEOUT", "8"))
    web_use_env_proxy: bool = os.getenv("RAG_VQA_WEB_USE_ENV_PROXY", "1") == "1"
    enable_generator: bool = os.getenv("RAG_VQA_ENABLE_GENERATOR", "1") != "0"
    enable_blip_vqa: bool = os.getenv("RAG_VQA_ENABLE_BLIP_VQA", "1") != "0"
    debug: bool = os.getenv("RAG_VQA_DEBUG", "0") == "1"
    cache_caption: bool = os.getenv("RAG_VQA_CACHE_CAPTION", "0") == "1"
    caption_cache_path: str | None = os.getenv("RAG_VQA_CAPTION_CACHE_PATH") or None
