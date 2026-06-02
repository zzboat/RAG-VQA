from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageStat

from .config import Settings
from .debug import debug_dump


class ImageDescriber:
    """Generate a short visual description for query construction."""

    def __init__(self, model_name: str, settings: Settings | None = None) -> None:
        self.model_name = model_name
        self.settings = settings or Settings()
        self._processor = None
        self._model = None
        self._torch = None
        self._device = "cpu"
        self._caption_cache: dict[str, str] | None = None
        self._caption_cache_loaded = False
        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor

            self._torch = torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = BlipProcessor.from_pretrained(
                model_name, local_files_only=self.settings.vision_local_files_only
            )
            self._model = BlipForConditionalGeneration.from_pretrained(
                model_name, local_files_only=self.settings.vision_local_files_only
            )
            self._model.to(self._device)
            self._model.eval()
            debug_dump(
                self.settings,
                "vision.caption.init",
                {"model_name": model_name, "device": self._device, "local_files_only": self.settings.vision_local_files_only},
            )
        except Exception as exc:
            self._processor = None
            self._model = None
            self._torch = None
            self._device = "cpu"
            debug_dump(
                self.settings,
                "vision.caption.init_error",
                {"model_name": model_name, "error": repr(exc), "local_files_only": self.settings.vision_local_files_only},
            )

    def describe(self, image_path: str | Path) -> str:
        path = Path(image_path)
        cache_key = str(path)
        cache = self._load_cache()
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        caption: str | None = None
        if self._processor is not None and self._model is not None and self._torch is not None:
            try:
                image = Image.open(path).convert("RGB")
                inputs = self._processor(images=image, return_tensors="pt")
                inputs = {key: value.to(self._device) for key, value in inputs.items()}
                with self._torch.no_grad():
                    output = self._model.generate(**inputs, max_new_tokens=40)
                text = self._processor.batch_decode(output, skip_special_tokens=True)[0].strip()
                if text:
                    caption = text
            except Exception as exc:
                debug_dump(
                    self.settings,
                    "vision.caption.inference_error",
                    {"image_path": str(path), "error": repr(exc)},
                )
        if caption is None:
            caption = self._fallback_description(path)
        if cache is not None:
            cache[cache_key] = caption
            self._flush_cache()
        return caption

    def _load_cache(self) -> dict[str, str] | None:
        if not self.settings.cache_caption or not self.settings.caption_cache_path:
            return None
        if self._caption_cache_loaded:
            return self._caption_cache
        cache_path = Path(self.settings.caption_cache_path)
        if cache_path.exists():
            try:
                with cache_path.open("r", encoding="utf-8") as f:
                    self._caption_cache = json.load(f)
            except Exception:
                self._caption_cache = {}
        else:
            self._caption_cache = {}
        self._caption_cache_loaded = True
        return self._caption_cache

    def _flush_cache(self) -> None:
        if self._caption_cache is None or not self.settings.caption_cache_path:
            return
        cache_path = Path(self.settings.caption_cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(self._caption_cache, f, ensure_ascii=False)

    def _fallback_description(self, path: Path) -> str:
        image = Image.open(path).convert("RGB")
        stat = ImageStat.Stat(image.resize((64, 64)))
        avg = stat.mean
        dominant = max(range(3), key=lambda i: avg[i])
        color = ["red", "green", "blue"][dominant]
        stem = path.stem.replace("_", " ").replace("-", " ")
        return f"Image named '{stem}', with a visually dominant {color} color tone."


class VisualQuestionAnswerer:
    """Direct VQA model used as visual-only evidence."""

    def __init__(self, model_name: str, settings: Settings | None = None, enabled: bool = True) -> None:
        self.model_name = model_name
        self.settings = settings or Settings()
        self.enabled = enabled
        self._processor = None
        self._model = None
        self._torch = None
        self._device = "cpu"
        if enabled:
            try:
                import torch
                from transformers import BlipForQuestionAnswering, BlipProcessor

                self._torch = torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._processor = BlipProcessor.from_pretrained(
                    model_name, local_files_only=self.settings.vision_local_files_only
                )
                self._model = BlipForQuestionAnswering.from_pretrained(
                    model_name, local_files_only=self.settings.vision_local_files_only
                )
                self._model.to(self._device)
                self._model.eval()
                debug_dump(
                    self.settings,
                    "vision.vqa.init",
                    {"model_name": model_name, "device": self._device, "local_files_only": self.settings.vision_local_files_only},
                )
            except Exception as exc:
                self._processor = None
                self._model = None
                self._torch = None
                self._device = "cpu"
                debug_dump(
                    self.settings,
                    "vision.vqa.init_error",
                    {"model_name": model_name, "error": repr(exc), "local_files_only": self.settings.vision_local_files_only},
                )

    def answer(self, image_path: str | Path, question: str) -> str | None:
        if self._processor is None or self._model is None or self._torch is None:
            return None
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=image, text=question, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.no_grad():
                output = self._model.generate(**inputs, max_new_tokens=40)
            text = self._processor.batch_decode(output, skip_special_tokens=True)[0].strip()
            return text or None
        except Exception as exc:
            debug_dump(
                self.settings,
                "vision.vqa.inference_error",
                {"image_path": str(image_path), "question": question, "error": repr(exc)},
            )
            return None
