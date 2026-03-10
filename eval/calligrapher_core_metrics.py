"""Thin bridge to `Calligrapher/eval/core` metrics."""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PIL import Image

DEFAULT_CALLIGRAPHER_ROOT = Path(__file__).resolve().parents[2] / "Calligrapher"
DEFAULT_VLM_PATH = "ApiCall"


@dataclass
class MetricSuiteConfig:
    calligrapher_root: Path = DEFAULT_CALLIGRAPHER_ROOT
    device: str = "cuda"
    vlm_path: str = DEFAULT_VLM_PATH
    vqa_model: str = "clip-flant5-xxl"
    metrics: tuple[str, ...] = ("ocr", "vlm", "vqa", "clip")


def _import_calligrapher_metrics(calligrapher_root: Path):
    calligrapher_root = calligrapher_root.resolve()
    if not calligrapher_root.exists():
        raise FileNotFoundError(f"Calligrapher repo not found: {calligrapher_root}")
    root_str = str(calligrapher_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    module = importlib.import_module("eval.core")
    return module.VLMMetrics, module.VQAScoreMetrics, module.CLIPMetrics


class CalligrapherCoreMetricSuite:
    """Unified six-metric evaluator backed by Calligrapher core."""

    def __init__(self, config: Optional[MetricSuiteConfig] = None):
        self.config = config or MetricSuiteConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        VLMMetrics, VQAScoreMetrics, CLIPMetrics = _import_calligrapher_metrics(
            self.config.calligrapher_root
        )

        self.metrics: dict[str, Any] = {}
        self._errors: dict[str, str] = {}
        builders = {"vqa": lambda: VQAScoreMetrics(model=self.config.vqa_model, device=self.config.device), "clip": lambda: CLIPMetrics(device=self.config.device)}

        if {"ocr", "vlm"} & set(self.config.metrics):
            try:
                self.metrics["vlm"] = VLMMetrics(model_path=self.config.vlm_path, device=self.config.device)
            except Exception as exc:
                self._errors["vlm"] = str(exc)
                self.logger.warning("Failed to initialize vlm metric: %s", exc)

        for name in ("vqa", "clip"):
            if name not in self.config.metrics:
                continue
            try:
                self.metrics[name] = builders[name]()
            except Exception as exc:
                self._errors[name] = str(exc)
                self.logger.warning("Failed to initialize %s metric: %s", name, exc)

    @property
    def init_errors(self) -> dict[str, str]:
        return dict(self._errors)

    @staticmethod
    def _normalize_ground_truth(ground_truth, prompt: str) -> str:
        if isinstance(ground_truth, list):
            return " ".join(str(item).strip() for item in ground_truth if str(item).strip())
        if ground_truth is None:
            return prompt
        text = str(ground_truth).strip()
        return text or prompt

    def evaluate_image(self, image_path: str | Path, prompt: str, ground_truth=None) -> dict[str, Any]:
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")
        ground_truth = self._normalize_ground_truth(ground_truth, prompt)
        result: dict[str, Any] = {"ground_truth": ground_truth}

        if "vlm" in self.metrics:
            try:
                scores = self.metrics["vlm"].evaluate_text_rendering(image, prompt)
                result["ocr_acc"] = float(scores["text_accuracy"])
                result["ocr_ned"] = float(scores["text_ned"])
                result["vlm_style"] = float(scores["image_quality"])
                result["vlm_faithfulness"] = float(scores["faithfulness"])
                result["vlm_recognized_text"] = scores.get("recognized_text", "")
            except Exception as exc:
                result["ocr_error"] = str(exc)
                result["vlm_error"] = str(exc)

        if "vqa" in self.metrics:
            try:
                result["vqa_score"] = float(self.metrics["vqa"].compute_score(str(image_path), prompt))
            except Exception as exc:
                result["vqa_error"] = str(exc)

        if "clip" in self.metrics:
            try:
                result["clip_score"] = float(self.metrics["clip"].compute_clip_score(str(image_path), prompt))
            except Exception as exc:
                result["clip_error"] = str(exc)

        return result
