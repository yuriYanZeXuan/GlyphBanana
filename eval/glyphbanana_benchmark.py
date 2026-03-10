"""GlyphBanana-Benchmark helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional


def normalize_text_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def iter_benchmark_files(dataset_dir: str | Path) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    return sorted(dataset_dir.glob("*.jsonl"))


def load_glyphbanana_benchmark_samples(
    dataset_dir: str | Path,
    include_sources: Optional[Iterable[str]] = None,
    offset: int = 0,
    limit: Optional[int] = None,
) -> list[dict]:
    dataset_dir = Path(dataset_dir)
    selected = set(include_sources or [])
    samples = []

    for jsonl_path in iter_benchmark_files(dataset_dir):
        if selected and jsonl_path.stem not in selected and jsonl_path.name not in selected:
            continue

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for row_idx, line in enumerate(f):
                if not line.strip():
                    continue
                record = json.loads(line)
                prompt_id = record.get("prompt_id", row_idx)
                sample_id = f"{jsonl_path.stem}_{prompt_id}"
                text_list = normalize_text_list(record.get("text"))
                samples.append(
                    {
                        "sample_id": sample_id,
                        "source_file": jsonl_path.name,
                        "source_name": jsonl_path.stem,
                        "prompt_id": prompt_id,
                        "prompt": record.get("prompt", ""),
                        "text": text_list,
                        "ground_truth": " ".join(text_list),
                        "raw": record,
                    }
                )

    samples = samples[offset:]
    if limit is not None:
        samples = samples[:limit]
    return samples


def build_prompt_map(samples: list[dict], image_template: str = "result_{sample_id}.png") -> dict[str, str]:
    return {
        image_template.format(sample_id=sample["sample_id"]): sample["prompt"]
        for sample in samples
    }
