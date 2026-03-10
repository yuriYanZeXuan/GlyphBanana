#!/usr/bin/env python3
"""Batch evaluation for GlyphBanana-Benchmark outputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_core_metrics = _load_module("glyphbanana_calligrapher_core_metrics", EVAL_DIR / "calligrapher_core_metrics.py")
_benchmark = _load_module("glyphbanana_benchmark", EVAL_DIR / "glyphbanana_benchmark.py")

DEFAULT_CALLIGRAPHER_ROOT = _core_metrics.DEFAULT_CALLIGRAPHER_ROOT
DEFAULT_VLM_PATH = _core_metrics.DEFAULT_VLM_PATH
CalligrapherCoreMetricSuite = _core_metrics.CalligrapherCoreMetricSuite
MetricSuiteConfig = _core_metrics.MetricSuiteConfig
load_glyphbanana_benchmark_samples = _benchmark.load_glyphbanana_benchmark_samples


def parse_args():
    parser = argparse.ArgumentParser(description="GlyphBanana batch evaluation")
    parser.add_argument("--dataset-dir", default=str(ROOT / "eval" / "GlyphBanana-Benchmark"))
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--calligrapher-root", default=str(DEFAULT_CALLIGRAPHER_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vlm-path", default=DEFAULT_VLM_PATH)
    parser.add_argument("--vqa-model", default="clip-flant5-xxl")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["ocr", "vlm", "vqa", "clip"],
        default=["ocr", "vlm", "vqa", "clip"],
        help="启用的评测模块；最终会产出六项指标：ocr_acc, ocr_ned, vlm_style, vlm_faithfulness, vqa_score, clip_score",
    )
    parser.add_argument("--missing-policy", choices=["skip", "error"], default="skip")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sources", nargs="*", help="指定 jsonl stem 或文件名过滤")
    return parser.parse_args()


def summarize_by_source(results: list[dict], metric_keys: list[str]) -> dict:
    grouped = defaultdict(list)
    for result in results:
        grouped[result["source_name"]].append(result)

    summary = {}
    for name, rows in sorted(grouped.items()):
        item = {"num_images": len(rows)}
        for metric in metric_keys:
            values = [row[metric] for row in rows if isinstance(row.get(metric), (int, float))]
            if values:
                item[metric] = sum(values) / len(values)
        summary[name] = item
    return summary


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = load_glyphbanana_benchmark_samples(
        args.dataset_dir,
        include_sources=args.sources,
        offset=args.offset,
        limit=args.limit,
    )
    if not samples:
        raise SystemExit("No samples found.")

    evaluator = CalligrapherCoreMetricSuite(
        MetricSuiteConfig(
            calligrapher_root=Path(args.calligrapher_root),
            device=args.device,
            vlm_path=args.vlm_path,
            vqa_model=args.vqa_model,
            metrics=tuple(args.metrics),
        )
    )
    results = []

    for index, sample in enumerate(samples):
        image_path = image_dir / f"result_{sample['sample_id']}.png"
        if not image_path.exists():
            message = f"Missing image: {image_path.name}"
            if args.missing_policy == "error":
                raise FileNotFoundError(message)
            print(f"[skip] {message}")
            continue

        print(f"[{index + 1}/{len(samples)}] {image_path.name}")
        result = evaluator.evaluate_image(
            image_path=image_path,
            prompt=sample["prompt"],
            ground_truth=sample["ground_truth"],
        )
        result["sample_id"] = sample["sample_id"]
        result["prompt"] = sample["prompt"]
        result["text"] = sample["text"]
        result["source_name"] = sample["source_name"]
        results.append(result)

    metric_keys = ["ocr_acc", "ocr_ned", "vlm_style", "vlm_faithfulness", "vqa_score", "clip_score"]
    summary = {}
    for metric in metric_keys:
        values = [item[metric] for item in results if isinstance(item.get(metric), (int, float))]
        if values:
            summary[metric] = sum(values) / len(values)

    payload = {
        "dataset_dir": str(args.dataset_dir),
        "image_dir": str(image_dir),
        "num_images": len(results),
        "metrics": args.metrics,
        "calligrapher_root": args.calligrapher_root,
        "init_errors": evaluator.init_errors,
        "summary": summary,
        "by_source": summarize_by_source(results, metric_keys),
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved evaluation to {output_path}")
    for metric, value in summary.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
