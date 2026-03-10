#!/usr/bin/env python3
"""Batch generate images for GlyphBanana-Benchmark datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.glyphbanana_benchmark import build_prompt_map, load_glyphbanana_benchmark_samples
from generate import build_backend, build_generation_config
from infer.generation_utils import DEFAULT_KLEIN_MODEL_PATH


def parse_args():
    parser = argparse.ArgumentParser(description="GlyphBanana batch generation")
    parser.add_argument("--dataset-dir", default=str(ROOT / "eval" / "GlyphBanana-Benchmark"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=["zimage", "qwen-image"], default="zimage")
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--klein-model-path", default=DEFAULT_KLEIN_MODEL_PATH)
    parser.add_argument("--klein-steps", type=int, default=10)
    parser.add_argument("--klein-guidance", type=float, default=4.0)
    parser.add_argument("--no-harmonize", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sources", nargs="*", help="指定 jsonl stem 或文件名过滤")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_glyphbanana_benchmark_samples(
        args.dataset_dir,
        include_sources=args.sources,
        offset=args.offset,
        limit=args.limit,
    )
    if not samples:
        raise SystemExit("No samples found.")

    generator = build_backend(
        backend=args.backend,
        model_path=args.model_path,
        device=args.device,
    )

    manifest = []
    completed_samples = []

    for index, sample in enumerate(samples):
        output_path = output_dir / f"result_{sample['sample_id']}.png"
        if args.skip_existing and output_path.exists():
            print(f"[skip] {output_path.name}")
            completed_samples.append(sample)
            manifest.append(
                {
                    "sample_id": sample["sample_id"],
                    "image": output_path.name,
                    "prompt": sample["prompt"],
                    "text": sample["text"],
                    "source_name": sample["source_name"],
                    "skipped": True,
                }
            )
            continue

        sample_seed = args.seed + index
        print(f"[{index + 1}/{len(samples)}] {sample['sample_id']}")
        config = build_generation_config(
            args.backend,
            seed=sample_seed,
            steps=args.steps,
            height=args.height,
            width=args.width,
            no_harmonize=args.no_harmonize,
            klein_model_path=args.klein_model_path,
            klein_steps=args.klein_steps,
            klein_guidance=args.klein_guidance,
            true_cfg_scale=args.true_cfg_scale,
            guidance_scale=args.guidance_scale,
        )
        image = generator.generate(
            prompt=sample["prompt"],
            text_contents=sample["text"],
            config=config,
            output_path=str(output_path),
        )

        image.save(output_path)
        completed_samples.append(sample)
        manifest.append(
            {
                "sample_id": sample["sample_id"],
                "image": output_path.name,
                "prompt": sample["prompt"],
                "text": sample["text"],
                "source_name": sample["source_name"],
                "skipped": False,
            }
        )

    prompts = build_prompt_map(completed_samples)
    with open(output_dir / "prompts.json", "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(manifest)} records to {output_dir}")


if __name__ == "__main__":
    main()
