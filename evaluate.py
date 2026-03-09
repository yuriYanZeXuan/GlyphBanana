#!/usr/bin/env python3
"""
GlyphBanana: Lightweight OCR evaluation script for text rendering quality.

Usage:
    python evaluate.py --image output.png --prompt 'A sign saying "Hello World"'
    python evaluate.py --image_dir outputs/ --prompt_file prompts.json
"""

import os
import re
import json
import base64
import argparse
from io import BytesIO
from pathlib import Path
from typing import Optional
from PIL import Image


def extract_text_from_prompt(prompt: str) -> str:
    """Extract quoted text from prompt (supports English/Chinese quotes)."""
    # English double quotes, single quotes, Chinese quotes
    patterns = [
        r'"([^"]+)"',
        r"'([^']+)'",
        r'"([^"]+)"',
        r'\u2018([^\u2019]+)\u2019',
        r'\u300c([^\u300d]+)\u300d',
        r'\u300e([^\u300f]+)\u300f',
    ]
    matches = re.findall('|'.join(patterns), prompt)
    texts = [g for m in matches for g in m if g]
    return ' '.join(texts) if texts else prompt


def encode_image_b64(image: Image.Image) -> str:
    """PIL Image to base64 string."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def compute_text_accuracy(ground_truth: str, recognized: str) -> float:
    """Compute character-level accuracy using Levenshtein distance."""
    try:
        import Levenshtein
    except ImportError:
        raise ImportError("pip install python-Levenshtein")
    gt = ' '.join(ground_truth.lower().split())
    rec = ' '.join(recognized.lower().split())
    if not gt:
        return 1.0 if not rec else 0.0
    return max(0.0, 1 - Levenshtein.distance(gt, rec) / len(gt))


class VLMOCR:
    """VLM-based OCR evaluator using OpenAI-compatible API."""
    OCR_PROMPT = (
        "Please read and output ALL the text content visible in this image.\n"
        "Only output the text you can see, nothing else.\n"
        "Do not add explanations, just the raw text content."
    )

    def __init__(self, model: str = "qwen3-vl-235b-a22b-instruct",
                 api_key: Optional[str] = None, base_url: Optional[str] = None):
        from openai import OpenAI
        self.model = model
        self.base_url = base_url or os.getenv("QST_BASE_URL")
        api_key = api_key or os.getenv("QST_API_KEY")
        if not self.base_url or not api_key:
            raise ValueError("Set QST_BASE_URL and QST_API_KEY env vars")
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

    def ocr(self, image: Image.Image) -> str:
        """Perform OCR on image, return recognized text."""
        b64 = encode_image_b64(image)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": self.OCR_PROMPT},
            ]}],
            max_tokens=512, temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]
        return text

    def evaluate(self, image: Image.Image, prompt: str) -> dict:
        """Evaluate image against expected text from prompt."""
        ground_truth = extract_text_from_prompt(prompt)
        recognized = self.ocr(image)
        return {
            "ground_truth": ground_truth,
            "recognized": recognized,
            "accuracy": compute_text_accuracy(ground_truth, recognized),
        }


def load_prompts(prompt_file: str) -> dict:
    """Load prompts from JSON file."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        return {item.get('image', item.get('filename')): item.get('prompt', item.get('text'))
                for item in data if isinstance(item, dict)}
    return {}


def evaluate_single(ocr: VLMOCR, image_path: Path, prompt: str) -> dict:
    """Evaluate a single image."""
    print(f"\n📷 {image_path.name}")
    image = Image.open(image_path).convert('RGB')
    result = ocr.evaluate(image, prompt)
    gt_short = result['ground_truth'][:60] + '...' if len(result['ground_truth']) > 60 else result['ground_truth']
    rec_short = result['recognized'][:60] + '...' if len(result['recognized']) > 60 else result['recognized']
    print(f"   Expected: \"{gt_short}\"")
    print(f"   OCR:      \"{rec_short}\"")
    print(f"   Accuracy: {result['accuracy']:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description="GlyphBanana OCR Evaluation")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--image_dir", type=str, help="Directory containing images")
    parser.add_argument("--prompt", type=str, help="Text prompt for single image")
    parser.add_argument("--prompt_file", type=str, help="JSON file with prompts")
    parser.add_argument("--vlm_model", type=str, default="qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--extensions", nargs='+', default=['.png', '.jpg', '.jpeg'])
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Either --image or --image_dir is required")

    print("🔍 Initializing VLM OCR...")
    ocr = VLMOCR(model=args.vlm_model)
    prompts = load_prompts(args.prompt_file) if args.prompt_file else {}

    images = [Path(args.image)] if args.image else \
             [p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in args.extensions]
    images.sort()

    if not images:
        print("❌ No images found!")
        return 1

    print(f"\n🚀 Evaluating {len(images)} image(s)...")
    results, total_acc = [], 0.0

    for img_path in images:
        prompt = args.prompt if (args.prompt and len(images) == 1) else \
                 prompts.get(img_path.name, prompts.get(img_path.stem))
        if not prompt:
            print(f"⚠️  No prompt for {img_path.name}, skipping...")
            continue
        result = evaluate_single(ocr, img_path, prompt)
        result["image"] = img_path.name
        results.append(result)
        total_acc += result["accuracy"]

    if results:
        avg_acc = total_acc / len(results)
        print(f"\n{'='*50}")
        print(f"📊 Summary: {len(results)} images, Avg Accuracy: {avg_acc:.4f}")
        print(f"{'='*50}")
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump({"average_accuracy": avg_acc, "num_images": len(results), "results": results},
                          f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
