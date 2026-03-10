# GlyphBanana: Text Rendering with Glyph Injection

GlyphBanana is a text-to-image generation framework designed for high-quality text rendering in images. It uses a three-stage pipeline with VLM-guided typography planning and glyph injection for precise text placement.

## Overview

The framework consists of three main stages:

1. **Pass 1 (Reference)**: Generate a reference image using the full prompt
2. **VLM Planning**: Analyze the reference image and plan typography layout (bbox, color, font)
3. **Pass 2 (Injection)**: Generate with a clean prompt and inject glyph latents at specified regions
4. **Pass 3 (Harmonization)**: Optional style transfer to harmonize text with background

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd GlyphBanana

# Install dependencies
pip install -r requirements.txt

# Set up VLM API credentials
export QST_BASE_URL="your-api-url"
export QST_API_KEY="your-api-key"
```

## Quick Start

### Generate an Image

```bash
python generate.py \
    --prompt 'A whiteboard displaying "E=mc²" in a classroom' \
    --text "E=mc²" \
    --output output.png
```

### Generate with `qwen-image`

```bash
python generate.py \
    --backend qwen-image \
    --prompt 'A whiteboard displaying "E=mc²" in a classroom' \
    --text "E=mc²" \
    --output output_qwen.png
```

### Evaluate Text Accuracy

```bash
python evaluate.py \
    --image output.png \
    --prompt 'A whiteboard displaying "E=mc²"'
```

### Batch Evaluation

```bash
python evaluate.py \
    --image_dir outputs/ \
    --prompt_file prompts.json \
    --output results.json
```

### GlyphBanana-Benchmark Batch Generation

```bash
python scripts/batch_generate.py \
    --dataset-dir eval/GlyphBanana-Benchmark \
    --output-dir outputs/glyphbanana_benchmark_zimage \
    --backend zimage
```

### GlyphBanana-Benchmark Batch Evaluation

```bash
python scripts/batch_evaluate.py \
    --dataset-dir eval/GlyphBanana-Benchmark \
    --image-dir outputs/glyphbanana_benchmark_zimage \
    --output outputs/glyphbanana_benchmark_zimage/results.json
```

### Gradio Demo

```bash
python demo/gradio_app.py --server-port 7860
```

## Usage

### Generation Options

```bash
python generate.py \
    --backend zimage \
    --prompt "Your prompt with text description" \
    --text "Text to render" "More text" \
    --output result.png \
    --steps 20 \
    --seed 42 \
    --height 1024 \
    --width 1024 \
    --no-harmonize  # Skip Pass 3
```

手工 layout 可通过 `--text-regions-file regions.json` 传入，格式为：

```json
[
  {
    "content": "GlyphBanana",
    "bbox": [0.1, 0.35, 0.9, 0.55],
    "font": "auto",
    "color": "#FFFFFF"
  }
]
```

### Evaluation Options

```bash
python evaluate.py \
    --image_dir results/ \
    --prompt_file data.json \
    --vlm_model qwen3-vl-235b-a22b-instruct \
    --output scores.json
```

The `prompts.json` file should map image filenames to their expected text:

```json
{
  "image1.png": "A sign saying \"Hello World\"",
  "image2.png": "A poster with \"Sale 50% Off\""
}
```

## Project Structure

```
GlyphBanana/
├── generate.py          # Unified generation entry for zimage/qwen-image
├── evaluate.py          # OCR evaluation script + reusable helpers
├── example.py           # Usage examples
├── demo/                # Gradio demo
├── scripts/             # Batch generation/evaluation scripts
├── requirements.txt     # Dependencies
├── infer/               # Core inference modules
│   ├── VLM_agent.py         # VLM API interface
│   ├── glyph_injector.py    # Glyph rendering and latent injection
│   ├── qwen_image_inference.py # Qwen-image backend
│   ├── formula_helper.py    # Text/LaTeX rendering
│   └── attn_enhancement.py  # Attention enhancement
├── train/zimage_ip/     # ZImage pipeline (core generation model)
├── baselines/           # Minimal baselines
│   ├── fluxklein/       # Pass 3 harmonization (required)
│   └── results/         # Output directory
└── eval/                # Evaluation module and benchmark datasets
```

## Model Requirements

- **Base Model**: FLUX.1-Fill-dev or compatible
- **VLM API**: OpenAI-compatible API for typography analysis and OCR
- **Optional**: FluxKlein for harmonization (Pass 3)

## API Configuration

Set environment variables for VLM access:

```bash
export QST_BASE_URL="https://your-api-endpoint.com/v1"
export QST_API_KEY="your-api-key"
export QST_API_KEY2="backup-api-key"  # Optional
```

## Citation

```bibtex
@article{glyphbanana2025,
  title={GlyphBanana: Text Rendering with Glyph Injection},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT License
