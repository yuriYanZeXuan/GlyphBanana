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
python3 generate.py \
    --prompt 'A whiteboard displaying "E=mc²" in a classroom' \
    --text "E=mc²" \
    --output output.png
```

### Test Both Backends

```bash
bash scripts/test_backends.sh
```

Only test one backend:

```bash
bash scripts/test_backends.sh zimage
bash scripts/test_backends.sh qwen
```

### Evaluate Text Accuracy

```bash
python3 evaluate.py \
    --image output.png \
    --prompt 'A whiteboard displaying "E=mc²"'
```

### Batch Evaluation

```bash
python3 evaluate.py \
    --image_dir outputs/ \
    --prompt_file prompts.json \
    --output results.json
```

## Usage

### Generation Options

```bash
python3 generate.py \
    --prompt "Your prompt with text description" \
    --text "Text to render" "More text" \
    --output result.png \
    --backend zimage \
    --steps 20 \
    --seed 42 \
    --height 1024 \
    --width 1024 \
    --no-harmonize  # Skip Pass 3
```

Backend-specific examples:

```bash
python3 generate.py \
    --backend zimage \
    --prompt 'A storefront sign saying "CAFE"' \
    --text "CAFE" \
    --output output_zimage.png

python3 generate.py \
    --backend qwen \
    --prompt 'A storefront sign saying "CAFE"' \
    --text "CAFE" \
    --output output_qwen.png \
    --qwen-true-cfg-scale 4.0
```

### Evaluation Options

```bash
python3 evaluate.py \
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
├── generate.py          # Main generation script
├── evaluate.py          # OCR evaluation script (182 lines)
├── example.py           # Usage examples
├── requirements.txt     # Dependencies
├── scripts/             # Simple backend test scripts
│   └── test_backends.sh
├── infer/               # Core inference modules
│   ├── VLM_agent.py         # VLM API interface
│   ├── glyph_injector.py    # Glyph rendering and latent injection
│   ├── formula_helper.py    # Text/LaTeX rendering
│   └── attn_enhancement.py  # Attention enhancement
├── models/              # Local model definitions
│   ├── zimage_ip/       # Z-Image backend
│   ├── qwen_ip/         # Qwen-Image backend
│   └── fluxklein/       # Pass 3 harmonization backend
├── baselines/           # Other baseline assets
└── eval/                # Evaluation module
```

## Model Requirements

- **Backends**: Z-Image and Qwen-Image
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
