#!/bin/bash

# Flux-Klein Text-to-Image and Image Editing Script
# Usage: bash run.sh

# Example 1: Text-to-Image Generation
echo "=== Example 1: Text-to-Image Generation ==="
python inference_fluxklein.py \
  --prompt "A cat holding a sign that says hello world" \
  --output_path "output/fluxklein_t2i.png" \
  --seed 42 \
  --steps 50 \
  --guidance_scale 4.0 \
  --height 1024 \
  --width 1024

# Example 2: Image Editing (uncomment and provide an image path)
# echo "=== Example 2: Image Editing ==="
# python inference_fluxklein.py \
#   --prompt "A dog holding a sign that says hello world" \
#   --image_path "path/to/input_image.png" \
#   --output_path "output/fluxklein_edit.png" \
#   --seed 42 \
#   --steps 50 \
#   --guidance_scale 4.0
