#!/usr/bin/env python3
"""Gradio demo for GlyphBanana generation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate import build_backend, build_generation_config
from infer.formula_helper import get_font_registry
from infer.generation_utils import DEFAULT_KLEIN_MODEL_PATH

MAX_LINES = 5
CANVAS_SIZE = 512
_BACKEND_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser(description="GlyphBanana Gradio demo")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def get_backend(backend: str, model_path: str, device: str):
    cache_key = (backend, model_path or "", device)
    if cache_key not in _BACKEND_CACHE:
        _BACKEND_CACHE[cache_key] = build_backend(backend=backend, model_path=model_path or None, device=device)
    return _BACKEND_CACHE[cache_key]


def parse_texts(raw: str) -> list[str]:
    quoted = re.findall(r'"([^"]+)"', raw or "")
    if quoted:
        return quoted[:MAX_LINES]
    lines = [line.strip() for line in (raw or "").splitlines() if line.strip()]
    if lines:
        return lines[:MAX_LINES]
    raw = (raw or "").strip()
    return [raw] if raw else []


def normalize_color(value: str | None) -> str:
    if not value:
        return "#FFFFFF"
    if isinstance(value, str) and value.startswith("rgba"):
        nums = [int(float(x.strip())) for x in value[value.find("(") + 1 : value.find(")")].split(",")[:3]]
        return "#{:02X}{:02X}{:02X}".format(*nums)
    return value


def blank_canvas(size: int = CANVAS_SIZE):
    bg = np.full((size, size, 3), 255, dtype=np.uint8)
    layer = np.zeros((size, size, 4), dtype=np.uint8)
    return {"background": bg, "layers": [layer], "composite": bg}


def extract_mask(sketch_value) -> np.ndarray:
    if sketch_value is None:
        return np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)

    if isinstance(sketch_value, dict):
        layers = sketch_value.get("layers") or []
        if layers:
            layer = layers[0]
            if isinstance(layer, np.ndarray):
                if layer.ndim == 3 and layer.shape[-1] >= 4:
                    alpha = layer[..., 3]
                    if alpha.max() > 0:
                        return alpha.astype(np.uint8)
                if layer.ndim == 3:
                    gray = cv2.cvtColor(layer.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    if gray.max() > 0:
                        return gray
        bg = sketch_value.get("background")
        if isinstance(bg, np.ndarray):
            gray = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            return (255 - gray).clip(0, 255).astype(np.uint8)

    if isinstance(sketch_value, np.ndarray):
        if sketch_value.ndim == 3:
            return cv2.cvtColor(sketch_value.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return sketch_value.astype(np.uint8)

    return np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)


def auto_layout_boxes(n_texts: int) -> list[list[float]]:
    if n_texts <= 0:
        return []
    margin_x = 0.1
    start_y = 0.15
    total_h = 0.7
    line_h = total_h / max(n_texts, 1)
    boxes = []
    for idx in range(n_texts):
        y1 = start_y + idx * line_h + 0.05 * line_h
        y2 = start_y + (idx + 1) * line_h - 0.05 * line_h
        boxes.append([margin_x, y1, 1 - margin_x, y2])
    return boxes


def mask_to_boxes(mask: np.ndarray, n_texts: int, sort_priority: str) -> list[list[float]]:
    if mask.size == 0:
        return auto_layout_boxes(n_texts)

    _, thresh = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    boxes = []
    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if area < 64:
            continue
        cx, cy = centroids[idx]
        boxes.append((x, y, w, h, cx, cy))

    if not boxes:
        return auto_layout_boxes(n_texts)

    if sort_priority == "top-to-bottom":
        boxes.sort(key=lambda item: (item[5], item[4]))
    else:
        boxes.sort(key=lambda item: (item[4], item[5]))

    height, width = mask.shape[:2]
    normalized = []
    for x, y, w, h, _, _ in boxes[:n_texts]:
        normalized.append([x / width, y / height, (x + w) / width, (y + h) / height])

    if len(normalized) < n_texts:
        normalized.extend(auto_layout_boxes(n_texts - len(normalized)))
    return normalized[:n_texts]


def build_text_regions(texts: list[str], boxes: list[list[float]], fonts: list[str], colors: list[str]) -> list[dict]:
    regions = []
    for idx, text in enumerate(texts):
        font_name = fonts[idx] if idx < len(fonts) else "auto"
        color = colors[idx] if idx < len(colors) else "#FFFFFF"
        regions.append(
            {
                "content": text,
                "bbox": boxes[idx],
                "font": None if font_name == "auto" else font_name,
                "color": color,
                "font_weight": "regular",
                "font_size_ratio": 0.8,
                "alignment": "center",
                "rotation": 0,
                "is_latex": "\\" in text or "$" in text,
            }
        )
    return regions


def preview_layout(boxes: list[list[float]]) -> np.ndarray:
    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), 255, dtype=np.uint8)
    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        pt1 = (int(x1 * CANVAS_SIZE), int(y1 * CANVAS_SIZE))
        pt2 = (int(x2 * CANVAS_SIZE), int(y2 * CANVAS_SIZE))
        cv2.rectangle(canvas, pt1, pt2, (60, 120, 255), 2)
        cv2.putText(canvas, str(idx), (pt1[0] + 4, pt1[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 80, 80), 2)
    return canvas


def run_generation(
    prompt,
    text_input,
    backend,
    model_path,
    device,
    layout_mode,
    sort_priority,
    sketchpad,
    width,
    height,
    steps,
    seed,
    no_harmonize,
    true_cfg_scale,
    guidance_scale,
    klein_model_path,
    klein_steps,
    klein_guidance,
    font1,
    font2,
    font3,
    font4,
    font5,
    color1,
    color2,
    color3,
    color4,
    color5,
):
    texts = parse_texts(text_input)
    if not prompt.strip():
        raise gr.Error("请输入 prompt。")
    if not texts:
        raise gr.Error("请输入至少一行指定 text，支持逐行输入或双引号格式。")

    fonts = [font1, font2, font3, font4, font5]
    colors = [normalize_color(color1), normalize_color(color2), normalize_color(color3), normalize_color(color4), normalize_color(color5)]

    text_regions = None
    layout_preview = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), 255, dtype=np.uint8)
    plan_json = {"mode": "vlm_planner"}

    if layout_mode == "brush_layout":
        mask = extract_mask(sketchpad)
        boxes = mask_to_boxes(mask, len(texts), sort_priority)
        text_regions = build_text_regions(texts, boxes, fonts, colors)
        layout_preview = preview_layout(boxes)
        plan_json = {"mode": "brush_layout", "text_regions": text_regions}

    runner = get_backend(backend, model_path, device)
    config = build_generation_config(
        backend,
        seed=seed,
        steps=steps,
        height=height,
        width=width,
        no_harmonize=no_harmonize,
        klein_model_path=klein_model_path,
        klein_steps=klein_steps,
        klein_guidance=klein_guidance,
        true_cfg_scale=true_cfg_scale,
        guidance_scale=guidance_scale if guidance_scale != 0 else None,
    )
    image = runner.generate(prompt=prompt, text_contents=texts, text_regions=text_regions, config=config)

    return image, layout_preview, json.dumps(plan_json, ensure_ascii=False, indent=2)


def build_demo():
    font_choices = ["auto"] + sorted(get_font_registry().keys())

    with gr.Blocks(title="GlyphBanana Demo") as demo:
        gr.Markdown(
            """
            # GlyphBanana Demo
            支持 `zimage` / `qwen-image` 两个后端。
            默认使用 VLM 自动分析 layout；切到 `Brush Layout` 后，可以通过涂抹区域和逐行字体设置替代 planner。
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="Prompt", lines=3, placeholder='例如：A poster on a wall that reads "GlyphBanana"')
                text_input = gr.Textbox(
                    label="Specified Text",
                    lines=5,
                    placeholder='逐行输入，或写成 "line1" "line2"',
                )

                with gr.Row():
                    backend = gr.Dropdown(["zimage", "qwen-image"], value="zimage", label="Backend")
                    device = gr.Textbox(value="cuda", label="Device")
                model_path = gr.Textbox(label="Model Path (optional)", placeholder="留空使用默认模型路径")

                with gr.Row():
                    layout_mode = gr.Radio(
                        ["vlm_planner", "brush_layout"],
                        value="vlm_planner",
                        label="Layout Mode",
                    )
                    sort_priority = gr.Radio(
                        ["top-to-bottom", "left-to-right"],
                        value="top-to-bottom",
                        label="Brush Sort",
                    )

                sketchpad = gr.Sketchpad(
                    label="Brush Layout Canvas",
                    value=blank_canvas(),
                    brush=gr.Brush(default_size=32, default_color="rgb(80, 80, 80)"),
                    layers=False,
                    transforms=(),
                    height=CANVAS_SIZE,
                )

                with gr.Row():
                    width = gr.Slider(512, 1024, value=1024, step=64, label="Width")
                    height = gr.Slider(512, 1024, value=1024, step=64, label="Height")
                with gr.Row():
                    steps = gr.Slider(1, 60, value=20, step=1, label="Steps")
                    seed = gr.Slider(0, 99999999, value=42, step=1, label="Seed")
                    no_harmonize = gr.Checkbox(label="Disable Harmonization", value=False)
                with gr.Row():
                    true_cfg_scale = gr.Slider(1.0, 8.0, value=4.0, step=0.1, label="Qwen True CFG")
                    guidance_scale = gr.Slider(0.0, 10.0, value=0.0, step=0.1, label="Qwen Guidance")
                with gr.Accordion("Harmonization", open=False):
                    klein_model_path = gr.Textbox(value=DEFAULT_KLEIN_MODEL_PATH, label="Klein Model Path")
                    with gr.Row():
                        klein_steps = gr.Slider(1, 30, value=10, step=1, label="Klein Steps")
                        klein_guidance = gr.Slider(1.0, 10.0, value=4.0, step=0.1, label="Klein Guidance")

                gr.Markdown("### 字体与颜色（前 5 行）")
                with gr.Row():
                    font1 = gr.Dropdown(font_choices, value="auto", label="Line 1 Font")
                    color1 = gr.ColorPicker(value="#FFFFFF", label="Line 1 Color")
                with gr.Row():
                    font2 = gr.Dropdown(font_choices, value="auto", label="Line 2 Font")
                    color2 = gr.ColorPicker(value="#FFFFFF", label="Line 2 Color")
                with gr.Row():
                    font3 = gr.Dropdown(font_choices, value="auto", label="Line 3 Font")
                    color3 = gr.ColorPicker(value="#FFFFFF", label="Line 3 Color")
                with gr.Row():
                    font4 = gr.Dropdown(font_choices, value="auto", label="Line 4 Font")
                    color4 = gr.ColorPicker(value="#FFFFFF", label="Line 4 Color")
                with gr.Row():
                    font5 = gr.Dropdown(font_choices, value="auto", label="Line 5 Font")
                    color5 = gr.ColorPicker(value="#FFFFFF", label="Line 5 Color")

                run_button = gr.Button("Generate", variant="primary")

            with gr.Column(scale=2):
                output_image = gr.Image(label="Generated Image")
                layout_preview = gr.Image(label="Layout Preview")
                plan_json = gr.Code(label="Resolved Plan", language="json")

        run_button.click(
            fn=run_generation,
            inputs=[
                prompt,
                text_input,
                backend,
                model_path,
                device,
                layout_mode,
                sort_priority,
                sketchpad,
                width,
                height,
                steps,
                seed,
                no_harmonize,
                true_cfg_scale,
                guidance_scale,
                klein_model_path,
                klein_steps,
                klein_guidance,
                font1,
                font2,
                font3,
                font4,
                font5,
                color1,
                color2,
                color3,
                color4,
                color5,
            ],
            outputs=[output_image, layout_preview, plan_json],
        )

    return demo


def main():
    args = parse_args()
    demo = build_demo()
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
