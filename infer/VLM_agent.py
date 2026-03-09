"""
VLM Agent: 统一的 VLM 调用中心

所有 VLM/LLM 调用、客户端初始化和 prompt 模板集中在此文件。
提供 Agent 式的排版分析、prompt 改写、图像评分等功能。
"""

import os
import re
import json
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

# 加载环境变量
load_dotenv(Path(__file__).parent.parent / ".env")


def _extract_text_from_prompt(prompt: str) -> str:
    """Extract quoted text from prompt (supports English/Chinese quotes)."""
    patterns = [
        r'"([^"]+)"',          # English double quotes
        r"'([^']+)'",          # English single quotes
        r'"([^"]+)"',          # Chinese double quotes
        r'\u2018([^\u2019]+)\u2019',  # Chinese single quotes
        r'\u300c([^\u300d]+)\u300d',  # Corner brackets
        r'\u300e([^\u300f]+)\u300f'   # Double corner brackets
    ]
    matches = re.findall('|'.join(patterns), prompt)
    texts = [g for m in matches for g in m if g]
    return ' '.join(texts) if texts else prompt


# ============ Prompt 模板 ============

ANALYZE_TYPOGRAPHY_PROMPT = (
    "You are an expert in image typography analysis. Given a reference image with a 5×5 grid and coordinate annotations, "
    "analyze the natural text rendering style and overall scene. Then plan the best typography layout for each text/formula item.\n\n"
    "CRITICAL: The reference image shows text that is FLAT and FACING the screen directly (frontal view, no perspective distortion). "
    "You must plan bboxes that are also flat and frontal - bboxes should have parallel top and bottom edges.\n\n"
    "IMPORTANT: The red grid lines and coordinate labels are ONLY positioning aids. Ignore them in analysis.\n\n"
    "For each text block, determine:\n"
    "- content: the text to render\n"
    "- bbox: [x_min, y_min, x_max, y_max] in 0-1 range\n"
    "- font: font name or 'auto'\n"
    "- font_weight: light/regular/bold\n"
    "- font_size_ratio: 0.1-1.0 relative to bbox height\n"
    "- color: white, black, red, blue, green, yellow, orange, brown, gray, gold, silver, purple, pink\n"
    "- is_latex: true/false\n"
    "- alignment: left/center/right\n"
    "- rotation: text rotation angle in degrees (-30 to 30)\n\n"
    "Available fonts: {font_list}\n\n"
    "Rules:\n"
    "- bboxes must not overlap or exceed image bounds\n"
    "- bboxes must be FLAT (no perspective distortion)\n"
    "- color must be one of the predefined names\n\n"
    "Output strictly in JSON format:\n"
    '```json\n'
    '{{\n'
    '  "image_analysis": {{\n'
    '    "background_style": "description",\n'
    '    "dominant_colors": ["#hex1", "#hex2"],\n'
    '    "text_style_hint": "description"\n'
    '  }},\n'
    '  "text_regions": [\n'
    '    {{\n'
    '      "content": "text",\n'
    '      "bbox": [x_min, y_min, x_max, y_max],\n'
    '      "font": "auto",\n'
    '      "font_weight": "regular",\n'
    '      "font_size_ratio": 0.7,\n'
    '      "color": "white",\n'
    '      "is_latex": false,\n'
    '      "alignment": "center",\n'
    '      "rotation": 0\n'
    '    }}\n'
    '  ]\n'
    '}}\n'
    "```"
)

GENERATE_CLEAN_PROMPT = (
    "Remove ALL quoted text, formulas, and text-rendering instructions from the prompt. "
    "Keep ONLY the scene/background/style description. Add 'no text visible' at the end.\n\n"
    "Examples:\n"
    'Input: A classroom blackboard displays "E=mc²"\n'
    "Output: An empty classroom blackboard. No text visible.\n\n"
    "Output ONLY the cleaned prompt, nothing else."
)

OCR_PROMPT = (
    "Please read and output ALL the text content visible in this image.\n"
    "Only output the text you can see, nothing else."
)


# ============ 工具函数 ============

def _encode_image_b64(image: Image.Image) -> str:
    """PIL Image -> base64 字符串"""
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _text_accuracy(ground_truth: str, recognized: str) -> float:
    """计算文本准确度（Levenshtein 距离）"""
    import Levenshtein
    gt = ' '.join(ground_truth.lower().split())
    rec = ' '.join(recognized.lower().split())
    if not gt:
        return 1.0 if not rec else 0.0
    distance = Levenshtein.distance(gt, rec)
    return max(0.0, 1 - distance / len(gt))


def _get_grid_font() -> ImageFont.FreeTypeFont:
    """获取用于网格坐标标注的字体"""
    font_path = Path(__file__).parent.parent / "assets" / "Arial-Unicode-Bold.ttf"
    if font_path.exists():
        return ImageFont.truetype(str(font_path), 16)
    return ImageFont.load_default()


def _add_grid_overlay(image: Image.Image, grid_size: int = 6) -> Image.Image:
    """在图像上添加网格和坐标标注"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size
    font = _get_grid_font()
    
    step = 1.0 / (grid_size - 1)
    grid_color = (255, 0, 0)
    label_interval = 2

    for i in range(grid_size):
        t = i * step
        x = int(t * width)
        y = int(t * height)
        is_edge = (i == 0 or i == grid_size - 1)
        line_width = 1 if is_edge else 2
        
        draw.line([(x, 0), (x, height)], fill=grid_color, width=line_width)
        draw.line([(0, y), (width, y)], fill=grid_color, width=line_width)

    for i in range(0, grid_size, label_interval):
        t = i * step
        y = int(t * height)
        for j in range(0, grid_size, label_interval):
            s = j * step
            x = int(s * width)
            coord_text = f"({s:.1f},{t:.1f})"
            text_bbox = draw.textbbox((0, 0), coord_text, font=font)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            
            text_x = 2 if j == 0 else (width - text_w - 2 if j == grid_size - 1 else x - text_w // 2)
            text_y = 2 if i == 0 else (height - text_h - 2 if i == grid_size - 1 else y - text_h // 2)
            draw.text((text_x, text_y), coord_text, fill=grid_color, font=font)

    return img


def _extract_json_from_response(text: str) -> dict:
    """从 VLM 响应中提取 JSON 对象"""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"无法从 VLM 响应中提取 JSON:\n{text[:500]}")


# ============ VLMAgent 类 ============

class VLMAgent:
    """统一的 VLM 调用接口"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "qwen3-vl-235b-a22b-instruct",
    ):
        self._base_url = base_url or os.getenv("QST_BASE_URL")
        self._model = model

        api_key = api_key or os.getenv("QST_API_KEY")
        if not api_key:
            raise RuntimeError("QST_API_KEY not set")
        self._client = OpenAI(api_key=api_key, base_url=self._base_url)

    def _api_call(self, **call_kwargs) -> str:
        """调用 VLM API，返回响应文本"""
        resp = self._client.chat.completions.create(**call_kwargs)
        return resp.choices[0].message.content

    def call_vlm(
        self,
        system_prompt: str,
        user_content: str,
        images: Optional[list[Image.Image]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **format_kwargs,
    ) -> str:
        """通用 VLM 调用"""
        if format_kwargs:
            system_prompt = system_prompt.format(**format_kwargs)

        user_parts = [{"type": "text", "text": user_content}]
        if images:
            for img in images:
                b64 = _encode_image_b64(img)
                user_parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_parts},
        ]
        return self._api_call(
            model=self._model, messages=messages,
            stream=False, max_tokens=max_tokens, temperature=temperature,
        )

    def analyze_typography(
        self,
        image: Image.Image,
        prompt: str,
        text_contents: list[str],
    ) -> dict:
        """分析 Pass 1 参考图，自主规划文本排版
        
        Returns:
            typography_plan 字典，包含 image_analysis 和 text_regions
        """
        image_with_grid = _add_grid_overlay(image)

        contents_desc = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(text_contents))
        user_content = (
            f"Original prompt: {prompt}\n\n"
            f"Text/formula items to render:\n{contents_desc}\n\n"
            f"The red grid lines are positioning aids only."
        )

        from .formula_helper import get_font_registry
        font_names = sorted(get_font_registry().keys())
        font_list = ", ".join(font_names) if font_names else "auto only"

        raw = self.call_vlm(
            ANALYZE_TYPOGRAPHY_PROMPT,
            user_content,
            images=[image_with_grid],
            max_tokens=2048,
            temperature=0.3,
            font_list=font_list,
        )
        return _extract_json_from_response(raw)

    def generate_clean_prompt(self, prompt: str) -> str:
        """将含文字描述的 prompt 改写为不渲染任何文本的 clean 版本"""
        raw = self.call_vlm(
            GENERATE_CLEAN_PROMPT,
            f"Original prompt:\n{prompt}",
            max_tokens=512,
            temperature=0,
        )
        return raw.strip()

    def ocr_score_images(
        self,
        images: list[Image.Image],
        prompt: str,
    ) -> list[float]:
        """对每张图做 OCR 并计算文本准确度，返回 scores 列表"""
        ground_truth = _extract_text_from_prompt(prompt)

        scores = []
        for img in images:
            b64 = _encode_image_b64(img)
            recognized = self._api_call(
                model=self._model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": OCR_PROMPT},
                ]}],
                max_tokens=512,
                temperature=0.0,
            ).strip()
            
            if (recognized.startswith('"') and recognized.endswith('"')) or \
               (recognized.startswith("'") and recognized.endswith("'")):
                recognized = recognized[1:-1]

            score = _text_accuracy(ground_truth, recognized)
            scores.append(score)

        return scores
