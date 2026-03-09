"""Qwen-VL aesthetic quality scorer."""

import re

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

TASK_PROMPT = """Rate the aesthetic quality of this image (1-5):
1: Very poor (blurry, bad exposure, messy)
2: Poor (noticeable issues)
3: Fair (decent but average)
4: Good (sharp, good composition)
5: Excellent (masterful, impactful)

Provide your analysis in <Thought> tags, then give the score in <Score> tags.
<Thought>[analysis]</Thought>
<Score>X</Score>"""


def extract_score(text: str) -> float:
    """Extract score from 1-5 and normalize to 0-1."""
    m = re.search(r"<Score>(\d+)</Score>", text)
    return float(m.group(1)) / 5.0 if m else 0.0


class QwenScorer:
    """Qwen2.5-VL based aesthetic scorer."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
        ).eval()

    def score(self, img: Image.Image, prompt: str = "") -> float:
        """Score image aesthetic quality. Prompt is ignored (for API compatibility)."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": TASK_PROMPT},
            ],
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(self.device)

        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=512)
            resp_ids = gen_ids[:, inputs["input_ids"].shape[1]:]
            resp = self.processor.batch_decode(resp_ids, skip_special_tokens=True)[0].strip()

        return extract_score(resp)
