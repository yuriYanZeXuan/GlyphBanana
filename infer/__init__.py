"""
GlyphBanana 推理模块

包含核心接口:
- VLMAgent: 统一的 VLM 调用中心（排版分析、OCR评分）
- GlyphInjector: 文字渲染和 latent 注入
- AttentionEnhancement: 注意力增强
"""

from .VLM_agent import VLMAgent
from .glyph_injector import GlyphInjector, create_glyph_injector
from .attn_enhancement import AttentionEnhancement

__all__ = [
    "VLMAgent",
    "GlyphInjector",
    "create_glyph_injector",
    "AttentionEnhancement",
]
