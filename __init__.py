"""GlyphBanana: Text Rendering with Glyph Injection"""

__version__ = "1.0.0"

from .generate import generate_image
from .evaluate import VLMOCR, evaluate_image

__all__ = ["generate_image", "VLMOCR", "evaluate_image"]
