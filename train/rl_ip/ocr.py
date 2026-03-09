"""OCR scoring using PaddleOCR."""

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR


class OCRScorer:
    """PaddleOCR-based text recognition scorer."""

    def __init__(self):
        # Disable unnecessary modules for speed
        self.model = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def score(self, img: Image.Image) -> tuple[str, float]:
        """Recognize text and return (text, confidence)."""
        arr = np.array(img.convert("RGB"))
        result = self.model.predict(input=arr)

        if not result or not result[0]:
            return "", 0.0

        data = result[0].json.get("res", {})
        texts = data.get("rec_texts", [])
        confs = data.get("rec_scores", [])

        if not texts or not confs:
            return "", 0.0

        return " ".join(texts), float(np.mean(confs))


def main():
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (200, 50), color="black")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    draw.text((10, 10), "Hello World", fill="white", font=font)

    scorer = OCRScorer()
    text, conf = scorer.score(img)
    print(f"Text: '{text}', Confidence: {conf:.3f}")


if __name__ == "__main__":
    main()
