#!/usr/bin/env python3
"""
GlyphBanana Usage Example

Generate an image of Einstein writing the quadratic formula
"""

import subprocess
import sys


def main():
    # Example 1: Generate formula image
    print("=" * 60)
    print("Example 1: Generate Einstein writing quadratic formula")
    print("=" * 60)
    
    cmd = [
        sys.executable, "generate.py",
        "--prompt", 'Albert Einstein writing the quadratic formula "x = {-b \\pm \\sqrt{b^2-4ac}} / {2a}" on a blackboard',
        "--text", "x = {-b \\pm \\sqrt{b^2-4ac}} / {2a}",
        "--output", "einstein_formula.png",
        "--seed", "42",
        "--steps", "20",
    ]
    
    print(f"Run: {' '.join(cmd)}")
    print()
    
    # Example 2: Evaluate generation result
    print("=" * 60)
    print("Example 2: Evaluate text accuracy")
    print("=" * 60)
    
    eval_cmd = [
        sys.executable, "evaluate.py",
        "--image", "einstein_formula.png",
        "--prompt", 'Albert Einstein writing the quadratic formula "x = {-b \\pm \\sqrt{b^2-4ac}} / {2a}" on a blackboard',
    ]
    
    print(f"Run: {' '.join(eval_cmd)}")
    print()
    
    # Example 3: More generation examples
    examples = [
        {
            "name": "Store Sign",
            "prompt": 'A storefront sign saying "CAFE" in neon lights',
            "text": "CAFE",
            "output": "cafe_sign.png"
        },
        {
            "name": "Book Cover",
            "prompt": 'A book cover with title "Deep Learning" in elegant typography',
            "text": "Deep Learning",
            "output": "book_cover.png"
        },
        {
            "name": "Poster",
            "prompt": 'A movie poster with text "COMING SOON" in bold letters',
            "text": "COMING SOON",
            "output": "movie_poster.png"
        },
    ]
    
    print("=" * 60)
    print("More Examples:")
    print("=" * 60)
    for ex in examples:
        print(f"\n{ex['name']}:")
        print(f"  python generate.py \\")
        print(f"      --prompt '{ex['prompt']}' \\")
        print(f"      --text '{ex['text']}' \\")
        print(f"      --output {ex['output']}")


if __name__ == "__main__":
    main()
