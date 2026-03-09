#!/usr/bin/env python3
"""
GlyphBanana 使用示例

生成爱因斯坦写二次方程求根公式的图像
"""

import subprocess
import sys


def main():
    # 示例 1: 生成公式图像
    print("=" * 60)
    print("示例 1: 生成爱因斯坦写二次方程求根公式")
    print("=" * 60)
    
    cmd = [
        sys.executable, "generate.py",
        "--prompt", 'Albert Einstein writing the quadratic formula "x = {-b \\pm \\sqrt{b^2-4ac}} / {2a}" on a blackboard',
        "--text", "x = {-b \\pm \\sqrt{b^2-4ac}} / {2a}",
        "--output", "einstein_formula.png",
        "--seed", "42",
        "--steps", "20",
    ]
    
    print(f"运行: {' '.join(cmd)}")
    print()
    
    # 示例 2: 评估生成结果
    print("=" * 60)
    print("示例 2: 评估文本准确度")
    print("=" * 60)
    
    eval_cmd = [
        sys.executable, "evaluate.py",
        "--image", "einstein_formula.png",
        "--prompt", 'Albert Einstein writing the quadratic formula "x = {-b \\pm \\sqrt{b^2-4ac}} / {2a}" on a blackboard',
    ]
    
    print(f"运行: {' '.join(eval_cmd)}")
    print()
    
    # 示例 3: 更多生成示例
    examples = [
        {
            "name": "商店招牌",
            "prompt": 'A storefront sign saying "CAFE" in neon lights',
            "text": "CAFE",
            "output": "cafe_sign.png"
        },
        {
            "name": "书籍封面",
            "prompt": 'A book cover with title "Deep Learning" in elegant typography',
            "text": "Deep Learning",
            "output": "book_cover.png"
        },
        {
            "name": "海报",
            "prompt": 'A movie poster with text "COMING SOON" in bold letters',
            "text": "COMING SOON",
            "output": "movie_poster.png"
        },
    ]
    
    print("=" * 60)
    print("更多示例:")
    print("=" * 60)
    for ex in examples:
        print(f"\n{ex['name']}:")
        print(f"  python generate.py \\")
        print(f"      --prompt '{ex['prompt']}' \\")
        print(f"      --text '{ex['text']}' \\")
        print(f"      --output {ex['output']}")


if __name__ == "__main__":
    main()
