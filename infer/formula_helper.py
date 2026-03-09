"""
公式渲染辅助模块

渲染优先级：
1. MathJax (Node.js) — 完整 LaTeX 支持，包括 array/matrix/cases 等环境，绝对不能渲染中文！
2. matplotlib mathtext — 无需 Node.js，支持常用 LaTeX 子集
3. PIL 纯文本 — 最后兜底

支持：
- LaTeX 数学公式渲染
- Unicode 数学符号自动转换为 LaTeX
- 纯文本字体渲染（PIL）
- 自动检测并选择渲染路径
"""

import io
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# MathJax 渲染脚本路径
_MATHJAX_SCRIPT = Path(__file__).parent / "render_mathjax.js"
# 缓存 node 可用性检测结果
from typing import Optional as _Opt
_node_available: _Opt[bool] = None


# ============ LaTeX 检测 ============


def is_latex(text: str) -> bool:
    """启发式检测文本是否包含 LaTeX 公式。

    匹配条件（任一即触发）：
    - 包含 LaTeX 命令：\\frac, \\sqrt, \\int, \\sum 等
    - 被 $ 包裹
    - 包含上下标花括号组合：^{...} 或 _{...}
    - 包含 \\begin / \\end 环境
    """
    latex_patterns = [
        r'\\(?:frac|sqrt|int|oint|sum|prod|lim|infty|partial|nabla|left|right|'
        r'begin|end|alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|omega|'
        r'pi|phi|psi|chi|rho|tau|eta|zeta|xi|kappa|nu|'
        r'mathbb|mathcal|mathbf|mathrm|text|hat|bar|vec|dot|tilde|'
        r'cdot|times|div|pm|mp|leq|geq|neq|approx|equiv|sim|'
        r'rightarrow|leftarrow|Rightarrow|Leftarrow|mapsto|to)',
        r'\$.*\$',                     # $...$
        r'[_\^]\{[^}]+\}',            # ^{...} or _{...}
        r'\\begin\{',                  # \begin{...}
    ]
    return any(re.search(p, text) for p in latex_patterns)


# ============ Unicode → LaTeX 转换 ============


_UNICODE_TO_LATEX = [
    ("∫∫∫", r"\iiint"),
    ("∫∫", r"\iint"),
    ("∫", r"\int"),
    ("∑", r"\sum"),
    ("∏", r"\prod"),
    ("∞", r"\infty"),
    ("±", r"\pm"),
    ("≈", r"\approx"),
    ("≠", r"\neq"),
    ("≤", r"\leq"),
    ("≥", r"\geq"),
    ("→", r"\rightarrow"),
    ("←", r"\leftarrow"),
    ("⟨", r"\langle"),
    ("⟩", r"\rangle"),
    # Unicode 上下标数字
    ("₀", "_0"), ("₁", "_1"), ("₂", "_2"), ("₃", "_3"), ("₄", "_4"),
    ("₅", "_5"), ("₆", "_6"), ("₇", "_7"), ("₈", "_8"), ("₉", "_9"),
    ("⁰", "^0"), ("¹", "^1"), ("²", "^2"), ("³", "^3"), ("⁴", "^4"),
    ("⁵", "^5"), ("⁶", "^6"), ("⁷", "^7"), ("⁸", "^8"), ("⁹", "^9"),
    # Greek letters
    ("α", r"\alpha"),
    ("β", r"\beta"),
    ("γ", r"\gamma"),
    ("δ", r"\delta"),
    ("ε", r"\epsilon"),
    ("ζ", r"\zeta"),
    ("η", r"\eta"),
    ("θ", r"\theta"),
    ("λ", r"\lambda"),
    ("μ", r"\mu"),
    ("ν", r"\nu"),
    ("ξ", r"\xi"),
    ("π", r"\pi"),
    ("ρ", r"\rho"),
    ("σ", r"\sigma"),
    ("τ", r"\tau"),
    ("φ", r"\varphi"),
    ("ψ", r"\psi"),
    ("ω", r"\omega"),
    ("Δ", r"\Delta"),
    ("Σ", r"\Sigma"),
    ("Ω", r"\Omega"),
    # Operators / symbols
    ("∂", r"\partial"),
    ("∇", r"\nabla"),
    ("·", r"\cdot"),
    ("×", r"\times"),
    ("√", r"\sqrt"),
    ("½", r"\frac{1}{2}"),
    ("¼", r"\frac{1}{4}"),
    ("¾", r"\frac{3}{4}"),
    # Superscript letters & signs
    ("⁺", "^+"), ("⁻", "^-"), ("⁼", "^="),
    ("ⁿ", "^n"), ("ⁱ", "^i"),
    ("ᵀ", "^T"), ("ᵘ", "^u"), ("ᵛ", "^v"),
    # Subscript letters (standard Unicode subscript block)
    ("ₐ", "_a"), ("ₑ", "_e"), ("ₒ", "_o"), ("ₓ", "_x"),
    ("ₕ", "_h"), ("ₖ", "_k"), ("ₗ", "_l"), ("ₘ", "_m"),
    ("ₙ", "_n"), ("ₚ", "_p"), ("ₛ", "_s"), ("ₜ", "_t"),
    # Subscript letters (modifier letter / phonetic extensions block)
    ("ᵢ", "_i"), ("ⱼ", "_j"), ("ᵣ", "_r"), ("ᵤ", "_u"),
    ("ᵥ", "_v"), ("ᵧ", "_y"),
    # Subscript signs (plain char, grouping step will wrap in braces)
    ("₊", "+"), ("₋", "-"), ("₌", "="),
    # Greek capitals
    ("Γ", r"\Gamma"),
    # Arrows / relations
    ("⇌", r"\rightleftharpoons"),
    ("⇒", r"\Rightarrow"),
    ("↔", r"\leftrightarrow"),
    ("↑", r"\uparrow"),
    # Integrals / contour
    ("∬", r"\iint"),
    ("∮", r"\oint"),
    # Misc symbols
    ("□", r"\Box"),
    ("Ĥ", r"\hat{H}"),
    # Special
    ("ℏ", r"\hbar"),
    ("ħ", r"\hbar"),
]

# Combining diacritical marks: character + combining mark → \cmd{character}
_COMBINING_TO_LATEX = {
    "\u20D7": r"\vec",      # COMBINING RIGHT ARROW ABOVE  (B⃗ → \vec{B})
    "\u0302": r"\hat",      # COMBINING CIRCUMFLEX ACCENT  (x̂ → \hat{x})
    "\u0303": r"\tilde",    # COMBINING TILDE              (x̃ → \tilde{x})
    "\u0304": r"\bar",      # COMBINING MACRON             (x̄ → \bar{x})
    "\u0307": r"\dot",      # COMBINING DOT ABOVE          (ẋ → \dot{x})
    "\u0308": r"\ddot",     # COMBINING DIAERESIS          (ẍ → \ddot{x})
}


def _check_node() -> bool:
    """检测 Node.js 是否可用（结果缓存）。
    
    如果 PATH 中找不到 node，会探测常见安装路径并自动补充 PATH。
    这解决了 conda activate 重置 PATH 导致 ~/.local/bin/node 丢失的问题。
    """
    global _node_available
    if _node_available is not None:
        return _node_available

    if shutil.which("node") is not None:
        _node_available = True
        return True

    # PATH 中找不到，探测常见安装位置
    home = os.path.expanduser("~")
    fallback_dirs = [
        os.path.join(home, ".local", "bin"),
        os.path.join(home, ".nvm", "current", "bin"),
        "/usr/local/bin",
    ]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        fallback_dirs.insert(0, os.path.join(conda_prefix, "bin"))

    for d in fallback_dirs:
        node_path = os.path.join(d, "node")
        if os.path.isfile(node_path) and os.access(node_path, os.X_OK):
            os.environ["PATH"] = d + ":" + os.environ.get("PATH", "")
            print(f"[MathJax] 在 {d} 找到 node，已补充到 PATH")
            _node_available = True
            return True

    _node_available = False
    return False


def _fix_sqrt_parens(text: str) -> str:
    """将 \\sqrt(...) 修正为 \\sqrt{...}（matplotlib 要求花括号）。"""
    # 匹配 \sqrt( 并找到对应的闭括号，替换为花括号
    result = []
    i = 0
    pat = r"\sqrt"
    while i < len(text):
        if text[i:i+5] == pat + "(" :
            result.append(pat + "{")
            depth = 1
            j = i + 6
            while j < len(text) and depth > 0:
                if text[j] == "(":
                    depth += 1
                    result.append("(")
                elif text[j] == ")":
                    depth -= 1
                    if depth == 0:
                        result.append("}")
                    else:
                        result.append(")")
                else:
                    result.append(text[j])
                j += 1
            i = j
        else:
            # 也处理单独的 √( 未被转换的情况
            if text[i] == "√" and i + 1 < len(text) and text[i+1] == "(":
                result.append(r"\sqrt{")
                depth = 1
                j = i + 2
                while j < len(text) and depth > 0:
                    if text[j] == "(":
                        depth += 1
                        result.append("(")
                    elif text[j] == ")":
                        depth -= 1
                        if depth == 0:
                            result.append("}")
                        else:
                            result.append(")")
                    else:
                        result.append(text[j])
                    j += 1
                i = j
            else:
                result.append(text[i])
                i += 1
    return "".join(result)


def plaintext_to_latex(text: str) -> str:
    """将含 Unicode 数学符号的 plaintext 转换为 LaTeX 命令。"""
    result = text

    # 1) Combining diacriticals: X⃗ → \vec{X}
    for combining, cmd in _COMBINING_TO_LATEX.items():
        result = re.sub(
            rf'(\w){re.escape(combining)}',
            lambda m, c=cmd: f'{c}{{{m.group(1)}}}',
            result,
        )

    # 2) √(...) → \sqrt{...}
    result = _fix_sqrt_parens(result)

    # 3) 逐符号替换（\command 后追加空格防止与后续字母粘连，math mode 中多余空格无影响）
    for old, new in _UNICODE_TO_LATEX:
        if new.startswith('\\') and new[-1:].isalpha():
            result = result.replace(old, new + ' ')
        else:
            result = result.replace(old, new)

    # 4) 再修一次 \sqrt(...)
    result = _fix_sqrt_parens(result)

    # 5) 合并相邻的下标/上标: _v-_1 → _{v-1}, ^-^1 → ^{-1}, _1_2 → _{12}
    result = _group_scripts(result)

    return result


def _group_scripts(text: str) -> str:
    """合并相邻的下标/上标到花括号组。

    例:  _v-_1  → _{v-1}     (Bessel 阶)
         _n=_1  → _{n=1}     (求和下界)
         _1_2   → _{12}      (矩阵下标)
         ^-^1   → ^{-1}      (逆矩阵)
    """
    def _is_script_char(ch: str) -> bool:
        return ch.isalnum() or ch in '+-='

    parts: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if (i < n - 1 and text[i] in ('_', '^')
                and _is_script_char(text[i + 1])):
            marker = text[i]
            group = [text[i + 1]]
            j = i + 2
            while j < n:
                # connector (+-=) followed by same marker + char
                if (j + 2 < n and text[j] in '-+='
                        and text[j + 1] == marker
                        and _is_script_char(text[j + 2])):
                    group.append(text[j])
                    group.append(text[j + 2])
                    j += 3
                # same marker + char (no connector)
                elif (j + 1 < n and text[j] == marker
                      and _is_script_char(text[j + 1])):
                    group.append(text[j + 1])
                    j += 2
                else:
                    break
            if len(group) > 1:
                parts.append(f'{marker}{{{"".join(group)}}}')
            else:
                parts.append(marker + group[0])
            i = j
        else:
            parts.append(text[i])
            i += 1
    return ''.join(parts)


# ============ LaTeX 渲染 ============


def render_mathjax(
    latex: str,
    width: int,
    height: int,
    text_color: str = "black",
    font_weight: str = "regular",
) -> _Opt[Image.Image]:
    """使用 MathJax (Node.js) 渲染 LaTeX → SVG → PIL Image。

    支持完整的 LaTeX 语法（array, matrix, cases 等环境）。
    需要 Node.js 和 cairosvg（或 Pillow SVG 支持）。

    Args:
        font_weight: 字体粗细 ("light"/"regular"/"bold")，bold 时自动使用 \boldsymbol

    Returns:
        PIL Image，如果 Node.js 不可用或渲染失败则返回 None。
    """
    if not _check_node():
        print("[MathJax] 回退: Node.js 不可用 (shutil.which('node') 为空)")
        return None
    if not _MATHJAX_SCRIPT.exists():
        print(f"[MathJax] 回退: 渲染脚本不存在 {_MATHJAX_SCRIPT}")
        return None

    # 去掉 $ 包裹（MathJax 自己处理）
    formula = latex.strip().strip("$")
    
    # bold 时自动包裹 \boldsymbol
    if font_weight == "bold":
        formula = rf"\boldsymbol{{{formula}}}"

    # 调用 Node.js 渲染 SVG
    try:
        result = subprocess.run(
            ["node", str(_MATHJAX_SCRIPT), formula],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            print(f"[MathJax] node 错误: {result.stderr.strip()}")
            return None
        svg_str = result.stdout
        if not svg_str or "<svg" not in svg_str:
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[MathJax] 调用失败: {e}")
        return None

    # SVG → PNG：使用 cairosvg 转换
    img = _convert_svg_to_png(svg_str, width, height, text_color)
    
    if img is None:
        print("[MathJax] SVG→PNG 转换失败")
        return None

    return img


def _convert_svg_to_png(svg_str: str, width: int, height: int,
                        text_color: str = "black") -> _Opt[Image.Image]:
    """将 MathJax 生成的 SVG 转换为 PNG。
    
    MathJax 输出包裹在 <mjx-container> 中，需要提取内部 SVG 并添加样式。
    """
    try:
        from xml.etree import ElementTree as ET
        import re
        
        # 解析 XML
        root = ET.fromstring(svg_str)
        
        # 查找内部 SVG 元素（MathJax 包裹在 mjx-container 中）
        svg_element = None
        if 'mjx-container' in root.tag:
            for child in root.iter():
                if child.tag.endswith('svg') or child.tag == 'svg':
                    svg_element = child
                    break
        
        if svg_element is None:
            print("[MathJax] 无法在 mjx-container 中找到 SVG 元素")
            return None
        
        # 获取原始尺寸（用于计算宽高比）
        orig_width = svg_element.get('width', '')
        orig_height = svg_element.get('height', '')
        
        # 解析尺寸（支持 ex 单位，1ex ≈ 8px）
        def parse_size(s):
            if not s:
                return None
            s = s.strip()
            if s.endswith('ex'):
                return float(s[:-2]) * 8
            try:
                return float(s)
            except:
                return None
        
        orig_w = parse_size(orig_width)
        orig_h = parse_size(orig_height)
        
        # 计算保持宽高比的新尺寸
        if orig_w and orig_h:
            scale_w = width / orig_w
            scale_h = height / orig_h
            scale = min(scale_w, scale_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
        else:
            new_w, new_h = width, height
        
        # 修改 SVG 属性
        svg_element.set('width', str(new_w))
        svg_element.set('height', str(new_h))
        
        # 构建 style：只设置文字颜色，背景透明
        existing_style = svg_element.get('style', '')
        # 移除 MathJax 的 vertical-align 和 background-color
        style_parts = []
        if existing_style:
            for part in existing_style.split(';'):
                part = part.strip()
                # 过滤掉 vertical-align 和 background-color
                if part and not part.startswith(('vertical-align', 'background-color')):
                    style_parts.append(part)
        
        # MathJax 使用 currentColor，设置 color 即可改变文字颜色
        # 默认黑色文字，透明背景
        svg_element.set('style', f"color: {text_color or 'black'}")
        
        # 序列化修改后的 SVG
        svg_bytes = ET.tostring(svg_element, encoding='utf-8')
        
        # 使用 cairosvg 转换为 PNG（保留透明背景）
        import cairosvg
        png_bytes = cairosvg.svg2png(
            bytestring=svg_bytes,
            output_width=new_w,
            output_height=new_h,
        )
        
        # 使用 RGBA 保留透明通道
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        return img
        
    except Exception as e:
        print(f"[MathJax] SVG 转换失败: {e}")
        return None


def render_latex(
    latex: str,
    width: int,
    height: int,
    text_color: str = "black",
    font_weight: str = "regular",
) -> Image.Image:
    """使用 matplotlib 渲染 LaTeX 公式为 PIL Image。

    支持完整的 LaTeX math mode 语法（分数、积分、矩阵、上下标等）。
    自动二分搜索字号以填满给定区域。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 确保被 $ 包裹
    # 注意：不用 \boldsymbol 包裹整个公式——matplotlib 的 bold math font
    # 缺少 \cdot, \times 等运算符 glyph，会产生 dummy 替换。
    # MathJax 路径（render_mathjax）已正确处理 bold。
    formula = latex.strip()
    if not formula.startswith("$"):
        formula = f"${formula}$"

    fg = text_color
    bg = "black"

    # 自适应字号：二分搜索
    dpi = 150
    best_fontsize = 12
    lo, hi = 8, 200

    for _ in range(15):
        mid = (lo + hi) // 2
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_facecolor(bg)
        try:
            fig.text(
                0.5, 0.5, formula,
                fontsize=mid, color=fg,
                ha="center", va="center",
                math_fontfamily="stix",
            )
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            texts = fig.texts
            if texts:
                bb = texts[0].get_window_extent(renderer)
                tw, th = bb.width, bb.height
                if tw <= width * 0.95 and th <= height * 0.9:
                    best_fontsize = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            else:
                hi = mid - 1
        except Exception:
            hi = mid - 1
        finally:
            plt.close(fig)

    # 最终渲染
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor(bg)
    fig.text(
        0.5, 0.5, formula,
        fontsize=best_fontsize, color=fg,
        ha="center", va="center",
        math_fontfamily="stix",
    )

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, facecolor=bg, bbox_inches="tight", pad_inches=0.05)
    except (ValueError, RuntimeError) as e:
        plt.close(fig)
        print(f"[formula_helper] matplotlib 渲染失败，降级为纯文本: {e}")
        plain = re.sub(r'\\[a-zA-Z]+', ' ', latex.strip().strip("$"))
        plain = re.sub(r'[{}^_]', '', plain).strip()
        fallback_color = text_color if text_color != "black" else "white"
        math_font = _BUNDLED_MATH_FONT if os.path.exists(_BUNDLED_MATH_FONT) else None
        return render_plaintext(plain, width, height, fallback_color,
                                font_weight=font_weight, font_path=math_font)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)

    return img


# ============ 纯文本字体渲染 ============


# 项目自带字体（优先级最高，保证跨平台可用）
_BUNDLED_FONT_DIR = str(Path(__file__).resolve().parent.parent / "assets")
_BUNDLED_CJK_FONT = os.path.join(_BUNDLED_FONT_DIR, "Arial-Unicode-Bold.ttf")
_BUNDLED_MATH_FONT = os.path.join(_BUNDLED_FONT_DIR, "cambria.ttc")

# 服务器上已知的 CJK 字体
_SERVER_CJK_FONT = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/baselines/anytext/font/Arial_Unicode.ttf"

_FONT_SEARCH_PATHS = {
    # weight -> [路径列表]，优先级从上到下
    "bold": [
        # 项目自带 / 服务器
        _BUNDLED_CJK_FONT,
        _SERVER_CJK_FONT,
        # macOS
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica Bold.ttc",
        "/System/Library/Fonts/HelveticaNeue Bold.ttc",
        # Linux
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ],
    "light": [
        _BUNDLED_CJK_FONT,
        _SERVER_CJK_FONT,
        # macOS
        "/System/Library/Fonts/HelveticaNeue Light.ttc",
        "/System/Library/Fonts/Helvetica Light.ttc",
        # Linux
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Light.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf",
    ],
    "regular": [
        # 项目自带 / 服务器（CJK 优先）
        _BUNDLED_CJK_FONT,
        _SERVER_CJK_FONT,
        # macOS
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        # Linux (CJK first, then fallback)
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ],
}


def get_available_font(
    size: int = 100,
    weight: str = "regular",
    font_path: _Opt[str] = None,
) -> ImageFont.FreeTypeFont:
    """获取系统中可用的字体。

    Args:
        size: 字号
        weight: 字体粗细 ("light"/"regular"/"bold")
        font_path: 显式指定字体路径（优先级最高）
    """
    # 优先使用显式路径
    if font_path and os.path.exists(font_path):
        return ImageFont.truetype(font_path, size)

    # 按 weight 查找
    candidates = _FONT_SEARCH_PATHS.get(weight, [])
    # 找不到指定 weight 时回退到 regular
    if not candidates:
        candidates = _FONT_SEARCH_PATHS["regular"]
    # 再追加 regular 作为兜底
    if weight != "regular":
        candidates = candidates + _FONT_SEARCH_PATHS["regular"]

    for fp in candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    print("警告：使用默认字体")
    return ImageFont.load_default()


def calculate_font_size(
    text: str, bbox_width: int, bbox_height: int,
    weight: str = "regular", font_path: _Opt[str] = None,
) -> int:
    """计算能填满 bbox 的字体大小（使用实际渲染字体测量，避免度量偏差）。"""
    estimated_size = int(bbox_height * 0.8)
    font = get_available_font(estimated_size, weight=weight, font_path=font_path)

    test_img = Image.new("RGB", (bbox_width * 2, bbox_height * 2), "white")
    draw = ImageDraw.Draw(test_img)

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    scale_w = bbox_width / max(text_width, 1)
    scale_h = bbox_height / max(text_height, 1)
    scale = min(scale_w, scale_h) * 0.9

    return max(int(estimated_size * scale), 12)


def render_plaintext(
    text: str,
    width: int,
    height: int,
    text_color: str = "black",
    font_weight: str = "regular",
    font_path: _Opt[str] = None,
) -> Image.Image:
    """使用 PIL + 系统字体渲染纯文本。"""
    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)

    font_size = calculate_font_size(text, width, height, weight=font_weight, font_path=font_path)
    font = get_available_font(font_size, weight=font_weight, font_path=font_path)

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # 补偿 textbbox 相对于锚点的偏移（ascender/descender/bearing），否则文字会被裁切
    x = (width - text_width) // 2 - text_bbox[0]
    y = (height - text_height) // 2 - text_bbox[1]

    draw.text((x, y), text, fill=text_color, font=font)
    return img


# ============ 工具函数 ============


def _composite_to_canvas(
    img: Image.Image,
    width: int,
    height: int,
) -> Image.Image:
    """将任意尺寸的图像（可能含透明通道）居中合成到 (width, height) 的黑色 RGB 画布上。

    如果图像大于目标尺寸，先按比例缩小。
    """
    if img.size == (width, height) and img.mode == "RGB":
        return img

    iw, ih = img.size
    scale = min(width / max(iw, 1), height / max(ih, 1), 1.0)
    if scale < 1.0:
        iw, ih = int(iw * scale), int(ih * scale)
        img = img.resize((iw, ih), Image.LANCZOS)

    canvas = Image.new("RGB", (width, height), "black")
    # 居中粘贴
    x = (width - iw) // 2
    y = (height - ih) // 2
    if img.mode == "RGBA":
        canvas.paste(img, (x, y), mask=img)
    else:
        canvas.paste(img.convert("RGB"), (x, y))
    return canvas


# ============ 长文本自动换行 ============

# 可在此处断行的分隔符 pattern（在匹配位置之后插入 \n）
_BREAK_PATTERNS = [
    r'\\rightarrow\s?',
    r'\\Rightarrow\s?',
    r'\\approx\s?',
    r'→',
    r'⇒',
    r'≈',
    r'(?<!\\)=(?!=)',
]

def _visual_len(text: str) -> int:
    """估算 LaTeX 字符串的视觉字符宽度（去除标记后的长度）。"""
    s = re.sub(r'\\[a-zA-Z]+', 'X', text)
    s = re.sub(r'[{}^_$]', '', s)
    return len(s)


def _auto_linebreak(text: str, min_visual_len: int = 20) -> str:
    """当文本视觉长度较长时，在特定符号后插入 \\n 换行（统一标记，渲染层处理）。

    使用视觉长度（去除 LaTeX 标记）判断，避免短公式被误换行。
    只在第一个匹配位置断一次（拆成两行）。
    """
    if _visual_len(text) < min_visual_len:
        return text

    for pat in _BREAK_PATTERNS:
        m = re.search(pat, text)
        if m:
            pos = m.end()
            if pos < len(text) * 0.85 and pos > len(text) * 0.15:
                return text[:pos] + '\n' + text[pos:]

    return text


# ============ 统一入口 ============


def _render_single_line(
    text: str, width: int, height: int,
    text_color: str, use_latex: bool,
    font_weight: str, font_path: _Opt[str],
) -> Image.Image:
    """渲染单行文本/公式（黑底），内部按优先级选择渲染器。"""
    if use_latex:
        img = render_mathjax(text, width, height, text_color, font_weight)
        if img is not None:
            return _composite_to_canvas(img, width, height)
        print(f"[formula_helper] MathJax 不可用，回退 matplotlib: {text[:60]}")
        return render_latex(text, width, height, text_color,
                            font_weight=font_weight)
    return render_plaintext(text, width, height, text_color,
                            font_weight=font_weight, font_path=font_path)


def render_formula(
    text: str,
    width: int,
    height: int,
    text_color: str = "black",
    force_latex: bool = False,
    font_weight: str = "regular",
    font_path: _Opt[str] = None,
    rotation: float = 0.0,
) -> Image.Image:
    """渲染公式/文本图像（黑底 + 指定文字颜色）。

    优先级：
    1. MathJax (Node.js) — 完整 LaTeX 支持（array, matrix, cases 等）
    2. matplotlib mathtext — 无需 Node.js，支持常用 LaTeX 子集
    3. PIL 纯文本 — 最后兜底

    支持 \\n 多行：逐行渲染后垂直拼接。
    """
    converted = plaintext_to_latex(text)
    use_latex = force_latex or is_latex(text) or (converted != text)
    if converted != text:
        text = converted

    text = _auto_linebreak(text)

    lines = text.split('\n')
    if len(lines) > 1:
        line_h = height // len(lines)
        parts = [
            _render_single_line(
                line.strip(), width, line_h,
                text_color, use_latex, font_weight, font_path,
            )
            for line in lines
        ]
        canvas = Image.new("RGB", (width, height), "black")
        for i, part in enumerate(parts):
            if part.size != (width, line_h):
                part = part.resize((width, line_h), Image.LANCZOS)
            canvas.paste(part, (0, i * line_h))
        img = canvas
    else:
        img = _render_single_line(
            text, width, height,
            text_color, use_latex, font_weight, font_path,
        )

    if rotation != 0:
        img = img.rotate(rotation, expand=False, resample=Image.BICUBIC,
                         fillcolor=(0, 0, 0))

    return img


# ============ 测试入口 ============


if __name__ == "__main__":
    from pathlib import Path

    out_dir = Path("output/formula_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    W, H = 512, 192

    test_cases = [
        # (名称, 公式文本, 是否强制latex)
        ("e2",
         r"$E^2=(pc)^2+(m_0c^2)^2$",
         True),
    ]

    print(f"渲染 {len(test_cases)} 个测试用例到 {out_dir}")
    print(f"图像尺寸: {W}x{H}")
    print("=" * 60)

    for name, text, force in test_cases:
        detected = "latex" if (force or is_latex(text)) else "auto"
        converted = plaintext_to_latex(text)
        if not force and not is_latex(text) and converted != text:
            detected = "unicode→latex"

        print(f"\n[{name}] 检测: {detected}")
        print(f"  输入: {text[:80]}{'...' if len(text) > 80 else ''}")

        img = render_formula(text, W, H, force_latex=force)
        path = out_dir / f"{name}.png"
        img.save(path)
        print(f"  保存: {path}")

    print(f"\n{'=' * 60}")
    print(f"全部完成！共 {len(test_cases)} 张图，保存在 {out_dir}")


# ============ 字体库（从 AnyText lang_font_dict 加载） ============

import numpy as np

_ANYTEXT_FONT_BASE = Path(__file__).resolve().parent.parent / "baselines" / "anytext"
_LANG_FONT_CACHE: _Opt[dict] = None


def _detect_text_lang(text: str) -> str:
    """简易语言检测：遍历字符的 Unicode 范围。"""
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            return "ch_sim_char"
        if 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:
            return "ja"
        if 0xAC00 <= cp <= 0xD7AF:
            return "ko"
        if 0x0900 <= cp <= 0x097F:
            return "hi"
    return "en"


def get_font_candidates(text: str, n: int = 3) -> list[str]:
    """根据文本语言从 AnyText 字体库中返回最多 n 个可用字体路径。

    字体按 lang_font_dict.npy 中的覆盖率降序排列，优先返回覆盖最全的。
    """
    global _LANG_FONT_CACHE
    if _LANG_FONT_CACHE is None:
        dict_path = _ANYTEXT_FONT_BASE / "font" / "lang_font_dict.npy"
        if not dict_path.exists():
            return []
        _LANG_FONT_CACHE = np.load(str(dict_path), allow_pickle=True).item()

    lang = _detect_text_lang(text)
    entry = _LANG_FONT_CACHE.get(lang, _LANG_FONT_CACHE.get("en", {}))
    raw_fonts = entry.get("fonts", [])

    resolved = []
    seen = set()
    for f in raw_fonts:
        p = _ANYTEXT_FONT_BASE / f
        ps = str(p)
        if p.exists() and ps not in seen:
            seen.add(ps)
            resolved.append(ps)
        if len(resolved) >= n:
            break
    return resolved


# ============ 字体注册表 ============


_FONT_REGISTRY: _Opt[dict[str, str]] = None

_FONT_SCAN_DIRS = [
    Path(_BUNDLED_FONT_DIR),
    _ANYTEXT_FONT_BASE / "font" / "lang_font",
    _ANYTEXT_FONT_BASE / "font" / "fontlib" / "googlefont",
    _ANYTEXT_FONT_BASE / "font" / "fontlib" / "wordart",
]


def get_font_registry() -> dict[str, str]:
    """返回 {字体名: 路径} 注册表（惰性构建，扫描所有已知字体目录）。"""
    global _FONT_REGISTRY
    if _FONT_REGISTRY is None:
        registry = {}
        for d in _FONT_SCAN_DIRS:
            if not d.exists():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in (".ttf", ".ttc", ".otf") and f.stem not in registry:
                    registry[f.stem] = str(f)
        _FONT_REGISTRY = registry
    return _FONT_REGISTRY


def resolve_font_name(name: _Opt[str]) -> _Opt[str]:
    """将字体名解析为路径。返回 None 表示使用默认字体。"""
    if not name or name == "auto":
        return None
    return get_font_registry().get(name)
