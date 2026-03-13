# GlyphBanana

GlyphBanana 是一个面向图像文字渲染的三阶段推理项目，支持两种主生成后端：

- `zimage`
- `qwen`

并支持可选的 `FluxKlein` refinement，用于在 Pass 3 对文字和背景做进一步风格融合。

## 当前流程

生成流程固定为三段：

1. Pass 1: 用完整 prompt 生成参考图
2. VLM Planning: 分析参考图，自动规划文字区域和版式
3. Pass 2: 用 clean prompt + glyph injection 生成带字结果
4. Pass 3: 用 FluxKlein 生成多个 refinement 候选，并由 VLM 选优

如果传了 `--no-harmonize`，就会跳过 Pass 3，最终结果直接使用 Pass 2 输出。

## 安装

```bash
git clone <repo-url>
cd GlyphBanana
pip install -r requirements.txt
```

配置 VLM API：

```bash
export QST_BASE_URL="https://your-api-endpoint.com/v1"
export QST_API_KEY="your-api-key"
```

## 快速开始

### 生成

`zimage`：

```bash
python3 generate.py \
  --backend zimage \
  --prompt 'A storefront sign saying "CAFE"' \
  --text "CAFE" \
  --output output_zimage.png
```

`qwen`：

```bash
python3 generate.py \
  --backend qwen \
  --prompt 'A storefront sign saying "CAFE"' \
  --text "CAFE" \
  --output output_qwen.png
```

### 冒烟测试

默认会分别测试两个 backend，并保留 harmonization：

```bash
bash scripts/test_backends.sh
```

只测单个 backend：

```bash
bash scripts/test_backends.sh zimage
bash scripts/test_backends.sh qwen
```

输出目录默认是 `output/backend_smoke_test/`。

## 候选图与 VLM 选优

当没有传 `--no-harmonize` 时，会进入 Pass 3：

1. `klein_single_masked`
2. `klein_nomask`
3. `klein_dual`

这 3 张候选图会先被拼接保存为：

- `<output>_pass3_variants.png`

例如如果输出是 `output/result.png`，则候选拼接图是：

- `output/result_pass3_variants.png`

之后会调用 `vlm_agent.select_best_image(...)` 在这 3 张图中选择最终结果。

如果传了 `--no-harmonize`，则不会生成这张候选拼接图。

## 生成参数

```bash
python3 generate.py \
  --backend zimage \
  --prompt "A clean poster with the word \"HELLO\" centered" \
  --text "HELLO" \
  --output output.png \
  --steps 20 \
  --seed 42 \
  --height 1024 \
  --width 1024
```

常用参数：

- `--backend {zimage,qwen}`: 选择主生成后端
- `--prompt`: 场景 prompt，建议把待渲染文字写进引号里
- `--text`: 实际要渲染的文字内容，可传多个
- `--output`: 最终输出图路径
- `--no-harmonize`: 跳过 FluxKlein Pass 3
- `--qwen-true-cfg-scale`: Qwen backend 的 true CFG scale
- `--qwen-guidance-scale`: Qwen distilled guidance scale
- `--klein-model-path`: FluxKlein 权重路径
- `--klein-steps`: FluxKlein refinement 步数
- `--klein-guidance`: FluxKlein guidance

## 测评逻辑

`evaluate.py` 现在支持两种方式提供期望文本，和生成逻辑可以对齐：

1. 用 `--prompt`
2. 直接用 `--text`

推荐：

- 如果你生成时的 prompt 本身包含引号中的目标文字，可以直接传同一个 `--prompt`
- 如果你更想显式对齐 `generate.py --text`，就直接在评测时传 `--text`

### 单图测评

用 prompt：

```bash
python3 evaluate.py \
  --image output.png \
  --prompt 'A storefront sign saying "CAFE"'
```

用 text：

```bash
python3 evaluate.py \
  --image output.png \
  --text "CAFE"
```

### 批量测评

```bash
python3 evaluate.py \
  --image_dir outputs/ \
  --prompt_file prompts.json \
  --output results.json
```

`prompt_file` 可以是：

```json
{
  "image1.png": "A sign saying \"Hello World\"",
  "image2.png": "SALE 50% OFF"
}
```

也可以是 list，其中每项使用 `prompt` 或 `text` 字段。

## Python 接口

项目顶层导出了这两个接口：

```python
from GlyphBanana import generate_image, evaluate_image
```

示例：

```python
from GlyphBanana import generate_image, evaluate_image

image = generate_image(
    backend="zimage",
    prompt='A poster with the word "HELLO"',
    text=["HELLO"],
    output="output.png",
)

result = evaluate_image(image, text="HELLO")
print(result["accuracy"])
```

## 当前目录结构

```text
GlyphBanana/
├── generate.py
├── evaluate.py
├── __init__.py
├── scripts/
│   └── test_backends.sh
├── infer/
│   ├── VLM_agent.py
│   ├── glyph_injector.py
│   ├── formula_helper.py
│   └── attn_enhancement.py
├── models/
│   ├── zimage_ip/
│   ├── qwen_ip/
│   └── fluxklein/
└── output/
```

## 权重与依赖

需要准备：

- Z-Image 权重
- Qwen-Image 权重
- FluxKlein 权重
- 可用的 VLM API（用于版式规划、OCR、候选选优）

默认权重路径写在 `generate.py` 里，可通过命令行覆盖。
