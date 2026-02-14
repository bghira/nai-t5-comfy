# NAI-T5-Comfy: NovelAI T5 for ComfyUI

**SPDX-License-Identifier: GPL-3.0-or-later**

Drop-in replacement T5 text encoder nodes for ComfyUI using NovelAI's optimized T5 implementation ([nai-t5-wrapper](https://github.com/bghira/nai-t5-wrapper)). Supports workflows that use T5-based text encoders such as FLUX, Stable Diffusion 3, PixArt, and WAN.

## Features

- Drop-in replacement for ComfyUI's built-in T5-XXL and UMT5-XXL loaders
- Uses NovelAI's fused T5 encoder with RMSNorm fusion and fused QKV projections
- Always runs in bf16 for numerical stability (T5's unscaled attention overflows fp16)
- Flex Attention enabled by default (falls back to SDPA on older PyTorch)

## Installation

Clone into your `ComfyUI/custom_nodes` directory and install dependencies:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/bghira/nai-t5-comfy
cd nai-t5-comfy
pip install -r requirements.txt
```

Restart ComfyUI.

### Dependencies

- `torch>=2.0`
- `nai-t5-wrapper>=1.1.0`
- `transformers>=4.40`
- `sentencepiece`

## Nodes

Both nodes are in the **loaders/nai** category.

| Node | Description | Use with |
|------|-------------|----------|
| **NAI T5 XXL Loader** | Loads T5-XXL text encoder | FLUX, SD3, PixArt |
| **NAI UMT5 XXL Loader** | Loads UMT5-XXL text encoder | WAN video models |

Each node takes a text encoder checkpoint file and an optional `device` setting (`default` or `cpu`).

## Technical Details

- **bf16 compute**: The model always runs in bfloat16 regardless of what dtype ComfyUI requests. T5 was trained without QK scaling (`scale_qk=False`), and the unscaled `Q*K^T` dot products overflow fp16's narrow exponent range. bf16 has the same 2-byte memory footprint but a much wider exponent range.
- **fp32 weight materialization**: Incoming HuggingFace weights are upcast to fp32 before QKV concatenation and norm fusion, then cast to bf16 as the final step. This avoids precision loss during weight transformations.
- **fp32 embeddings and final layer norm**: The vocabulary embedding and final RMSNorm stay in fp32 to match the fp32 residual stream used by `RMSNormCast`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NAI_T5_DISABLE_FLEX_ATTENTION` | `0` | Set to `1` to force SDPA instead of Flex Attention. |

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
