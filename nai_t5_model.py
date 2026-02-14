# NAI-T5-Comfy: ComfyUI custom node for NovelAI's T5 with Flex Attention
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file provides a wrapper that makes nai-t5-wrapper compatible with
# ComfyUI's text encoder interface.

import os
import torch
import torch.nn as nn
from typing import Optional, Any
import logging

from nai_t5_wrapper.t5_encoder import T5EncoderStack, ActAndResidual, SDPAArgs, FlexArgs
from nai_t5_wrapper.t5_hf import to_based_config, hf_to_based_t5_enc_state
from nai_t5_wrapper.t5_common import T5Config, T5AttnImpl
from nai_t5_wrapper.fuse_norm_scales import fuse_norm_scales_enc

logger = logging.getLogger(__name__)


def check_flex_attention_support() -> bool:
    """Check if PyTorch version supports Flex Attention.

    Flex Attention is always enabled unless the import fails (old PyTorch)
    or explicitly disabled via NAI_T5_DISABLE_FLEX_ATTENTION=1.
    """
    if os.environ.get("NAI_T5_DISABLE_FLEX_ATTENTION", "").lower() in ("1", "true", "yes"):
        logger.info("Flex Attention disabled via NAI_T5_DISABLE_FLEX_ATTENTION environment variable")
        return False
    try:
        from torch.nn.attention.flex_attention import flex_attention
        return True
    except ImportError:
        logger.warning("Flex Attention not available (requires PyTorch >= 2.5). Falling back to SDPA.")
        return False


def is_hip() -> bool:
    """Check if running on HIP/ROCm (AMD GPU)."""
    return hasattr(torch.version, 'hip') and torch.version.hip is not None


class NAIT5Embeddings(nn.Module):
    """
    Wrapper around the NAI-T5 embedding layer to provide the interface
    ComfyUI expects (with out_dtype parameter).
    """

    def __init__(self, embed: nn.Module):
        super().__init__()
        self.weight = embed.weight
        self._embed = embed

    def forward(self, input_ids: torch.Tensor, out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Get embeddings with optional dtype casting."""
        embeds = self._embed(input_ids)
        if out_dtype is not None:
            embeds = embeds.to(out_dtype)
        return embeds


class NAIT5ForComfy(nn.Module):
    """
    Wrapper around NAI-T5 encoder that provides the interface expected by
    ComfyUI's SDClipModel/ClipTokenWeightEncoder.
    """

    def __init__(
        self,
        config_dict: dict,
        dtype: Optional[torch.dtype] = None,
        device: str = "cpu",
        operations=None,  # Unused, for ComfyUI compatibility
    ):
        super().__init__()

        self.config_dict = config_dict
        # Always use bf16: T5's unscaled attention (scale_qk=False) overflows fp16's
        # narrow exponent range, and bf16 has the same memory footprint.
        self.dtype = torch.bfloat16
        self.device = device
        self.num_layers = config_dict.get("num_layers", 24)
        self.model_type = config_dict.get("model_type", "t5")
        self.is_umt5 = self.model_type == "umt5"

        self._supports_flex_attention = check_flex_attention_support()
        if self.is_umt5 and self._supports_flex_attention and is_hip():
            logger.warning("Disabling Flex Attention for UMT5 on HIP/ROCm due to Triton compatibility issues.")
            self._supports_flex_attention = False

        self._encoder: Optional[T5EncoderStack] = None
        self._config: Optional[T5Config] = None
        self._max_seq_len = 256  # Default, can be adjusted
        self._score_mods_device: Optional[torch.device] = None  # Track device for lazy rebinding
        self._flex_compiled = False  # Whether attention layers have been torch.compiled

        # Create a placeholder embedding so get_input_embeddings() works before
        # load_state_dict(). ComfyUI's process_tokens() calls this during encoding.
        # The weights are uninitialized — they get replaced by load_state_dict().
        vocab_size = config_dict.get("vocab_size", 32128)
        d_model = config_dict.get("d_model", 4096)
        self._shared = nn.Embedding(vocab_size, d_model, device=device, dtype=self.dtype)
        self._embeddings = NAIT5Embeddings(self._shared)

    def get_input_embeddings(self):
        return self._embeddings

    def set_input_embeddings(self, embeddings):
        if self._encoder is not None:
            self._encoder.vocab_embed = embeddings
        self._embeddings = NAIT5Embeddings(embeddings)

    def _ensure_flex_ready(self, device: torch.device) -> None:
        """Bind score mods and compile attention layers on the execution device.

        Called lazily on first forward (and after device changes) because:
        - ComfyUI's ModelPatcher moves submodules to CUDA individually,
          bypassing our .to() override.
        - torch.compile must happen on the execution device; compiling on CPU
          bakes CPU tensor references into the Inductor graph.
        - The compiled forward looks up self.score_mod at runtime, so
          re-binding score mods works without re-compiling.
        """
        if not self._supports_flex_attention:
            return
        if self._score_mods_device == device:
            return

        logger.info(f"Binding Flex Attention score mods on {device} for seq_len={self._max_seq_len}...")
        self._encoder.bind_score_mods(seq_len=self._max_seq_len)
        self._score_mods_device = device

        if not self._flex_compiled:
            logger.info("Compiling attention layers for Flex Attention...")
            for layer in self._encoder.layers:
                layer.attn.forward = torch.compile(
                    layer.attn.forward,
                    dynamic=False,
                    mode="max-autotune-no-cudagraphs",
                )
            self._flex_compiled = True
            logger.info("Attention layers compiled successfully.")

    def _manual_forward(
        self,
        hidden_states: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        intermediate_output: Optional[int] = None,
        final_layer_norm_intermediate: bool = True,
    ):
        """Manual forward pass to get intermediate outputs."""
        intermediate = None
        with torch.inference_mode():
            attn_args: list[SDPAArgs|FlexArgs]
            if self._supports_flex_attention:
                from torch.nn.attention.flex_attention import create_block_mask
                from nai_t5_wrapper.t5_encoder import make_self_attn_block_mask

                self._ensure_flex_ready(hidden_states.device)
                block_mask = make_self_attn_block_mask(mask=input_mask, mask_pad_queries=True, create_block_mask=create_block_mask) if input_mask is not None else None
                attn_args = [FlexArgs(block_mask=block_mask)] * self._config.num_layers
            else:
                batch_size = hidden_states.size(0)
                seq_len = hidden_states.size(-2)
                position_bias = self._encoder.relative_attention_bias(seq_len)
                # Ensure position_bias matches hidden_states dtype for numerical stability
                if position_bias.dtype != hidden_states.dtype:
                    position_bias = position_bias.to(dtype=hidden_states.dtype)
                biases = position_bias.unbind() if self._config.pos_emb_per_layer else [position_bias] * self._config.num_layers
                # Create dummy input_ids with correct shape for broadcast_mask
                # broadcast_mask expects [batch, seq_len] but we only have hidden_states [batch, seq_len, hidden_dim]
                if input_mask is not None:
                    dummy_ids = torch.zeros(batch_size, seq_len, device=hidden_states.device, dtype=torch.long)
                    attn_mask = self._encoder.broadcast_mask(input_mask, dummy_ids)
                else:
                    attn_mask = None
                attn_args = [SDPAArgs(position_bias=bias, mask=attn_mask) for bias in biases]

            x_r = ActAndResidual(self._encoder.dropout(hidden_states), None)

            for i, (layer, attn_args_) in enumerate(zip(self._encoder.layers, attn_args)):
                x_r = layer(x_r, attn_args_)
                if i == intermediate_output:
                    intermediate = x_r[0].clone()

            x, residual = x_r
            hidden_states = self._encoder.ln(x, residual=residual, prenorm=False)

            if intermediate is not None and final_layer_norm_intermediate:
                intermediate = self._encoder.ln(intermediate, prenorm=False)

        return hidden_states, intermediate

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        embeds: Optional[torch.Tensor] = None,
        intermediate_output: Optional[int] = None,
        final_layer_norm_intermediate: bool = True,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if self._encoder is None:
            raise RuntimeError("Model weights not loaded. Call load_state_dict first.")

        device = self._encoder.vocab_embed.weight.device
        model_dtype = self._encoder.layers[0].attn.qkv_proj.weight.dtype

        # Convert attention_mask to bool
        input_mask = attention_mask.bool().to(device) if attention_mask is not None else None

        if embeds is not None:
            hidden_states = embeds.to(device=device, dtype=model_dtype)
        else:
            # Embeddings are fp32; cast to model dtype (bf16)
            hidden_states = self._encoder.vocab_embed(input_ids.to(device)).to(dtype=model_dtype)

        hidden_states, intermediate = self._manual_forward(
            hidden_states, input_mask, intermediate_output, final_layer_norm_intermediate
        )

        if dtype is not None:
            hidden_states = hidden_states.to(dtype)
            if intermediate is not None:
                intermediate = intermediate.to(dtype)

        return hidden_states, intermediate

    def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = False):
        if not any(k.startswith("encoder.block.") for k in state_dict.keys()):
            raise ValueError("Unknown state dict format. Expected HuggingFace T5 keys.")

        # Upcast all incoming weights to fp32 so that QKV concatenation, norm fusion,
        # and every other weight transformation happens at full precision.
        # The cast down to the target dtype (fp16/bf16) is the very last step.
        state_dict = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}

        if self._encoder is None:
            self._create_encoder(state_dict)

        nai_state = hf_to_based_t5_enc_state(state_dict, self._config)
        self._encoder.load_state_dict(nai_state, assign=assign)
        self._encoder.eval()

        # Fuse norm scales while everything is still fp32 — the fused weights
        # are stored back in-place, so this avoids the precision loss of fusing
        # into an fp16 buffer and round-tripping through a narrow format.
        logger.info("Fusing norm scales...")
        fuse_norm_scales_enc(self._encoder, fuse_via_f32=True)

        # NOW cast to bf16 and move to device
        self._encoder.to(device=self.device, dtype=self.dtype)

        # Keep embeddings in fp32 to prevent overflow/Inf in low-precision lookups
        self._encoder.vocab_embed.to(dtype=torch.float32)

        # Keep final layer norm in fp32 to match the fp32 residual stream
        # (RMSNormCast uses residual_in_fp32=True)
        if self._encoder.ln.weight is not None:
            self._encoder.ln.weight.data = self._encoder.ln.weight.data.float()

        # Score mods for flex attention are bound lazily in _manual_forward
        # on the actual execution device, since ComfyUI's ModelPatcher moves
        # submodules to CUDA individually (bypassing our .to() override).

        self._embeddings = NAIT5Embeddings(self._encoder.vocab_embed)
        logger.info("NAI-T5 encoder loaded successfully.")
        return torch.nn.modules.module._IncompatibleKeys([], [])

    def _create_encoder(self, state_dict: dict):
        hf_config = self._build_hf_config_from_state(state_dict)
        attn_impl = T5AttnImpl.Flex if self._supports_flex_attention else T5AttnImpl.SDPA

        self._config = to_based_config(hf_config, n_tokens=self._max_seq_len)
        self._config = self._config.model_copy(update={
            # Allocate all parameters in fp32 so that QKV concatenation, norm fusion,
            # and other weight transformations happen at full precision.
            # load_state_dict() casts to self.dtype as the very last step.
            'emb_weight_dtype': torch.float32,
            'linear_weight_dtype': torch.float32,
            'norm_weight_dtype': torch.float32,
            'attn_impl': attn_impl,
            'elementwise_affine': True,
            'flex_kernel_options': {'BLOCK_M': 128, 'BLOCK_N': 64} if self._supports_flex_attention else {},
            # T5 was trained without QK scaling (scale_qk=False). ComfyUI's native T5
            # handles fp16 by pre-scaling K to cancel SDPA's default scaling. We must
            # also use scale_qk=False to match the original attention distribution.
            'scale_qk': False,
        })

        logger.info(f"Creating NAI-T5 encoder ({self.model_type}) with dtype={self.dtype}, attn={attn_impl.value}")
        self._encoder = T5EncoderStack(self._config)
        self.num_layers = self._config.num_layers

    def _build_hf_config_from_state(self, state_dict: dict) -> Any:
        class HFConfig:
            def __init__(self, config_dict):
                self.config_dict = config_dict
                # Ensure a few required fields have defaults if not in config
                self.relative_attention_num_buckets = self.config_dict.get('relative_attention_num_buckets', 32)
                self.relative_attention_max_distance = self.config_dict.get('relative_attention_max_distance', 128)

            def __getattr__(self, name):
                return self.config_dict.get(name)

        return HFConfig(self.config_dict)

    def to(self, *args, **kwargs):
        if 'device' in kwargs:
            self.device = kwargs['device']

        # Strip dtype from args/kwargs — we always stay in bf16.
        # ComfyUI's ModelPatcher may try to cast us to fp16 via .to(dtype=fp16).
        kwargs.pop('dtype', None)
        args = tuple(a for a in args if not isinstance(a, torch.dtype))

        if self._encoder is not None:
            self._encoder.to(*args, **kwargs)
            # Invalidate cached score mods so they get re-bound on the new device
            self._score_mods_device = None

        return super().to(*args, **kwargs)
