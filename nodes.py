# NAI-T5-Comfy: ComfyUI custom node for NovelAI's T5 with Flex Attention
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file contains the ComfyUI node definitions for NAI-T5 text encoders.
# These nodes provide drop-in replacements for ComfyUI's built-in T5 loaders,
# using NovelAI's optimized T5 implementation with Flex Attention.

import os
import logging
from typing import Optional

import torch

# ComfyUI imports
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management
import comfy.model_patcher
import comfy.hooks
from comfy import sd1_clip
from comfy.text_encoders.spiece_tokenizer import SPieceTokenizer
from transformers import T5TokenizerFast

# Local imports
from .nai_t5_model import NAIT5ForComfy

logger = logging.getLogger(__name__)


# T5 XXL configuration (matches ComfyUI's t5_config_xxl.json)
T5_XXL_CONFIG = {
    "d_ff": 10240,
    "d_kv": 64,
    "d_model": 4096,
    "decoder_start_token_id": 0,
    "dropout_rate": 0.1,
    "eos_token_id": 1,
    "dense_act_fn": "gelu_pytorch_tanh",
    "initializer_factor": 1.0,
    "is_encoder_decoder": True,
    "is_gated_act": True,
    "layer_norm_epsilon": 1e-6,
    "model_type": "t5",
    "num_decoder_layers": 24,
    "num_heads": 64,
    "num_layers": 24,
    "output_past": True,
    "pad_token_id": 0,
    "relative_attention_num_buckets": 32,
    "relative_attention_max_distance": 128,
    "tie_word_embeddings": False,
    "vocab_size": 32128,
}

# UMT5 XXL configuration (matches ComfyUI's umt5_config_xxl.json)
UMT5_XXL_CONFIG = {
    "d_ff": 10240,
    "d_kv": 64,
    "d_model": 4096,
    "decoder_start_token_id": 0,
    "dropout_rate": 0.1,
    "eos_token_id": 1,
    "dense_act_fn": "gelu_pytorch_tanh",
    "initializer_factor": 1.0,
    "is_encoder_decoder": True,
    "is_gated_act": True,
    "layer_norm_epsilon": 1e-6,
    "model_type": "umt5",
    "num_decoder_layers": 24,
    "num_heads": 64,
    "num_layers": 24,
    "output_past": True,
    "pad_token_id": 0,
    "relative_attention_num_buckets": 32,
    "relative_attention_max_distance": 128,
    "tie_word_embeddings": False,
    "vocab_size": 256384,
}


class NAIT5XXLModel(sd1_clip.SDClipModel):
    """
    NAI-T5 XXL model that uses NovelAI's optimized T5 encoder with Flex Attention.
    This is a drop-in replacement for ComfyUI's T5XXLModel.
    """

    def __init__(
        self,
        device: str = "cpu",
        layer: str = "last",
        layer_idx: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        attention_mask: bool = False,
        model_options: dict = {},
    ):
        t5xxl_scaled_fp8 = model_options.get("t5xxl_scaled_fp8", None)
        if t5xxl_scaled_fp8 is not None:
            model_options = model_options.copy()
            model_options["scaled_fp8"] = t5xxl_scaled_fp8

        model_options = {**model_options, "model_name": "nai_t5xxl"}

        super().__init__(
            device=device,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config=T5_XXL_CONFIG,
            dtype=dtype,
            special_tokens={"end": 1, "pad": 0},
            model_class=NAIT5ForComfy,
            enable_attention_masks=attention_mask,
            return_attention_masks=attention_mask,
            model_options=model_options,
        )


class NAIUMT5XXLModel(sd1_clip.SDClipModel):
    """
    NAI-UMT5 XXL model that uses NovelAI's optimized T5 encoder with Flex Attention.
    This is a drop-in replacement for ComfyUI's UMT5XXlModel.
    """

    def __init__(
        self,
        device: str = "cpu",
        layer: str = "last",
        layer_idx: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        model_options: dict = {},
    ):
        model_options = {**model_options, "model_name": "nai_umt5xxl"}

        super().__init__(
            device=device,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config=UMT5_XXL_CONFIG,
            dtype=dtype,
            special_tokens={"end": 1, "pad": 0},
            model_class=NAIT5ForComfy,
            enable_attention_masks=True,
            zero_out_masked=True,
            model_options=model_options,
        )


class NAIT5XXLTokenizer(sd1_clip.SDTokenizer):
    """
    Tokenizer for T5 XXL, using the HuggingFace T5TokenizerFast.
    """

    def __init__(self, embedding_directory=None, tokenizer_data={}):
        # Find the t5_tokenizer directory more robustly
        tokenizer_path = None
        text_encoder_paths = folder_paths.get_folder_paths("text_encoders")
        for path in text_encoder_paths:
            p = os.path.join(path, "t5_tokenizer")
            if os.path.isdir(p):
                tokenizer_path = p
                break
        
        if tokenizer_path is None:
            # Fallback to old method for compatibility
            logger.warning("Could not find 't5_tokenizer' directory in text_encoder paths. Falling back to old method.")
            comfy_path = os.path.dirname(os.path.dirname(os.path.realpath(sd1_clip.__file__)))
            tokenizer_path = os.path.join(comfy_path, "comfy", "text_encoders", "t5_tokenizer")

        super().__init__(
            tokenizer_path,
            embedding_directory=embedding_directory,
            pad_with_end=False,
            embedding_size=4096,
            embedding_key='t5xxl',
            tokenizer_class=T5TokenizerFast,
            has_start_token=False,
            pad_to_max_length=False,
            max_length=99999999,
            min_length=256,
            tokenizer_data=tokenizer_data,
        )


class NAIUMT5XXLTokenizer(sd1_clip.SDTokenizer):
    """
    Tokenizer for UMT5 XXL, using SentencePiece.
    """

    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer = tokenizer_data.get("spiece_model", None)
        super().__init__(
            tokenizer,
            pad_with_end=False,
            embedding_size=4096,
            embedding_key='umt5xxl',
            tokenizer_class=SPieceTokenizer,
            has_start_token=False,
            pad_to_max_length=False,
            max_length=99999999,
            min_length=512,
            pad_token=0,
            tokenizer_data=tokenizer_data,
        )

    def state_dict(self):
        return {"spiece_model": self.tokenizer.serialize_model()}


class NAIT5XXLClipModel(sd1_clip.SD1ClipModel):
    """
    CLIP model wrapper for NAI-T5 XXL.
    """

    def __init__(self, device="cpu", dtype=None, model_options={}, **kwargs):
        super().__init__(
            device=device,
            dtype=dtype,
            model_options=model_options,
            name="t5xxl",
            clip_model=NAIT5XXLModel,
            **kwargs,
        )


class NAIUMT5XXLClipModel(sd1_clip.SD1ClipModel):
    """
    CLIP model wrapper for NAI-UMT5 XXL.
    """

    def __init__(self, device="cpu", dtype=None, model_options={}, **kwargs):
        super().__init__(
            device=device,
            dtype=dtype,
            model_options=model_options,
            name="umt5xxl",
            clip_model=NAIUMT5XXLModel,
            **kwargs,
        )


class NAIT5XXLTokenizerWrapper(sd1_clip.SD1Tokenizer):
    """
    Tokenizer wrapper for NAI-T5 XXL.
    """

    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(
            embedding_directory=embedding_directory,
            tokenizer_data=tokenizer_data,
            clip_name="t5xxl",
            tokenizer=NAIT5XXLTokenizer,
        )


class NAIUMT5XXLTokenizerWrapper(sd1_clip.SD1Tokenizer):
    """
    Tokenizer wrapper for NAI-UMT5 XXL.
    """

    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(
            embedding_directory=embedding_directory,
            tokenizer_data=tokenizer_data,
            clip_name="umt5xxl",
            tokenizer=NAIUMT5XXLTokenizer,
        )


# =============================================================================
# ComfyUI Node Definitions
# =============================================================================


class NAIT5XXLLoader:
    """
    Load a T5 XXL text encoder using NovelAI's optimized implementation.

    This provides a drop-in replacement for ComfyUI's CLIPLoader with type "sd3"
    but uses NovelAI's T5 with Flex Attention for better performance.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "t5xxl_name": (folder_paths.get_filename_list("text_encoders"),),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_t5"
    CATEGORY = "loaders/nai"
    DESCRIPTION = "Load T5 XXL text encoder using NovelAI's optimized implementation with Flex Attention. For use with Flux, SD3, PixArt, etc."

    def load_t5(self, t5xxl_name: str, device: str = "default"):
        # Get full path to the weights
        t5_path = folder_paths.get_full_path_or_raise("text_encoders", t5xxl_name)

        # Set up model options
        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        # Create the CLIP object
        clip = self._create_clip(
            clip_class=NAIT5XXLClipModel,
            tokenizer_class=NAIT5XXLTokenizerWrapper,
            ckpt_path=t5_path,
            model_options=model_options,
        )

        return (clip,)

    def _create_clip(self, clip_class, tokenizer_class, ckpt_path: str, model_options: dict):
        """Create a CLIP object with the given model and tokenizer classes."""
        # Load the checkpoint
        sd, metadata = comfy.utils.load_torch_file(ckpt_path, safe_load=True, return_metadata=True)
        if metadata is None:
            metadata = {}

        # Handle quantization metadata if present
        quant_metadata = metadata.get("_quantization_metadata", None)
        if quant_metadata is not None:
            model_options["quantization_metadata"] = quant_metadata
        
        tokenizer_data = metadata.get("tokenizer", {})

        # Get embedding directory
        embedding_directory = folder_paths.get_folder_paths("embeddings")

        # Create a target-like object for CLIP initialization
        class ClipTarget:
            def __init__(self, clip, tokenizer, params):
                self.clip = clip
                self.tokenizer = tokenizer
                self.params = params

        # Calculate model size for device placement
        parameters = sum(p.numel() for p in sd.values() if hasattr(p, 'numel'))

        target = ClipTarget(
            clip=clip_class,
            tokenizer=tokenizer_class,
            params={},
        )

        # Create CLIP object
        clip = comfy.sd.CLIP(
            target=target,
            embedding_directory=embedding_directory,
            parameters=parameters,
            model_options=model_options,
        )

        # Load the weights
        clip.cond_stage_model.load_sd(sd)

        return clip


class NAIUMT5XXLLoader:
    """
    Load a UMT5 XXL text encoder using NovelAI's optimized implementation.

    This provides a drop-in replacement for ComfyUI's CLIPLoader with type "wan"
    but uses NovelAI's T5 with Flex Attention for better performance.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "umt5xxl_name": (folder_paths.get_filename_list("text_encoders"),),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_umt5"
    CATEGORY = "loaders/nai"
    DESCRIPTION = "Load UMT5 XXL text encoder using NovelAI's optimized implementation with Flex Attention. For use with Wan video models."

    def load_umt5(self, umt5xxl_name: str, device: str = "default"):
        # Get full path to the weights
        umt5_path = folder_paths.get_full_path_or_raise("text_encoders", umt5xxl_name)

        # Set up model options
        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        # Create the CLIP object
        clip = self._create_clip(
            clip_class=NAIUMT5XXLClipModel,
            tokenizer_class=NAIUMT5XXLTokenizerWrapper,
            ckpt_path=umt5_path,
            model_options=model_options,
        )

        return (clip,)

    def _create_clip(self, clip_class, tokenizer_class, ckpt_path: str, model_options: dict):
        """Create a CLIP object with the given model and tokenizer classes."""
        # Load the checkpoint
        sd, metadata = comfy.utils.load_torch_file(ckpt_path, safe_load=True, return_metadata=True)
        if metadata is None:
            metadata = {}

        # Handle quantization metadata if present
        quant_metadata = metadata.get("_quantization_metadata", None)
        if quant_metadata is not None:
            model_options["quantization_metadata"] = quant_metadata
        
        tokenizer_data = metadata.get("tokenizer", {})

        # Get embedding directory
        embedding_directory = folder_paths.get_folder_paths("embeddings")

        # Create a target-like object for CLIP initialization
        class ClipTarget:
            def __init__(self, clip, tokenizer, params):
                self.clip = clip
                self.tokenizer = tokenizer
                self.params = params

        # Calculate model size for device placement
        parameters = sum(p.numel() for p in sd.values() if hasattr(p, 'numel'))

        target = ClipTarget(
            clip=clip_class,
            tokenizer=tokenizer_class,
            params={},
        )

        # Create CLIP object
        clip = comfy.sd.CLIP(
            target=target,
            embedding_directory=embedding_directory,
            parameters=parameters,
            model_options=model_options,
        )

        # Load the weights
        clip.cond_stage_model.load_sd(sd)

        return clip


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "NAIT5XXLLoader": NAIT5XXLLoader,
    "NAIUMT5XXLLoader": NAIUMT5XXLLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NAIT5XXLLoader": "NAI T5 XXL Loader",
    "NAIUMT5XXLLoader": "NAI UMT5 XXL Loader",
}
