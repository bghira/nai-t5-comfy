# NAI-T5-Comfy: ComfyUI custom node for NovelAI's T5 with Flex Attention
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This package provides drop-in replacement T5 loaders for ComfyUI that use
# NovelAI's optimized T5 implementation with Flex Attention.
#
# Available nodes:
# - NAI T5 XXL Loader: Load T5 XXL for Flux, SD3, PixArt, etc.
# - NAI UMT5 XXL Loader: Load UMT5 XXL for Wan video models

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Version info
__version__ = "1.0.0"
