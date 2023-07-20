# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .vision_transformer import VisionTransformerCustom


def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    if model_type == 'pjs':
        model = VisionTransformerCustom(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.PJS.PATCH_SIZE,
            in_chans=config.MODEL.PJS.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.PJS.EMBED_DIM,
            depth=config.MODEL.PJS.DEPTH,
            c_depth=config.MODEL.PJS.C_DEPTH,
            num_heads=config.MODEL.PJS.NUM_HEADS,
            mlp_ratio=config.MODEL.PJS.MLP_RATIO,
            qkv_bias=config.MODEL.PJS.QKV_BIAS,
            keep_attn=config.MODEL.PJS.KEEP_ATTN,
            arch_version=config.MODEL.PJS.ARCH_VERSION
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
