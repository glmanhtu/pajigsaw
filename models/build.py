# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import torch
from timm.models import VisionTransformer

from .resnet import ResNet32MixConv, ResNet, ResNetWrapper
from .simsiam import SimSiam, SimSiamV2, SimSiamV2CE
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
    elif model_type == 'vit':
        model = VisionTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            in_chans=config.MODEL.VIT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.VIT.EMBED_DIM,
            depth=config.MODEL.VIT.DEPTH,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            mlp_ratio=config.MODEL.VIT.MLP_RATIO,
            qkv_bias=config.MODEL.VIT.QKV_BIAS,
        )
    elif model_type == 'ss':
        model = SimSiam(
            arch=config.MODEL.SS.ARCH,
            pretrained=config.MODEL.SS.PRETRAINED,
            dim=config.MODEL.SS.EMBED_DIM,
            pred_dim=config.MODEL.SS.PRED_DIM,
            dropout=config.MODEL.SS.DROPOUT
        )
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif model_type == 'ss2':
        model = SimSiamV2(
            arch=config.MODEL.SS.ARCH,
            pretrained=config.MODEL.SS.PRETRAINED,
            dim=config.MODEL.SS.EMBED_DIM,
            pred_dim=config.MODEL.SS.PRED_DIM,
            dropout=config.MODEL.SS.DROPOUT
        )
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif model_type == 'ss2ce':
        model = SimSiamV2CE(
            arch=config.MODEL.SS.ARCH,
            pretrained=config.MODEL.SS.PRETRAINED,
            dim=config.MODEL.SS.EMBED_DIM,
            pred_dim=config.MODEL.SS.PRED_DIM,
            dropout=config.MODEL.SS.DROPOUT,
            n_classes=config.MODEL.SS.N_CLASSES
        )
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    elif model_type == 'resnet':
        model = ResNetWrapper(
            backbone=config.MODEL.RES.ARCH,
            weights=config.MODEL.RES.PRETRAINED,
            layers_to_freeze=config.MODEL.RES.LAYERS_FREEZE
        )
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif model_type == 'mixconv':
        model = ResNet32MixConv(
            img_size=(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
            backbone=config.MODEL.MIXCONV.ARCH,
            out_channels=config.MODEL.MIXCONV.OUT_CHANNELS,
            mix_depth=config.MODEL.MIXCONV.MIX_DEPTH,
            out_rows=config.MODEL.MIXCONV.OUT_ROWS,
            weights=config.MODEL.MIXCONV.PRETRAINED,
            layers_to_freeze=config.MODEL.MIXCONV.LAYERS_FREEZE
        )
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
