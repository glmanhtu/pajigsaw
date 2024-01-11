# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .datasets.div2k_patch import DIV2KPatch, Div2kPatchTriplet
from .datasets.hisfrag_dataset import HisFrag20
from .datasets.michigan_dataset import MichiganDataset
from .datasets.pajigsaw_dataset import Pajigsaw

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


# def build_loader(config):
#     config.defrost()
#     dataset_train, train_repeat = build_dataset(mode='train', config=config)
#     config.freeze()
#     print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
#     dataset_val, val_repeat = build_dataset(mode='validation', config=config)
#     print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")
#
#     num_tasks = dist.get_world_size()
#     global_rank = dist.get_rank()
#     sampler_train = DistributedRepeatSampler(
#         dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, repeat=train_repeat
#     )
#
#     sampler_val = DistributedEvalSampler(
#         dataset_val, shuffle=config.TEST.SHUFFLE, rank=global_rank, num_replicas=num_tasks, repeat=val_repeat
#     )
#
#     data_loader_train = DataLoader(
#         dataset_train, sampler=sampler_train,
#         batch_size=config.DATA.BATCH_SIZE,
#         num_workers=config.DATA.NUM_WORKERS,
#         pin_memory=config.DATA.PIN_MEMORY,
#         drop_last=True,
#     )
#
#     data_loader_val = torch.utils.data.DataLoader(
#         dataset_val, sampler=sampler_val,
#         batch_size=config.DATA.TEST_BATCH_SIZE,
#         shuffle=False,
#         num_workers=config.DATA.NUM_WORKERS,
#         pin_memory=config.DATA.PIN_MEMORY,
#         drop_last=False
#     )
#
#     # setup mixup / cutmix
#     mixup_fn = None
#     mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
#     if mixup_active:
#         mixup_fn = Mixup(
#             mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
#             prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
#             label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
#
#     return data_loader_train, data_loader_val, mixup_fn
#

def build_dataset(mode, config, transforms):
    patch_size = config.DATA.IMG_SIZE
    repeat = 1
    transform = transforms[mode]
    if config.DATA.DATASET == 'hisfrag20':
        split = HisFrag20.Split.from_string(mode)
        repeat = 3
        dataset = HisFrag20(config.DATA.DATA_PATH, split, transform=transform)
    elif config.DATA.DATASET == 'div2k':
        split = DIV2KPatch.Split.from_string(mode)
        repeat = 5 if split.is_train() else 10
        dataset = DIV2KPatch(config.DATA.DATA_PATH, split, transform=transform, with_negative=True,
                             image_size=patch_size, erosion_ratio=config.DATA.EROSION_RATIO)
    elif config.DATA.DATASET == 'div2k_triplet':
        split = DIV2KPatch.Split.from_string(mode)
        repeat = 5 if split.is_train() else 10
        dataset = Div2kPatchTriplet(config.DATA.DATA_PATH, split, transform=transform, with_negative=True,
                                    image_size=patch_size, erosion_ratio=config.DATA.EROSION_RATIO)

    elif config.DATA.DATASET == 'pajigsaw':
        split = Pajigsaw.Split.from_string(mode)
        dataset = Pajigsaw(config.DATA.DATA_PATH, split, transform=transform, image_size=patch_size)

    elif config.DATA.DATASET == 'michigan':
        split = MichiganDataset.Split.from_string(mode)
        repeat = 3 if split.is_train() else 1
        dataset = MichiganDataset(config.DATA.DATA_PATH, split, transforms=transform)

    else:
        raise NotImplementedError(f"We haven't supported {config.DATA.DATASET}")

    return dataset, repeat
