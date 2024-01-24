import argparse
import datetime
import os
import time

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from ml_engine.criterion.losses import BatchDotProduct, NegativeLoss
from ml_engine.criterion.triplet import BatchWiseTripletLoss
from ml_engine.evaluation.distances import compute_distance_matrix_from_embeddings
from pytorch_metric_learning import samplers
from torch.utils.data import DataLoader, SequentialSampler

from data.build import build_dataset
from data.datasets.hisfrag_dataset import HisFrag20Test, HisFrag20
from data.samplers import DistributedIndicatesSampler
from data.transforms import ACompose
from misc import wi19_evaluate, utils
from misc.engine import Trainer
from misc.utils import AverageMeter, get_combinations


def parse_option():
    parser = argparse.ArgumentParser('Geshaem training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--eval-n-items-per-category', type=int, default=5,
                        help="Number of items per category to test")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--distance-reduction', type=str, default='min')
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'test', 'throughput'], default='train')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    return parser.parse_known_args()


class HisfragTrainer(Trainer):

    def get_criterion(self):
        return BatchWiseTripletLoss(margin=0.5)

    def get_transforms(self):
        patch_size = self.config.DATA.IMG_SIZE

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomAffine(5, translate=(0.1, 0.1), fill=0),
            ACompose([
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5, value=(0, 0, 0),
                                   border_mode=cv2.BORDER_CONSTANT),
            ]),
            torchvision.transforms.RandomCrop(patch_size, pad_if_needed=True),
            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            ], p=.5),
            torchvision.transforms.RandomApply([
                torchvision.transforms.GaussianBlur((3, 3), (1.0, 2.0)),
            ], p=.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        val_transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(patch_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(patch_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        return {
            'train': train_transform,
            'val': val_transforms,
            'test': test_transforms
        }

    def get_dataloader(self, mode):
        if mode in self.data_loader_registers:
            return self.data_loader_registers[mode]

        transforms = self.get_transforms()
        dataset, repeat = build_dataset(mode=mode, config=self.config, transforms=transforms)
        drop_last = True
        if mode == 'train':
            max_dataset_length = len(dataset) * repeat
            sampler = samplers.MPerClassSampler(dataset.data_labels, m=3, length_before_new_iter=max_dataset_length)
        else:
            sampler = SequentialSampler(dataset)
            drop_last = False
        sampler.set_epoch = lambda x: x
        dataloader = DataLoader(dataset, sampler=sampler, pin_memory=True, batch_size=self.config.DATA.BATCH_SIZE,
                                drop_last=drop_last, num_workers=self.config.DATA.NUM_WORKERS)

        self.data_loader_registers[mode] = dataloader
        return dataloader

    def validate_dataloader(self, data_loader):
        self.model.eval()
        batch_time, m_ap_meter = AverageMeter(), AverageMeter()

        end = time.time()
        embeddings, labels = None, None
        for idx, (images, targets) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=self.config.AMP_ENABLE):
                embs = self.model(images)

            if embeddings is None:
                embeddings = embs
                labels = targets
            else:
                embeddings = torch.cat((embeddings, embs))
                labels = torch.cat((labels, targets))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % self.config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                self.logger.info(
                    f'Eval: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')

        self.logger.info(f'N samples: {len(embeddings)}, N categories: {len(torch.unique(labels))}')

        criterion = NegativeLoss(BatchDotProduct(reduction='none'))
        distance_matrix = compute_distance_matrix_from_embeddings(embeddings, criterion,
                                                                  batch_size=self.config.DATA.TEST_BATCH_SIZE)
        return wi19_evaluate.get_metrics(distance_matrix.numpy(), labels.numpy())

    @torch.no_grad()
    def test(self):
        self.model.eval()
        dataloader = self.get_dataloader('test')
        m_ap, top1, pr_k10, pr_k100 = self.validate_dataloader(dataloader)
        self.logger.info(f'Test results: {m_ap:.3f}\t' f'Top 1 {top1:.3f}\t' f'Pr@k10 {pr_k10:.3f}\t' 
                         f'Pr@k100 {pr_k100:.3f}')

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        dataloader = self.get_dataloader('val')
        m_ap, top1, pr_k10, pr_k100 = self.validate_dataloader(dataloader)
        self.logger.info(f'Validation results: mAP {m_ap:.3f}\t' f'Top 1 {top1:.3f}\t' f'Pr@k10 {pr_k10:.3f}\t' 
                         f'Pr@k100 {pr_k100:.3f}')
        return 1 - m_ap


if __name__ == '__main__':
    args, _ = parse_option()
    trainer = HisfragTrainer(args)
    if args.mode == 'eval':
        trainer.validate()
    elif args.mode == 'test':
        trainer.test()
    elif args == 'throughput':
        trainer.throughput()
    else:
        trainer.train()
