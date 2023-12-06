import argparse
import datetime
import os
import time

import albumentations as A
import cv2
import torch
import torchvision
from torch.utils.data import DataLoader

from data.build import build_dataset
from data.samplers import MPerClassSampler
from data.transforms import ACompose, CustomRandomCrop
from misc.engine import Trainer
from misc.losses import NegativeCosineSimilarityLoss
from misc.metric import calc_map_prak
from misc.utils import AverageMeter, compute_distance_matrix, get_combinations


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


class SimSiamLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = NegativeCosineSimilarityLoss()

    def forward(self, embeddings, targets):
        ps, zs = embeddings
        return self.forward_impl(ps, zs, targets)

    def forward_impl(self, ps, zs, targets):
        n = ps.size(0)
        eyes_ = torch.eye(n, dtype=torch.bool).cuda()
        pos_mask = targets.expand(
            targets.shape[0], n
        ).t() == targets.expand(n, targets.shape[0])
        pos_mask[:, :n] = pos_mask[:, :n] * ~eyes_

        groups = []
        for i in range(n):
            it = torch.tensor([i], device=ps.device)
            pos_pair_idx = torch.nonzero(pos_mask[i, i:]).view(-1)
            if pos_pair_idx.shape[0] > 0:
                combinations = get_combinations(it, pos_pair_idx + i)
                groups.append(combinations)

        groups = torch.cat(groups, dim=0)
        p1, p2 = ps[groups[:, 0]], ps[groups[:, 1]]
        z1, z2 = zs[groups[:, 0]], zs[groups[:, 1]]

        loss = (self.criterion(p1, z2) + self.criterion(p2, z1)) * 0.5
        return loss


class GeshaemTrainer(Trainer):

    def get_criterion(self):
        return SimSiamLoss()

    def get_transforms(self):
        img_size = self.config.DATA.IMG_SIZE
        train_transforms = torchvision.transforms.Compose([
            ACompose([
                A.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=15, p=0.5, value=(255, 255, 255),
                                   border_mode=cv2.BORDER_CONSTANT)
            ]),
            torchvision.transforms.RandomAffine(5, translate=(0.1, 0.1), fill=255),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomApply([
                torchvision.transforms.GaussianBlur((3, 3), (1.0, 2.0)),
            ], p=0.5),
            CustomRandomCrop(img_size, white_percentage_limit=0.85),
            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.3),
            ]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        val_transforms = torchvision.transforms.Compose([
            CustomRandomCrop(img_size, white_percentage_limit=0.85),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return {
            'train': train_transforms,
            'validation': val_transforms,
        }

    def get_dataloader(self, mode):
        if mode in self.data_loader_registers:
            return self.data_loader_registers[mode]

        dataset, repeat = build_dataset(mode, self.config, self.get_transforms())

        if mode == 'train':
            max_dataset_length = len(dataset) * repeat
            sampler = MPerClassSampler(dataset.data_labels, m=3, length_before_new_iter=max_dataset_length)
            sampler.set_epoch = lambda x: x
            data_loader = DataLoader(dataset, sampler=sampler, pin_memory=True, batch_size=self.config.DATA.BATCH_SIZE,
                                     drop_last=True, num_workers=self.config.DATA.NUM_WORKERS)

        else:
            data_loader = DataLoader(dataset, shuffle=False, pin_memory=True, batch_size=self.config.DATA.BATCH_SIZE,
                                     drop_last=False, num_workers=self.config.DATA.NUM_WORKERS)

        self.data_loader_registers[mode] = data_loader
        return data_loader

    def validate_dataloader(self, data_loader):
        batch_time = AverageMeter()
        mAP_meter = AverageMeter()
        top1_meter = AverageMeter()
        pk10_meter = AverageMeter()
        pk5_meter = AverageMeter()

        start = time.time()
        end = time.time()
        features = {}
        for idx, (images, targets) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=self.config.AMP_ENABLE):
                p, z = self.model(images)

            for feature, target in zip(p, targets.numpy()):
                features.setdefault(target, [])
                features[target].append(feature)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % self.config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                self.logger.info(
                    f'Eval: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')

        features = {k: torch.stack(v).cuda() for k, v in features.items()}
        distance_df = compute_distance_matrix(features, reduction=args.distance_reduction,
                                              distance_fn=NegativeCosineSimilarityLoss())

        index_to_fragment = {i: x for i, x in enumerate(data_loader.dataset.fragments)}
        distance_df.rename(columns=index_to_fragment, index=index_to_fragment, inplace=True)

        positive_pairs = data_loader.dataset.fragment_to_group
        distance_matrix = distance_df.to_numpy()
        self.logger.info(f"Number of groups: {len(distance_df.columns)}")
        m_ap, (top1, pr_a_k5, pr_a_k10) = calc_map_prak(distance_matrix, distance_df.columns, positive_pairs,
                                                        prak=(1, 5, 10))
        mAP_meter.update(m_ap)
        top1_meter.update(top1)
        pk5_meter.update(pr_a_k5)
        pk10_meter.update(pr_a_k10)
        test_time = datetime.timedelta(seconds=int(time.time() - start))

        mAP_meter.all_reduce()
        top1_meter.all_reduce()
        pk10_meter.all_reduce()
        pk5_meter.all_reduce()

        self.logger.info(
            f'Overall:\t'
            f'Time {test_time}\t'
            f'Batch Time {batch_time.avg:.3f}\t'
            f'mAP {mAP_meter.avg:.4f}\t'
            f'top1 {top1_meter.avg:.3f}\t'
            f'pr@k5 {pk5_meter.avg:.3f}'
            f'pr@k10 {pk10_meter.avg:.3f}\t')

        val_loss = 1 - mAP_meter.avg
        similarity_df = (2 - distance_df) / 2.

        return val_loss, similarity_df.round(3)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        data_loader = self.get_dataloader('test')
        _, similarity_df = self.validate_dataloader(data_loader)
        similarity_df.to_csv(os.path.join(self.config.OUTPUT, 'similarity_matrix.csv'))

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        data_loader = self.get_dataloader('validation')
        loss, _ = self.validate_dataloader(data_loader)
        return loss


if __name__ == '__main__':
    args, _ = parse_option()
    trainer = GeshaemTrainer(args)
    if args.mode == 'eval':
        trainer.validate()
    elif args.mode == 'test':
        trainer.test()
    elif args == 'throughput':
        trainer.throughput()
    else:
        trainer.train()
