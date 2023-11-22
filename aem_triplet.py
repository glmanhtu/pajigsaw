import argparse
import datetime
import json
import os
import time

import numpy as np
import torch
import albumentations as A
import torchvision
from torch.utils.data import DataLoader

from data.datasets.aem_dataset import AEMLetterDataset, AEMDataLoader
from data.transforms import PadCenterCrop
from misc import wi19_evaluate
from misc.engine import Trainer
from misc.utils import AverageMeter, compute_distance_matrix
from misc.wi19_evaluate import compute_pr_a_k


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
    parser.add_argument('--distance-reduction', type=str, default='mean')
    parser.add_argument('--letters', type=str, default=['α', 'ε', 'μ'], nargs='+')
    parser.add_argument('--triplet-files', type=str, default=['BT120220128.triplet', 'Eps20220408.triplet',
                                                            'mtest.triplet'], nargs='+')
    parser.add_argument('--with_likely', action='store_true')
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'throughput'], default='train')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    return parser.parse_known_args()


def add_items_to_group(items, groups):
    reference_group = {}
    for g_id, group in enumerate(groups):
        for fragment_id in items:
            if fragment_id in group and g_id not in reference_group:
                reference_group[g_id] = group

    if len(reference_group) > 0:
        reference_ids = list(reference_group.keys())
        for fragment_id in items:
            reference_group[reference_ids[0]].add(fragment_id)
        for g_id in reference_ids[1:]:
            for fragment_id in reference_group[g_id]:
                reference_group[reference_ids[0]].add(fragment_id)
            del groups[g_id]
    else:
        groups.append(set(items))


def load_triplet_file(filter_file, with_likely=False):
    positive_groups, negative_pairs = [], {}
    positive_pairs = {}
    mapping = {}
    with open(filter_file) as f:
        triplet_filter = json.load(f)
    for item in triplet_filter['relations']:
        current_tm = item['category']
        mapping[current_tm] = {}
        for second_item in item['relations']:
            second_tm = second_item['category']
            relationship = second_item['relationship']
            mapping[current_tm][second_tm] = relationship

    for item in triplet_filter['histories']:
        current_tm, second_tm = item['category'], item['secondary_category']
        relationship = mapping[current_tm][second_tm]
        if relationship == 4 or relationship == 3:
            negative_pairs.setdefault(current_tm, set([])).add(second_tm)
            negative_pairs.setdefault(second_tm, set([])).add(current_tm)
        if relationship == 1:
            add_items_to_group([current_tm, second_tm], positive_groups)
        if with_likely and relationship == 2:
            positive_pairs.setdefault(current_tm, set([])).add(second_tm)
            positive_pairs.setdefault(second_tm, set([])).add(current_tm)

    for group in positive_groups:
        for tm in group:
            for tm2 in group:
                positive_pairs.setdefault(tm, set([])).add(tm2)

    return positive_pairs, negative_pairs


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.1, n_subsets=3, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.n_subsets = n_subsets

    def forward(self, emb, target):
        mini_batch = len(emb) // self.n_subsets
        embeddings = torch.split(emb, [mini_batch] * self.n_subsets, dim=0)
        targets = torch.split(target, [mini_batch] * self.n_subsets, dim=0)
        losses = []
        for sub_emb, sub_target in zip(embeddings, targets):
            losses.append(self.forward_impl(sub_emb, sub_target, sub_emb, sub_target))

        return sum(losses) / len(losses)

    def forward_impl(self, inputs_col, targets_col, inputs_row, targets_row):
        n = inputs_col.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs_col, inputs_row.t())
        # split the positive and negative pairs
        eyes_ = torch.eye(n, dtype=torch.uint8).cuda()
        pos_mask = targets_col.expand(
            targets_row.shape[0], n
        ).t() == targets_row.expand(n, targets_row.shape[0])
        neg_mask = ~pos_mask
        pos_mask[:, :n] = pos_mask[:, :n] * ~eyes_

        loss = list()
        neg_count = list()
        for i in range(n):
            pos_pair_idx = torch.nonzero(pos_mask[i, :]).view(-1)
            if pos_pair_idx.shape[0] > 0:
                pos_pair_ = sim_mat[i, pos_pair_idx]
                pos_pair_ = torch.sort(pos_pair_)[0]

                neg_pair_idx = torch.nonzero(neg_mask[i, :]).view(-1)
                neg_pair_ = sim_mat[i, neg_pair_idx]
                neg_pair_ = torch.sort(neg_pair_)[0]

                select_pos_pair_idx = torch.nonzero(
                    pos_pair_ < neg_pair_[-1] + self.margin
                ).view(-1)
                pos_pair = pos_pair_[select_pos_pair_idx]

                select_neg_pair_idx = torch.nonzero(
                    neg_pair_ > max(0.6, pos_pair_[-1]) - self.margin
                ).view(-1)
                neg_pair = neg_pair_[select_neg_pair_idx]

                pos_loss = torch.sum(1 - pos_pair)
                if len(neg_pair) >= 1:
                    neg_loss = torch.sum(neg_pair)
                    neg_count.append(len(neg_pair))
                else:
                    neg_loss = 0
                loss.append(pos_loss + neg_loss)
            else:
                loss.append(0)

        loss = sum(loss) / n
        return loss


class AEMTripletTrainer(Trainer):

    def get_dataloader(self, mode):
        if mode in self.data_loader_registers:
            return self.data_loader_registers[mode]

        img_size = self.config.DATA.IMG_SIZE
        custom_transform = A.Compose(
            [
                A.LongestMaxSize(max_size=img_size),
            ]
        )
        transforms = torchvision.transforms.Compose([
            lambda x: np.array(x),
            lambda x: custom_transform(image=x)['image'],
            torchvision.transforms.ToPILImage(),
        ])
        if mode == 'train':
            transforms = torchvision.transforms.Compose([
                transforms,
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                ], p=0.5),
                torchvision.transforms.RandomCrop(img_size, pad_if_needed=True, fill=255),
            ])
        else:
            transforms = torchvision.transforms.Compose([
                transforms,
                PadCenterCrop(img_size, pad_if_needed=True, fill=255)
            ])

        transforms = torchvision.transforms.Compose([
            transforms,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        datasets = []
        for letter in args.letters:
            dataset = AEMLetterDataset(self.config.DATA.DATA_PATH, transforms, letter)
            datasets.append(dataset)

        if mode == 'train':
            data_loader = AEMDataLoader(datasets, batch_size=self.config.DATA.BATCH_SIZE,
                                        numb_workers=self.config.DATA.NUM_WORKERS,
                                        pin_memory=self.config.DATA.PIN_MEMORY)
        else:
            data_loader = {}
            for idx, letter in enumerate(args.letters):
                data_loader[letter] = DataLoader(datasets[idx], batch_size=self.config.DATA.BATCH_SIZE,
                                                 num_workers=self.config.DATA.NUM_WORKERS,
                                                 pin_memory=self.config.DATA.PIN_MEMORY, drop_last=False,
                                                 shuffle=False)
        self.data_loader_registers[mode] = data_loader
        return data_loader

    def get_criterion(self):
        return TripletLoss(margin=0.15, n_subsets=len(args.letters))

    def validate_dataloader(self, data_loader, triplet_def):
        batch_time = AverageMeter()
        mAP_meter = AverageMeter()
        top1_meter = AverageMeter()
        pk5_meter = AverageMeter()

        end = time.time()
        features = {}
        for idx, (images, targets) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=self.config.AMP_ENABLE):
                embs = self.model(images)

            for feature, target in zip(embs, targets.numpy()):
                tm = data_loader.dataset.labels[target]
                features.setdefault(tm, []).append(feature)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        features = {k: torch.stack(v).cuda() for k, v in features.items()}
        distance_df = compute_distance_matrix(features, reduction=args.distance_reduction)

        tms = []
        dataset_tms = set(distance_df.columns)
        positive_pairs, _ = triplet_def
        for tm in list(positive_pairs.keys()):
            if tm not in dataset_tms:
                del positive_pairs[tm]
            else:
                positive_pairs[tm] = positive_pairs[tm].intersection(dataset_tms)
        for tm in positive_pairs:
            if len(positive_pairs[tm]) > 1:
                tms.append(tm)

        categories = sorted(tms)
        distance_df = distance_df.loc[categories, categories]

        positive_pairs, _ = triplet_def
        correct_retrievals = distance_df.copy(deep=True) * 0
        for row in distance_df.index:
            for col in distance_df.columns:
                if col in positive_pairs[row]:
                    correct_retrievals[col][row] = 1
                    correct_retrievals[row][col] = 1
        correct_retrievals = correct_retrievals.to_numpy() > 0
        distance_matrix = distance_df.to_numpy()
        precision_at, recall_at, sorted_retrievals = wi19_evaluate.get_precision_recall_matrices(
            distance_matrix, classes=None, remove_self_column=True, correct_retrievals=correct_retrievals)

        non_singleton_idx = sorted_retrievals.sum(axis=1) > 0
        mAP = wi19_evaluate.compute_map(precision_at[non_singleton_idx, :], sorted_retrievals[non_singleton_idx, :])
        top_1 = sorted_retrievals[:, 0].sum() / len(sorted_retrievals)
        pr_a_k5 = compute_pr_a_k(sorted_retrievals, 5)

        mAP_meter.update(mAP)
        top1_meter.update(top_1)
        pk5_meter.update(pr_a_k5)

        mAP_meter.all_reduce()
        top1_meter.all_reduce()
        pk5_meter.all_reduce()

        return mAP_meter.avg, top1_meter.avg, pk5_meter.avg

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        mode = 'validation'
        final_map, final_top1, final_pra5 = [], [], []
        for letter, dataloader in self.get_dataloader(mode).items():
            letter_idx = args.letters.index(letter)
            triplet_def = load_triplet_file(args.triplet_files[letter_idx], args.with_likely)

            m_ap, top1, pra5 = self.validate_dataloader(dataloader, triplet_def)
            self.logger.info(
                f'Letter {letter}:'
                f'mAP {m_ap:.4f}\t'
                f'top1 {top1:.3f}\t'
                f'pr@k10 {pra5:.3f}\t')

            final_map.append(m_ap)
            final_top1.append(top1)
            final_pra5.append(pra5)

        final_map = sum(final_map) / len(final_map)
        final_top1 = sum(final_top1) / len(final_top1)
        final_pra5 = sum(final_pra5) / len(final_pra5)

        self.logger.info(
            f'Average:'
            f'mAP {final_map:.4f}\t'
            f'top1 {final_top1:.3f}\t'
            f'pr@k10 {final_pra5:.3f}\t')

        return 1 - final_map


if __name__ == '__main__':
    args, _ = parse_option()
    trainer = AEMTripletTrainer(args)
    if args.mode == 'eval':
        trainer.validate()
    elif args == 'throughput':
        trainer.throughput()
    else:
        trainer.train()
