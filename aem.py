import argparse
import copy
import json
import os.path
import time

import albumentations as A
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.decomposition import PCA
from data.datasets.aem_dataset import AEMLetterDataset, AEMDataLoader
from data.transforms import PadCenterCrop, ACompose
from misc.engine import Trainer
from misc.losses import LossCombination, NegativeCosineSimilarityLoss
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
    parser.add_argument('--m-per-class', type=int, default=5, help='number of samples per category')
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--train-data-path', type=str, help='Optional different train set', default='')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--combine-loss-weight', type=float, default=0.7)
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--distance-reduction', type=str, default='mean')
    parser.add_argument('--letters', type=str, default=['α', 'ε', 'μ'], nargs='+')
    parser.add_argument('--triplet-files', type=str, default=['BT120220128.triplet', 'Eps20220408.triplet',
                                                            'mtest.triplet'], nargs='+')
    parser.add_argument('--with-likely', action='store_true')
    parser.add_argument('--use-pca', action='store_true', help='Utilise PCA whitening')
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


class ClassificationLoss(torch.nn.Module):
    def __init__(self, n_subsets=3, weight=1.):
        super().__init__()
        self.n_subsets = n_subsets
        self.criterion = torch.nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, embeddings, targets):
        if self.n_subsets == 1:
            return 0.

        _, _, cls = embeddings
        mini_batch = len(targets) // self.n_subsets

        labels = []
        for i in range(self.n_subsets):
            labels.append(torch.tensor([i] * mini_batch, device=cls.device, dtype=torch.int64))

        labels = torch.cat(labels, dim=0)
        return self.criterion(cls, labels) * self.weight


class SimSiamLoss(torch.nn.Module):
    def __init__(self, n_subsets=3, weight=1.):
        super().__init__()
        self.n_subsets = n_subsets
        self.criterion = NegativeCosineSimilarityLoss()
        self.weight = weight

    def forward(self, embeddings, targets):
        ps, zs, _ = embeddings
        mini_batch = len(targets) // self.n_subsets
        ps = torch.split(ps, [mini_batch] * self.n_subsets, dim=0)
        zs = torch.split(zs, [mini_batch] * self.n_subsets, dim=0)
        targets = torch.split(targets, [mini_batch] * self.n_subsets, dim=0)

        losses = []
        for p, z, target in zip(ps, zs, targets):
            losses.append(self.forward_impl(p, z, target))

        return sum(losses) / len(losses)

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
        return loss * self.weight


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.1, n_subsets=3):
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
        eyes_ = torch.eye(n, dtype=torch.bool).cuda()
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


class AEMTrainer(Trainer):

    def get_transforms(self):
        img_size = self.config.DATA.IMG_SIZE
        train_transforms = torchvision.transforms.Compose([
            ACompose([
                A.LongestMaxSize(max_size=img_size),
                A.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=15, p=0.5),
            ]),
            torchvision.transforms.RandomApply([
                torchvision.transforms.GaussianBlur((3, 3), (1.0, 2.0)),
                torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            ], p=0.5),
            torchvision.transforms.RandomCrop(img_size, pad_if_needed=True, fill=255),
            torchvision.transforms.RandomGrayscale(p=0.3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        val_transforms = torchvision.transforms.Compose([
            ACompose([
                A.LongestMaxSize(max_size=img_size),
            ]),
            PadCenterCrop(img_size, pad_if_needed=True, fill=255),
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

        transforms = self.get_transforms()[mode]
        datasets = []
        for letter in args.letters:
            dataset_path = self.config.DATA.DATA_PATH
            if mode == 'train' and args.train_data_path != '':
                dataset_path = args.train_data_path
            dataset = AEMLetterDataset(dataset_path, transforms, letter)
            datasets.append(dataset)

        data_loader = []
        if mode == 'train':
            data_loader = AEMDataLoader(datasets, batch_size=self.config.DATA.BATCH_SIZE,
                                        m=args.m_per_class,
                                        numb_workers=self.config.DATA.NUM_WORKERS,
                                        pin_memory=self.config.DATA.PIN_MEMORY)
        else:
            for idx, letter in enumerate(args.letters):
                sub_data_loader = DataLoader(datasets[idx], batch_size=self.config.DATA.BATCH_SIZE,
                                             num_workers=self.config.DATA.NUM_WORKERS,
                                             pin_memory=self.config.DATA.PIN_MEMORY, drop_last=False, shuffle=False)
                data_loader.append(sub_data_loader)
        self.data_loader_registers[mode] = data_loader
        return data_loader

    def get_criterion(self):
        if self.is_simsiam():
            ssl = SimSiamLoss(n_subsets=len(args.letters), weight=args.combine_loss_weight)
            cls = ClassificationLoss(n_subsets=len(args.letters), weight=1 - args.combine_loss_weight)
            return LossCombination([ssl, cls])
        return TripletLoss(margin=0.15, n_subsets=len(args.letters))

    def is_simsiam(self):
        return 'ss' in self.config.MODEL.TYPE

    def validate_dataloader(self, data_loader, triplet_def):
        batch_time = AverageMeter()
        mAP_meter = AverageMeter()
        top1_meter = AverageMeter()
        pk5_meter = AverageMeter()

        end = time.time()
        embeddings, labels = [], []
        for idx, (images, targets) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=self.config.AMP_ENABLE):
                embs = self.model(images)
                if self.is_simsiam():
                    embs, _, _ = embs

            embeddings.append(embs)
            labels.append(targets)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        embeddings = torch.cat(embeddings)
        labels = torch.cat(labels)

        if args.use_pca:
            pca = PCA(self.config.PCA.DIM, whiten=True)
            self.logger.info(f"Fitting PCA with dim {self.config.PCA.DIM}!")
            desc = embeddings.cpu().numpy()
            desc = np.nan_to_num(desc)

            try:
                embeddings = pca.fit_transform(desc)
                embeddings = torch.from_numpy(embeddings)
            except:
                self.logger.info("Found nans in input. Skipping PCA!")

        # embeddings = F.normalize(embeddings, p=2, dim=1)
        features = {}
        for feature, target in zip(embeddings, labels.numpy()):
            tm = data_loader.dataset.labels[target]
            features.setdefault(tm, []).append(feature)

        features = {k: torch.stack(v).cuda() for k, v in features.items()}
        distance_df = compute_distance_matrix(features, reduction=args.distance_reduction,
                                              distance_fn=NegativeCosineSimilarityLoss())
        distance_file = os.path.join(self.config.OUTPUT, 'distance_matrix.csv')
        distance_df.to_csv(distance_file)

        tms = []
        dataset_tms = set(distance_df.columns)
        positive_pairs, negative_pairs = copy.deepcopy(triplet_def)
        for tm in list(positive_pairs.keys()):
            if tm in dataset_tms:
                positive_tms = positive_pairs[tm].intersection(dataset_tms)
                if len(positive_tms) > 1:
                    tms.append(tm)

        categories = sorted(tms)
        distance_df = distance_df.loc[categories, categories]

        distance_matrix = distance_df.to_numpy()
        m_ap, (top_1, pr_a_k5) = calc_map_prak(distance_matrix, distance_df.columns, positive_pairs, negative_pairs)

        mAP_meter.update(m_ap)
        top1_meter.update(top_1)
        pk5_meter.update(pr_a_k5)

        mAP_meter.all_reduce()
        top1_meter.all_reduce()
        pk5_meter.all_reduce()

        return mAP_meter.avg, top1_meter.avg, pk5_meter.avg, tms

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        mode = 'validation'
        final_map, final_top1, final_pra5 = [], [], []
        dataloaders = self.get_dataloader(mode)
        for idx, letter in enumerate(args.letters):
            triplet_def = load_triplet_file(args.triplet_files[idx], args.with_likely)

            m_ap, top1, pra5, tms = self.validate_dataloader(dataloaders[idx], triplet_def)
            self.logger.info(
                f'Letter {letter}:\t'
                f'N TMs: {len(tms)}\t' 
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
            f'pr@k5 {final_pra5:.3f}\t')

        return 1 - final_map


if __name__ == '__main__':
    args, _ = parse_option()
    trainer = AEMTrainer(args)
    if args.mode == 'eval':
        trainer.validate()
    elif args == 'throughput':
        trainer.throughput()
    else:
        trainer.train()
