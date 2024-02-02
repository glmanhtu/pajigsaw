import argparse
import datetime
import os
import statistics
import time

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from ml_engine.data.samplers import MPerClassSampler
from torch.utils.data import DataLoader

from data import transforms
from data.build import build_dataset
from data.datasets.geshaem_dataset import GeshaemPatch, MergeDataset
from data.datasets.michigan_dataset import MichiganTest, MichiganDataset
from data.samplers import DistributedIndicatesSampler
from data.transforms import ACompose, PadCenterCrop
from misc import wi19_evaluate
from misc.engine import Trainer
from misc.metric import calc_map_prak
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
    parser.add_argument('--geshaem-data-path', type=str, help='path to dataset')
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
        return torch.nn.BCEWithLogitsLoss(reduction='sum')

    def get_transforms(self):
        img_size = self.config.DATA.IMG_SIZE

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(img_size, pad_if_needed=True, fill=(255, 255, 255)),
            torchvision.transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            ACompose([
                A.CoarseDropout(max_holes=16, min_holes=3, min_height=16, max_height=64, min_width=16, max_width=64,
                                fill_value=255, p=0.9),
            ]),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.1),
            ], p=.5),
            transforms.GaussianBlur(p=0.5, radius_max=1),
            # transforms.Solarization(p=0.2),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        val_transforms = torchvision.transforms.Compose([
            PadCenterCrop((img_size, img_size), pad_if_needed=True, fill=(255, 255, 255)),
            torchvision.transforms.Resize(int(img_size * 1.15)),
            torchvision.transforms.CenterCrop(img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        return {
            'train': train_transform,
            'validation': val_transforms,
        }

    def get_dataloader(self, mode):
        if mode in self.data_loader_registers:
            return self.data_loader_registers[mode]
        if mode != 'train':
            raise Exception('Only Train mode should be executed')

        dataset, repeat = build_dataset(mode=mode, config=self.config, transforms=self.get_transforms())
        max_dataset_length = len(dataset) * 20
        self.logger.info(f'[{mode}] Dataset length: {max_dataset_length}')
        sampler = MPerClassSampler(dataset.data_labels, m=3, length_before_new_iter=max_dataset_length)
        sampler.set_epoch = lambda x: x
        dataloader = DataLoader(dataset, sampler=sampler, pin_memory=True, batch_size=self.config.DATA.BATCH_SIZE,
                                drop_last=True, num_workers=self.config.DATA.NUM_WORKERS)

        self.data_loader_registers[mode] = dataloader
        return dataloader

    def prepare_data(self, samples, targets):
        n = samples.size(0)
        device = samples.device
        # split the positive and negative pairs
        eyes_ = torch.eye(n, dtype=torch.bool, device=device)
        pos_mask = targets.expand(
            targets.shape[0], n
        ).t() == targets.expand(n, targets.shape[0])
        neg_mask = ~pos_mask
        pos_mask[:, :n] = pos_mask[:, :n] * ~eyes_

        pos_groups, neg_groups = None, None
        for i in range(n):
            it = torch.tensor([i], device=device)
            pos_pair_idx = torch.nonzero(pos_mask[i, i:]).view(-1)
            if pos_pair_idx.shape[0] > 0:
                combinations = get_combinations(it, pos_pair_idx + i)
                if pos_groups is None:
                    pos_groups = combinations
                else:
                    pos_groups = torch.cat((pos_groups, combinations), dim=0)

            neg_pair_idx = torch.nonzero(neg_mask[i, :]).view(-1)
            if neg_pair_idx.shape[0] > 0:
                combinations = get_combinations(it, neg_pair_idx)
                if neg_groups is None:
                    neg_groups = combinations
                else:
                    neg_groups = torch.cat((neg_groups, combinations), dim=0)

        neg_length = min(neg_groups.shape[0], pos_groups.shape[0])
        neg_groups = neg_groups[torch.randperm(neg_groups.shape[0])[:neg_length]]

        labels = [1.] * pos_groups.shape[0] + [0.] * neg_groups.shape[0]
        labels = torch.tensor(labels, dtype=torch.float32, device=samples.device).view(-1, 1)
        groups = torch.cat([pos_groups, neg_groups], dim=0)

        with torch.cuda.amp.autocast(enabled=self.config.AMP_ENABLE):
            x1 = self.model(samples, forward_first_part=True)

        x = samples[groups[:, 0]]
        x1 = x1[groups[:, 1]]
        return (x, x1), labels

    def train_step(self, samples):
        x, x1 = samples
        return self.model(x1, x)

    @torch.no_grad()
    def geshaem_test(self, key='geshaem_test'):
        self.model.eval()
        transform = self.get_transforms()['validation']
        if key in self.data_loader_registers:
            dataset = self.data_loader_registers[key]
        else:
            dataset = GeshaemPatch(args.geshaem_data_path, GeshaemPatch.Split.VAL, transform=transform)
            self.data_loader_registers[key] = dataset
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.DATA.TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.DATA.NUM_WORKERS,
            pin_memory=True,
            drop_last=False
        )

        batch_time = AverageMeter()
        end = time.time()
        distance_map = {}
        index_to_fragment = {i: x for i, x in enumerate(dataset.fragments)}
        for idx, (images, pairs) in enumerate(dataloader):
            images = images.cuda(non_blocking=True)
            with torch.cuda.amp.autocast(enabled=self.config.AMP_ENABLE):
                output = self.model(images).view(-1)

            for pair, score in zip(pairs.numpy(), output.float().cpu().numpy()):
                frag_i_idx, frag_j_idx = tuple(pair)
                frag_i, frag_j = index_to_fragment[frag_i_idx], index_to_fragment[frag_j_idx]
                distance_map.setdefault(frag_i, {}).setdefault(frag_j, []).append(1 - score)
                distance_map.setdefault(frag_j, {}).setdefault(frag_i, []).append(1 - score)

            batch_time.update(time.time() - end)
            end = time.time()
            if idx % self.config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (len(dataloader) - idx)
                self.logger.info(
                    f'Testing: [{idx}/{len(dataloader)}]\t'
                    f'X2 eta {datetime.timedelta(seconds=int(etas))}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
        stds = []
        mean_distance_map, min_distance_map = {}, {}
        for source in distance_map:
            for dest in distance_map[source]:
                avg_dis = sum(distance_map[source][dest]) / len(distance_map[source][dest])
                if len(distance_map[source][dest]) > 1:
                    stds.append(statistics.stdev(distance_map[source][dest]))
                mean_distance_map.setdefault(source, {})[dest] = avg_dis
                min_distance_map.setdefault(source, {})[dest] = min(distance_map[source][dest])

        avg_std = sum(stds) / len(stds)
        std_std = statistics.stdev(stds)
        self.logger.info(f'N categories: {len(distance_map)}\t Avg_Std {avg_std:.3f}\t Std_Std {std_std:.3f}')

        pos_pairs = dataset.fragment_to_group
        dist_df = pd.DataFrame.from_dict(mean_distance_map, orient='index')
        mean_m_ap, (top_1, prk5, prk10) = calc_map_prak(dist_df.to_numpy(), dist_df.columns, pos_pairs, prak=(1, 5, 10))

        self.logger.info(f'Geshaem test MEAN: mAP {mean_m_ap:.3f}\t' f'Top 1 {top_1:.3f}\t' f'Pr@k5 {prk5:.3f}\t'
                         f'Pr@k10 {prk10:.3f}\t')

        dist_df = pd.DataFrame.from_dict(min_distance_map, orient='index')
        min_m_ap, (top_1, prk5, prk10) = calc_map_prak(dist_df.to_numpy(), dist_df.columns, pos_pairs, prak=(1, 5, 10))
        self.logger.info(f'Geshaem test MIN: mAP {min_m_ap:.3f}\t' f'Top 1 {top_1:.3f}\t' f'Pr@k5 {prk5:.3f}\t'
                         f'Pr@k10 {prk10:.3f}\t')

        return 1 - max(mean_m_ap, min_m_ap)

    def validate_dataloader(self, split: MichiganTest.Split, remove_cache_file=False):
        self.model.eval()
        transform = self.get_transforms()[split.value]
        if 'michigan_test' in self.data_loader_registers:
            dataset = self.data_loader_registers['michigan_test']
        else:
            dataset = MichiganTest(self.config.DATA.DATA_PATH, split, transforms=transform,
                                   val_n_items_per_writer=self.config.DATA.EVAL_N_ITEMS_PER_CATEGORY)
            self.data_loader_registers['michigan_test'] = dataset

        indicates = torch.arange(len(dataset)).type(torch.int).cuda()
        pairs = torch.combinations(indicates, r=2, with_replacement=True)
        del indicates

        sampler_val = DistributedIndicatesSampler(pairs[:, 0].cpu(), num_replicas=self.world_size, rank=self.rank)
        x1_dataloader = DataLoader(
            dataset, sampler=sampler_val,
            batch_size=self.config.DATA.BATCH_SIZE,
            shuffle=False,  # Very important, shuffle have to be False to maintain the order of the sample indexes
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )
        predicts = torch.zeros((0, 3), dtype=torch.float32).cuda()
        is_finished = False
        tmp_data_path = os.path.join(self.config.OUTPUT, f'{split.value}_result_rank{self.rank}.pt')
        if os.path.exists(tmp_data_path):
            if remove_cache_file:
                os.unlink(tmp_data_path)
            else:
                data = torch.load(tmp_data_path)
                predicts, is_finished = data['predicts'], data['is_finished']

        if not is_finished:
            batch_time = AverageMeter()
            for x1_idx, (x1, x1_indexes) in enumerate(x1_dataloader):
                x1_lower_bound, x1_upper_bound = x1_indexes[0], x1_indexes[-1]
                if len(predicts) > 0 and x1_upper_bound <= predicts[-1][0] and x1_lower_bound >= predicts[0][0]:
                    self.logger.info(f'Block {x1_lower_bound}:{x1_upper_bound} is processed, skipping...')
                    continue

                x1 = x1.cuda(non_blocking=True)
                pair_masks = torch.greater_equal(pairs[:, 0], x1_lower_bound)
                pair_masks = torch.logical_and(pair_masks, torch.less_equal(pairs[:, 0], x1_upper_bound))

                x2_dataset = MichiganTest(self.config.DATA.DATA_PATH, split, transforms=transform,
                                          samples=dataset.data, lower_bound=x1_lower_bound.item())
                self.logger.info(f'X2 dataset size: {len(x2_dataset)}, lower_bound: {x1_lower_bound}')
                x2_dataloader = DataLoader(
                    x2_dataset,
                    batch_size=self.config.DATA.BATCH_SIZE,
                    shuffle=False,  # Very important, shuffle have to be False
                    num_workers=self.config.DATA.NUM_WORKERS,
                    pin_memory=True,
                    drop_last=False
                )

                with torch.cuda.amp.autocast(enabled=self.config.AMP_ENABLE):
                    x1 = self.model(x1, forward_first_part=True)

                x1_pairs = pairs[pair_masks]
                end = time.time()
                for x2_id, (x2, x2_indicates) in enumerate(x2_dataloader):
                    x2 = x2.cuda(non_blocking=True)
                    x2_lower_bound, x2_upper_bound = x2_indicates[0], x2_indicates[-1]
                    pair_masks = torch.greater_equal(x1_pairs[:, 1], x2_lower_bound)
                    pair_masks = torch.logical_and(pair_masks, torch.less_equal(x1_pairs[:, 1], x2_upper_bound))
                    x1_x2_pairs = x1_pairs[pair_masks]
                    x1_pairs = x1_pairs[x1_pairs[:, 1] > x2_upper_bound]
                    for sub_pairs in torch.split(x1_x2_pairs, self.config.DATA.TEST_BATCH_SIZE):
                        x1_sub = x1[sub_pairs[:, 0] - x1_lower_bound]
                        x2_sub = x2[sub_pairs[:, 1] - x2_lower_bound]
                        with torch.cuda.amp.autocast(enabled=self.config.AMP_ENABLE):
                            outputs = self.model(x1_sub, x2_sub)
                        sub_pair_predicts = torch.column_stack([sub_pairs.float(), outputs.float()])
                        predicts = torch.cat([predicts, sub_pair_predicts])
                    batch_time.update(time.time() - end)
                    end = time.time()
                    if x2_id % self.config.PRINT_FREQ == 0:
                        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                        etas = batch_time.avg * (len(x2_dataloader) - x2_id)
                        self.logger.info(
                            f'Testing: [{x1_idx}/{len(x1_dataloader)}][{x2_id}/{len(x2_dataloader)}]\t'
                            f'X2 eta {datetime.timedelta(seconds=int(etas))}\t'
                            f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                            f'mem {memory_used:.0f}MB')

                if x1_idx % self.config.SAVE_TMP_FREQ == 0 or x1_idx == len(x1_dataloader) - 1:
                    if x1_idx == len(x1_dataloader) - 1:
                        is_finished = True
                    torch.save({'predicts': predicts, 'is_finished': is_finished}, tmp_data_path)

        del predicts
        torch.cuda.empty_cache()

        self.logger.info("Gathering data on all ranks...")
        self.logger.warning("Warning, this works only when all the ranks use the same file system! " 
                         "Otherwise, this will hang forever...!")

        if split.is_val():
            # In evaluation mode, we expect that the waiting time between processes is not too large
            # Hence, we use torch.distributed.barrier() here.
            torch.distributed.barrier()

        predicts = []
        for i in range(self.world_size):
            self.logger.info(f"Waiting for rank {i}.")
            rank_data_path = os.path.join(self.config.OUTPUT, f'{split.value}_result_rank{i}.pt')
            while not os.path.exists(rank_data_path):
                time.sleep(120)

            rank_i_finished = False
            while not rank_i_finished:
                data = torch.load(rank_data_path, map_location='cpu')
                rank_predicts, rank_i_finished = data['predicts'], data['is_finished']
                if not rank_i_finished:
                    time.sleep(120)
                else:
                    predicts.append(rank_predicts)

        predicts = torch.cat(predicts, dim=0)

        self.logger.info(f"Generating similarity map...")
        assert len(predicts) == len(pairs), f'Incorrect size {predicts.shape} vs {pairs.shape}'

        size = len(dataset.data)

        # Initialize a similarity matrix with zeros
        similarity_matrix = torch.zeros((size, size), dtype=torch.float16)

        # Extract index pairs and scores
        indices = predicts[:, :2].long()
        scores = predicts[:, 2].type(torch.float16)

        # Use indexing and broadcasting to fill the similarity matrix
        similarity_matrix[indices[:, 0], indices[:, 1]] = scores
        similarity_matrix[indices[:, 1], indices[:, 0]] = scores

        self.logger.info(f"Converting to distance matrix...")
        # max_score = torch.amax(similarity_matrix, dim=1)
        distance_matrix = 1 - similarity_matrix

        self.logger.info("Distance matrix is generated!")
        return distance_matrix.numpy(), dataset.data_labels

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.geshaem_test()
        distance_matrix, labels = self.validate_dataloader(MichiganTest.Split.VAL, remove_cache_file=True)
        m_ap, top1, pr_k10, pr_k100 = wi19_evaluate.get_metrics(distance_matrix, np.asarray(labels))
        self.logger.info(f'Michigan eval: mAP {m_ap:.3f}\t' f'Top 1 {top1:.3f}\t' f'Pr@k10 {pr_k10:.3f}\t'
                         f'Pr@k100 {pr_k100:.3f}')
        return 1 - m_ap


if __name__ == '__main__':
    args, _ = parse_option()
    trainer = HisfragTrainer(args)
    if args.mode == 'eval':
        trainer.validate()
    elif args.mode == 'test':
        trainer.geshaem_test()
    elif args == 'throughput':
        trainer.throughput()
    else:
        trainer.train()
