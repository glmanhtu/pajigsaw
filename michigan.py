import argparse
import datetime
import os
import time

import albumentations as A
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from pytorch_metric_learning import samplers
from torch.utils.data import DataLoader

from data.build import build_dataset
from data.datasets.geshaem_dataset import GeshaemPatch
from data.datasets.hisfrag_dataset import HisFrag20Test
from data.datasets.michigan_dataset import MichiganTest
from data.samplers import DistributedIndicatesSampler, DistributedEvalSampler
from data.transforms import ACompose, PadCenterCrop
from misc import wi19_evaluate, utils
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
        return torch.nn.BCEWithLogitsLoss()

    def get_transforms(self):
        patch_size = self.config.DATA.IMG_SIZE

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomAffine(5, translate=(0.1, 0.1), fill=(255, 255, 255)),
            ACompose([
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5, value=(255, 255, 255),
                                   border_mode=cv2.BORDER_CONSTANT),
            ]),
            torchvision.transforms.RandomCrop(patch_size, pad_if_needed=True, fill=(255, 255, 255)),
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
            PadCenterCrop((patch_size, patch_size), pad_if_needed=True, fill=(255, 255, 255)),
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

        transforms = self.get_transforms()
        dataset, repeat = build_dataset(mode=mode, config=self.config, transforms=transforms)

        max_dataset_length = len(dataset) * repeat
        sampler = samplers.MPerClassSampler(dataset.data_labels, m=3, length_before_new_iter=max_dataset_length)
        sampler.set_epoch = lambda x: x
        dataloader = DataLoader(dataset, sampler=sampler, pin_memory=True, batch_size=self.config.DATA.BATCH_SIZE,
                                drop_last=True, num_workers=self.config.DATA.NUM_WORKERS)

        self.data_loader_registers[mode] = dataloader
        return dataloader

    def prepare_data(self, samples, targets):
        n = samples.size(0)
        # split the positive and negative pairs
        eyes_ = torch.eye(n, dtype=torch.bool).cuda()
        pos_mask = targets.expand(
            targets.shape[0], n
        ).t() == targets.expand(n, targets.shape[0])
        neg_mask = ~pos_mask
        pos_mask[:, :n] = pos_mask[:, :n] * ~eyes_

        pos_groups, neg_groups = [], []
        for i in range(n):
            it = torch.tensor([i], device=samples.device)
            pos_pair_idx = torch.nonzero(pos_mask[i, i:]).view(-1)
            if pos_pair_idx.shape[0] > 0:
                combinations = get_combinations(it, pos_pair_idx + i)
                pos_groups.append(combinations)

            neg_pair_idx = torch.nonzero(neg_mask[i, i:]).view(-1)
            if neg_pair_idx.shape[0] > 0:
                combinations = get_combinations(it, neg_pair_idx + i)
                neg_groups.append(combinations)

        pos_groups = torch.cat(pos_groups, dim=0)
        neg_groups = torch.cat(neg_groups, dim=0)

        neg_length = min(neg_groups.shape[0], int(2 * pos_groups.shape[0]))
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
    def geshaem_test(self):
        self.model.eval()
        patch_size = self.config.DATA.IMG_SIZE
        transform = self.get_transforms()['validation']
        dataset = GeshaemPatch(args.geshaem_data_path, GeshaemPatch.Split.VAL, transform=transform, im_size=patch_size)
        sampler = DistributedEvalSampler(dataset, num_replicas=self.world_size, rank=self.rank)
        dataloader = torch.utils.data.DataLoader(
            dataset, sampler=sampler,
            batch_size=self.config.DATA.TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.DATA.NUM_WORKERS,
            pin_memory=True,
            drop_last=False
        )

        predicts = torch.zeros((0, 3), dtype=torch.float32).cuda()

        batch_time = AverageMeter()
        end = time.time()
        for idx, (images, pair) in enumerate(dataloader):
            images = images.cuda(non_blocking=True)
            with torch.cuda.amp.autocast(enabled=self.config.AMP_ENABLE):
                output = self.model(images)

            sub_pair_predicts = torch.column_stack([pair.cuda().float(), output.float()])
            predicts = torch.cat([predicts, sub_pair_predicts])
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

        if self.world_size > 1:
            max_n_items = int(len(dataset.pairs) * 1.2 / self.world_size)
            # create an empty list we will use to hold the gathered values
            predicts_list = [torch.zeros((max_n_items, 3), dtype=torch.float32, device=predicts.device)
                             for _ in range(self.world_size)]
            # Tensors from different processes have to have the same N items, therefore we pad it with -1
            predicts = F.pad(predicts, pad=(0, 0, 0, max_n_items - predicts.shape[0]), mode="constant", value=-1)

            # sending all tensors to the others
            torch.distributed.barrier()
            torch.distributed.all_gather(predicts_list, predicts, async_op=False)

            # Remove all padded items
            predicts_list = [x[x[:, 0] != -1] for x in predicts_list]
            predicts = torch.cat(predicts_list, dim=0)

        predicts = predicts.cpu()

        assert len(predicts) == len(dataset.pairs), f'Incorrect size {predicts.shape} vs {dataset.pairs.shape}'
        size = len(dataset.data)

        # Initialize a similarity matrix with zeros
        similarity_matrix = torch.zeros((size, size), dtype=torch.float16)

        # Extract index pairs and scores
        indices = predicts[:, :2].long()
        scores = predicts[:, 2].type(torch.float16)

        # Use indexing and broadcasting to fill the similarity matrix
        similarity_matrix[indices[:, 0], indices[:, 1]] = scores
        similarity_matrix[indices[:, 1], indices[:, 0]] = scores

        # max_score = torch.amax(similarity_matrix, dim=1)
        distance_matrix = 1 - similarity_matrix

        distance_matrix = distance_matrix.cpu().numpy()
        labels = dataset.data_labels
        distance_map = {}
        for i, i_val in enumerate(labels):
            for j, j_val in enumerate(labels):
                distance_map.setdefault(i_val, {}).setdefault(j_val, []).append(distance_matrix[i][j])
                distance_map.setdefault(j_val, {}).setdefault(i_val, []).append(distance_matrix[i][j])

        for source in distance_map.keys():
            for dest in distance_map[source].keys():
                distance_map[source][dest] = sum(distance_map[source][dest]) / len(distance_map[source][dest])

        matrix = pd.DataFrame.from_dict(distance_map, orient='index').sort_index()
        distance_df = matrix.reindex(sorted(matrix.columns), axis=1)

        self.logger.info(f'N samples: {len(distance_df)}, N categories: {len(distance_df.columns)}')
        index_to_fragment = {i: x for i, x in enumerate(dataset.fragments)}
        distance_df.rename(columns=index_to_fragment, index=index_to_fragment, inplace=True)

        positive_pairs = dataset.fragment_to_group

        distance_mat = distance_df.to_numpy()
        m_ap, (top_1, prk5, prk10) = calc_map_prak(distance_mat, distance_df.columns, positive_pairs, prak=(1, 5, 10))

        self.logger.info(f'Geshaem test: mAP {m_ap:.3f}\t' f'Top 1 {top_1:.3f}\t' f'Pr@k5 {prk5:.3f}\t'
                         f'Pr@k10 {prk10:.3f}')

    def validate_dataloader(self, split: MichiganTest.Split, remove_cache_file=False):
        self.model.eval()
        transform = self.get_transforms()[split.value]
        img_size = self.config.DATA.IMG_SIZE
        dataset = MichiganTest(self.config.DATA.DATA_PATH, split, transforms=transform, im_size=img_size,
                               val_n_items_per_writer=self.config.DATA.EVAL_N_ITEMS_PER_CATEGORY)
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

                x2_dataset = MichiganTest(self.config.DATA.DATA_PATH, split, transforms=transform, im_size=img_size,
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
