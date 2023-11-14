import argparse
import datetime
import os
import time

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader

from data.datasets.hisfrag_dataset import HisFrag20Test
from data.samplers import DistributedIndicatesSampler
from misc import wi19_evaluate, utils
from misc.engine import Trainer
from misc.utils import AverageMeter


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


class HisfragTrainer(Trainer):

    def get_criterion(self):
        return torch.nn.BCEWithLogitsLoss()

    def validate_dataloader(self, split: HisFrag20Test.Split, remove_cache_file=False):
        self.model.eval()
        transform = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(512),
            torchvision.transforms.Resize(self.config.DATA.IMG_SIZE),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = HisFrag20Test(self.config.DATA.DATA_PATH, split, transform=transform)
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

                x2_dataset = HisFrag20Test(self.config.DATA.DATA_PATH, split, transform=transform,
                                           samples=dataset.samples, lower_bound=x1_lower_bound.item())
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
        self.logger.warn("Warning, this works only when all the ranks use the same file system! " 
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
                if not is_finished:
                    time.sleep(120)
                else:
                    predicts.append(rank_predicts)

        predicts = torch.cat(predicts, dim=0)

        self.logger.info(f"Generating similarity map...")
        assert len(predicts) == len(pairs), f'Incorrect size {predicts.shape} vs {pairs.shape}'

        size = len(dataset.samples)

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

        labels = []
        for i in range(size):
            labels.append(os.path.splitext(os.path.basename(dataset.samples[i]))[0])
        self.logger.info("Distance matrix is generated!")
        return distance_matrix.numpy(), labels

    @torch.no_grad()
    def test(self):
        self.model.eval()
        distance_matrix, img_names = self.validate_dataloader(HisFrag20Test.Split.TEST)
        labels = utils.list_to_idx(img_names, lambda x: x.split('_')[0])
        m_ap, top1, pr_k10, pr_k100 = wi19_evaluate.get_metrics(distance_matrix, np.asarray(labels))
        self.logger.info(f'mAP {m_ap:.3f}\t' f'Top 1 {top1:.3f}\t' f'Pr@k10 {pr_k10:.3f}\t' f'Pr@k100 {pr_k100:.3f}')
        if self.rank == 0:
            df = pd.DataFrame(data=distance_matrix, columns=img_names, index=img_names)
            result_file = os.path.join(self.config.OUTPUT, f'distance_matrix_rank{self.rank}.csv')
            df.to_csv(result_file, index=True)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        distance_matrix, img_names = self.validate_dataloader(HisFrag20Test.Split.VAL, remove_cache_file=True)
        labels = utils.list_to_idx(img_names, lambda x: x.split('_')[0])
        m_ap, top1, pr_k10, pr_k100 = wi19_evaluate.get_metrics(distance_matrix, np.asarray(labels))
        self.logger.info(f'mAP {m_ap:.3f}\t' f'Top 1 {top1:.3f}\t' f'Pr@k10 {pr_k10:.3f}\t' f'Pr@k100 {pr_k100:.3f}')
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
