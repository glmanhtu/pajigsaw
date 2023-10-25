import argparse
import datetime
import os
import time

import numpy as np
import torch

from misc import wi19_evaluate
from misc.engine import Trainer
from misc.utils import AverageMeter, compute_distance_matrix


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
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    return parser.parse_known_args()


class NegativeCosineSimilarityLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = torch.nn.CosineSimilarity(dim=1)

    def forward(self, predicts, _):
        p1, p2, z1, z2 = predicts
        loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
        return loss + 1.    # Since the loss has its ra


class GeshaemTrainer(Trainer):

    def get_criterion(self):
        return NegativeCosineSimilarityLoss()

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        batch_time = AverageMeter()
        mAP_meter = AverageMeter()
        top1_meter = AverageMeter()
        pk10_meter = AverageMeter()
        pk100_meter = AverageMeter()

        start = time.time()
        end = time.time()
        features = {}
        for idx, (images, targets) in enumerate(self.data_loader_val):
            images = images.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=self.config.AMP_ENABLE):
                _, _, z1, z2 = self.model(images)

            for feature1, feature2, target in zip(z1, z2, targets.numpy()):
                features.setdefault(target, [])
                features[target].append(feature1)
                features[target].append(feature2)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % self.config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                self.logger.info(
                    f'Eval: [{idx}/{len(self.data_loader_val)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')

        features = {k: torch.stack(v).cuda() for k, v in features.items()}
        distance_df = compute_distance_matrix(features, reduction=args.distance_reduction)
        papyrus_ids = [self.data_loader_val.dataset.get_group_id(x) for x in distance_df.index]
        distance_matrix = distance_df.to_numpy()
        m_ap, top1, pr_a_k10, pr_a_k100 = wi19_evaluate.get_metrics(distance_matrix, np.asarray(papyrus_ids))
        mAP_meter.update(m_ap)
        top1_meter.update(top1)
        pk10_meter.update(pr_a_k10)
        pk100_meter.update(pr_a_k100)
        test_time = datetime.timedelta(seconds=int(time.time() - start))

        mAP_meter.all_reduce()
        top1_meter.all_reduce()
        pk10_meter.all_reduce()
        pk100_meter.all_reduce()

        self.logger.info(
            f'Overall:'
            f'Time {test_time}\t'
            f'Batch Time {batch_time.avg:.3f}\t'
            f'mAP {mAP_meter.avg:.4f}\t'
            f'top1 {top1_meter.avg:.3f}\t'
            f'pr@k10 {pk10_meter.avg:.3f}\t'
            f'pr@k100 {pk100_meter.avg:.3f}')

        val_loss = 1 - mAP_meter.avg
        if val_loss < self.min_loss:
            similarity_df = (2 - distance_df) / 2.
            similarity_df.to_csv(os.path.join(self.config.OUTPUT, 'similarity_matrix.csv'))

        return val_loss


if __name__ == '__main__':
    args, _ = parse_option()
    trainer = GeshaemTrainer(args)
    if args.eval:
        trainer.validate()
    elif args.throughput:
        trainer.throughput()
    else:
        trainer.train()
