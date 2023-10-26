import argparse
import datetime
import time

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from misc.engine import Trainer
from misc.utils import AverageMeter


def parse_option():
    parser = argparse.ArgumentParser('Pajigsaw training and evaluation script', add_help=False)
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
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'throughput'], default='train')

    # overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    return parser.parse_known_args()


class DefaultTrainer(Trainer):

    def get_criterion(self):
        return torch.nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        data_loader = self.get_dataloader('validation')
        criterion = self.get_criterion()
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        f1_meter = AverageMeter()
        precision_meter = AverageMeter()
        recall_meter = AverageMeter()

        start = time.time()
        end = time.time()
        for idx, (images, target) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=self.config.AMP_ENABLE):
                output = self.model(images)

            loss = criterion(output, target)

            outputs = torch.unbind(output.cpu(), dim=1)
            targets = torch.unbind(target.cpu(), dim=1)

            accuracies, f1s = [], []
            precisions, recalls = [], []
            for out, y in zip(outputs, targets):
                pred, gt = (out > 0).float().numpy(), y.numpy()
                acc = accuracy_score(gt, pred) * 100
                f1 = f1_score(gt, pred, average="macro")
                precision = precision_score(gt, pred, average="macro")
                recall = recall_score(gt, pred, average="macro")
                accuracies.append(acc)
                f1s.append(f1)
                precisions.append(precision)
                recalls.append(recall)

            acc = sum(accuracies) / len(accuracies)
            loss_meter.update(loss.item(), target.size(0))
            acc_meter.update(acc, target.size(0))
            f1_meter.update(sum(f1s) / len(f1s), target.size(0))
            precision_meter.update(sum(precisions) / len(precisions), target.size(0))
            recall_meter.update(sum(recalls) / len(recalls), target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % self.config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                self.logger.info(
                    f'Eval: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'ACC {acc_meter.val:.3f} ({acc_meter.avg:.3f})\t'
                    f'F1 {f1_meter.val:.3f} ({f1_meter.avg:.3f})\t'
                    f'Precision {precision_meter.val:.3f} ({precision_meter.avg:.3f})\t'
                    f'Recall {recall_meter.val:.3f} ({recall_meter.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')

        # Gathering results from gpus
        torch.distributed.barrier()
        loss_meter.all_reduce()
        acc_meter.all_reduce()
        f1_meter.all_reduce()
        batch_time.all_reduce()
        precision_meter.all_reduce()
        recall_meter.all_reduce()
        test_time = datetime.timedelta(seconds=int(time.time() - start))

        self.logger.info(
            f'Overall:'
            f'Time {test_time}\t'
            f'Batch Time {batch_time.avg:.3f}\t'
            f'Loss {loss_meter.avg:.4f}\t'
            f'ACC {acc_meter.avg:.3f}\t'
            f'F1 {f1_meter.avg:.3f}\t'
            f'Precision {precision_meter.avg:.3f}\t'
            f'Recall {recall_meter.avg:.3f}')

        return loss_meter.avg


if __name__ == '__main__':
    args, _ = parse_option()
    trainer = DefaultTrainer(args)
    if args.mode == 'eval':
        trainer.validate()
    elif args.mode == 'throughput':
        trainer.throughput()
    else:
        trainer.train()
