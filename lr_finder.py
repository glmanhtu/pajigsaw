import argparse

from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_trainer

import torch
from ignite.handlers import FastaiLRFinder

from misc.engine import Trainer
from misc.optimizer import build_optimizer


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
    parser.add_argument('--numb-iter', type=int, help="Number of iterations")
    parser.add_argument('--start-lr', type=float, default=1e-7)
    parser.add_argument('--end-lr', type=float, default=1e-2)
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--mode', type=str, choices=['lr_finder'], default='lr_finder')

    # overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    return parser.parse_known_args()


class DefaultTrainer(Trainer):

    def get_criterion(self):
        return torch.nn.BCEWithLogitsLoss()

    def find_lr(self):
        data_loader = self.get_dataloader('train')
        optimizer = build_optimizer(self.config, self.model_wo_ddp)
        criterion = self.get_criterion()
        device = next(self.model.parameters()).device
        scaler = torch.cuda.amp.GradScaler()
        model_trainer = create_supervised_trainer(self.model, optimizer, criterion, amp_mode='amp', scaler=scaler,
                                                  device=device)
        lr_finder = FastaiLRFinder()
        ProgressBar(persist=True).attach(model_trainer, output_transform=lambda x: {"batch_loss": x})
        with lr_finder.attach(model_trainer, {'optimizer': optimizer}, num_iter=args.numb_iter,
                              start_lr=args.start_lr, end_lr=args.end_lr) as trainer_with_lr_finder:
            trainer_with_lr_finder.run(data_loader)

        ax = lr_finder.plot(skip_end=0)
        ax.figure.savefig("lr_finder_result.jpg")

        lr_suggestion = lr_finder.lr_suggestion()
        self.logger.info(f"Lr suggestion: {lr_suggestion}")
        return lr_suggestion


if __name__ == '__main__':
    args, _ = parse_option()
    trainer = DefaultTrainer(args)
    trainer.find_lr()
