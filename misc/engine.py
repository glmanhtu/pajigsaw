import datetime
import json
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from config import get_config
from data.build import build_dataset
from data.samplers import DistributedRepeatSampler, DistributedEvalSampler
from data.transforms import TwoImgSyncEval
from misc import utils
from misc.logger import create_logger
from misc.lr_scheduler import build_scheduler
from misc.optimizer import build_optimizer
from misc.utils import configure_ddp, NativeScalerWithGradNormCount, auto_resume_helper, load_checkpoint, \
    load_pretrained, AverageMeter, save_checkpoint
from models import build_model


class Trainer:
    def __init__(self, args):
        self.config = get_config(args)
        self.local_rank, self.rank, self.world_size = configure_ddp()
        seed = self.config.SEED + self.rank
        utils.set_seed(seed)
        cudnn.benchmark = True

        # linear scale the learning rate according to total batch size, may not be optimal
        batch_size = self.config.DATA.BATCH_SIZE * dist.get_world_size()
        linear_scaled_lr = self.config.TRAIN.BASE_LR * batch_size / 256.0
        linear_scaled_warmup_lr = self.config.TRAIN.WARMUP_LR * batch_size / 256.0
        linear_scaled_min_lr = self.config.TRAIN.MIN_LR * batch_size / 256.0

        # gradient accumulation also need to scale the learning rate
        if self.config.TRAIN.ACCUMULATION_STEPS > 1:
            linear_scaled_lr = linear_scaled_lr * self.config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_warmup_lr = linear_scaled_warmup_lr * self.config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_min_lr = linear_scaled_min_lr * self.config.TRAIN.ACCUMULATION_STEPS
        self.config.defrost()
        self.config.TRAIN.BASE_LR = linear_scaled_lr
        self.config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        self.config.TRAIN.MIN_LR = linear_scaled_min_lr
        self.config.freeze()

        os.makedirs(self.config.OUTPUT, exist_ok=True)
        logger = create_logger(output_dir=self.config.OUTPUT, dist_rank=self.rank,
                               name=f"{self.config.MODEL.NAME}", affix=args.mode)

        if dist.get_rank() == 0:
            path = os.path.join(self.config.OUTPUT, "config.json")
            with open(path, "w") as f:
                f.write(self.config.dump())
            logger.info(f"Full config saved to {path}")

        # print config
        logger.info(self.config.dump())
        logger.info(json.dumps(vars(args)))

        logger.info(f"Creating model:{self.config.MODEL.TYPE}/{self.config.MODEL.NAME}")
        model = build_model(self.config)
        logger.info(str(model))

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"number of params: {n_parameters}")
        if hasattr(model, 'flops'):
            flops = model.flops()
            logger.info(f"number of GFLOPs: {flops / 1e9}")

        model.cuda()
        model_wo_ddp = model
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank], broadcast_buffers=False)

        self.min_loss = 99999
        self.model = model
        self.model_wo_ddp = model_wo_ddp
        self.logger = logger

        if self.config.TRAIN.AUTO_RESUME:
            resume_file = auto_resume_helper(self.config.OUTPUT)
            if resume_file:
                if self.config.MODEL.RESUME:
                    self.logger.warning(f"Auto-resume changing resume file "
                                        f"from {self.config.MODEL.RESUME} to {resume_file}")
                self.config.defrost()
                self.config.MODEL.RESUME = resume_file
                self.config.freeze()
                self.logger.info(f'Auto resuming from {resume_file}')
            else:
                self.logger.info(f'No checkpoint found in {self.config.OUTPUT}, ignoring auto resume')

        if self.config.MODEL.PRETRAINED and (not self.config.MODEL.RESUME):
            load_pretrained(self.config, model_wo_ddp, logger)

        self.data_loader_registers = {}

    def get_transforms(self):
        patch_size = self.config.DATA.IMG_SIZE
        transform = TwoImgSyncEval(patch_size)
        return {
            'train': transform,
            'validation': transform,
            'test': transform
        }

    def get_dataloader(self, mode):
        if mode in self.data_loader_registers:
            return self.data_loader_registers[mode]
        transforms = self.get_transforms()
        dataset, repeat = build_dataset(mode=mode, config=self.config, transforms=transforms)
        print(f"local rank {self.local_rank} / global rank {self.rank} successfully build {mode} dataset")

        num_tasks = self.world_size
        global_rank = self.rank
        if mode == 'train':
            sampler = DistributedRepeatSampler(
                dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True, repeat=repeat
            )
            data_loader = DataLoader(
                dataset, sampler=sampler,
                batch_size=self.config.DATA.BATCH_SIZE,
                num_workers=self.config.DATA.NUM_WORKERS,
                pin_memory=self.config.DATA.PIN_MEMORY,
                drop_last=True,
            )
        else:
            sampler = DistributedEvalSampler(
                dataset, shuffle=self.config.TEST.SHUFFLE, rank=global_rank, num_replicas=num_tasks, repeat=repeat
            )

            data_loader = torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                batch_size=self.config.DATA.TEST_BATCH_SIZE,
                shuffle=False,
                num_workers=self.config.DATA.NUM_WORKERS,
                pin_memory=self.config.DATA.PIN_MEMORY,
                drop_last=False
            )
        self.data_loader_registers[mode] = data_loader
        return data_loader

    def train(self):
        data_loader = self.get_dataloader('train')
        optimizer = build_optimizer(self.config, self.model_wo_ddp)
        loss_scaler = NativeScalerWithGradNormCount()
        lr_scheduler = build_scheduler(self.config, optimizer, len(data_loader) // self.config.TRAIN.ACCUMULATION_STEPS)
        criterion = self.get_criterion()

        if self.config.MODEL.RESUME:
            min_loss = load_checkpoint(self.config, self.model_wo_ddp, optimizer, lr_scheduler, loss_scaler, self.logger)
            self.logger.info(f"Model resuming success, starting to evaluate...")
            loss = self.validate()
            self.min_loss = min(loss, min_loss)
            self.logger.info(f"Loss of the network on the val set: {loss:.4f}")

        self.logger.info("Start training...")
        config = self.config
        start_time = time.time()
        loss = self.validate()
        self.logger.info(f"Init loss: {loss}")
        for epoch in range(self.config.TRAIN.START_EPOCH, self.config.TRAIN.EPOCHS):
            self.train_one_epoch(epoch, data_loader, optimizer, lr_scheduler, loss_scaler, criterion)

            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, self.model_wo_ddp, self.min_loss, optimizer, lr_scheduler, loss_scaler,
                                self.logger, 'checkpoint')

            loss = self.validate()
            if loss < self.min_loss:
                save_checkpoint(config, epoch, self.model_wo_ddp, self.min_loss, optimizer, lr_scheduler, loss_scaler,
                                self.logger, 'best_model')
                self.logger.info(f"Loss is reduced from {self.min_loss} to {loss}")

            self.min_loss = min(self.min_loss, loss)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info('Training time {}'.format(total_time_str))

    def train_step(self, samples):
        return self.model(samples)

    def prepare_data(self, samples, targets):
        return samples, targets

    def train_one_epoch(self, epoch, data_loader, optimizer, lr_scheduler, loss_scaler, criterion):
        self.model.train()
        optimizer.zero_grad()
        # data_loader.sampler.set_epoch(epoch)
        num_steps = len(data_loader)
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        scaler_meter = AverageMeter()

        start = time.time()
        end = time.time()
        optimizer.zero_grad()
        for idx, (samples, targets) in enumerate(data_loader):
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            samples, targets = self.prepare_data(samples, targets)

            with torch.cuda.amp.autocast(enabled=self.config.AMP_ENABLE):
                outputs = self.train_step(samples)

                loss = criterion(outputs, targets)
                loss = loss / self.config.TRAIN.ACCUMULATION_STEPS

            if self.config.AMP_ENABLE:
                # this attribute is added by timm on one optimizer (adahessian)
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                grad_norm = loss_scaler(loss, optimizer, clip_grad=self.config.TRAIN.CLIP_GRAD,
                                        parameters=self.model.parameters(), create_graph=is_second_order,
                                        update_grad=(idx + 1) % self.config.TRAIN.ACCUMULATION_STEPS == 0)
                loss_scale_value = loss_scaler.state_dict()["scale"]
                if grad_norm is not None:  # loss_scaler return None if not update
                    norm_meter.update(grad_norm)
                scaler_meter.update(loss_scale_value)
            else:
                loss.backward()

            if (idx + 1) % self.config.TRAIN.ACCUMULATION_STEPS == 0:
                lr_scheduler.step_update((epoch * num_steps + idx) // self.config.TRAIN.ACCUMULATION_STEPS)
                if not self.config.AMP_ENABLE:
                    optimizer.step()
                optimizer.zero_grad()

            torch.cuda.synchronize()

            loss_meter.update(loss.item() * self.config.TRAIN.ACCUMULATION_STEPS, targets.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % self.config.PRINT_FREQ == 0:
                lr = optimizer.param_groups[0]['lr']
                wd = optimizer.param_groups[0]['weight_decay']
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - idx)
                self.logger.info(
                    f'Train: [{epoch}/{self.config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')

        epoch_time = time.time() - start
        self.logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

        loss_meter.all_reduce()
        return loss_meter.avg

    def get_criterion(self):
        raise NotImplementedError()

    @torch.no_grad()
    def validate(self):
        raise NotImplementedError()

    def throughput(self):
        self.model.eval()
        data_loader = self.get_dataloader('validation')
        for idx, (images, _) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            batch_size = images.shape[0]
            for i in range(50):
                self.model(images)
            torch.cuda.synchronize()
            self.logger.info(f"throughput averaged with 30 times")
            tic1 = time.time()
            for i in range(30):
                self.model(images)
            torch.cuda.synchronize()
            tic2 = time.time()
            throughput_val = 30 * batch_size / (tic2 - tic1)
            self.logger.info(f"batch_size {batch_size} throughput {throughput_val}")
            return throughput_val
