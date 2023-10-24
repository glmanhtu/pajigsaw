import datetime
import json
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from config import get_config
from data.build import build_loader
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
        seed = self.config.SEED + dist.get_rank()
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
        logger = create_logger(output_dir=self.config.OUTPUT, dist_rank=self.rank, name=f"{self.config.MODEL.NAME}")

        if dist.get_rank() == 0:
            path = os.path.join(self.config.OUTPUT, "config.json")
            with open(path, "w") as f:
                f.write(self.config.dump())
            logger.info(f"Full config saved to {path}")

        # print config
        logger.info(self.config.dump())
        logger.info(json.dumps(vars(args)))

        data_loader_train, data_loader_val, mixup_fn = build_loader(self.config)
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val

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

        optimizer = build_optimizer(self.config, model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank], broadcast_buffers=False)
        loss_scaler = NativeScalerWithGradNormCount()
        lr_scheduler = build_scheduler(self.config, optimizer,
                                       len(data_loader_train) // self.config.TRAIN.ACCUMULATION_STEPS)

        self.criterion = self.get_criterion()
        self.min_loss = 99999
        self.model = model
        self.model_wo_ddp = model_wo_ddp
        self.loss_scaler = loss_scaler
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.logger = logger
        self.mixup_fn = mixup_fn

        if self.config.TRAIN.AUTO_RESUME:
            resume_file = auto_resume_helper(self.config.OUTPUT)
            if resume_file:
                if self.config.MODEL.RESUME:
                    logger.warning(f"auto-resume changing resume file from {self.config.MODEL.RESUME} to {resume_file}")
                self.config.defrost()
                self.config.MODEL.RESUME = resume_file
                self.config.freeze()
                logger.info(f'auto resuming from {resume_file}')
            else:
                logger.info(f'no checkpoint found in {self.config.OUTPUT}, ignoring auto resume')

        if self.config.MODEL.RESUME:
            self.min_loss = load_checkpoint(self.config, model_wo_ddp, optimizer, lr_scheduler, loss_scaler, logger)
            loss = self.validate()
            logger.info(f"Loss of the network on the val set: {loss:.4f}")

        if self.config.MODEL.PRETRAINED and (not self.config.MODEL.RESUME):
            load_pretrained(self.config, model_wo_ddp, logger)
            loss = self.validate()
            logger.info(f"Loss of the network on the val set: {loss:.4f}")

    def train(self):
        self.logger.info("Start training...")
        config = self.config
        start_time = time.time()
        for epoch in range(self.config.TRAIN.START_EPOCH, self.config.TRAIN.EPOCHS):
            self.train_one_epoch(epoch)

            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, self.model_wo_ddp, self.min_loss, self.optimizer, self.lr_scheduler,
                                self.loss_scaler, self.logger, 'checkpoint')

            loss = self.validate()
            if loss < self.min_loss:
                save_checkpoint(config, epoch, self.model_wo_ddp, self.min_loss, self.optimizer, self.lr_scheduler,
                                self.loss_scaler, self.logger, 'best_model')
                self.logger.info(f"Loss is reduced from {self.min_loss} to {loss}")

            self.min_loss = min(self.min_loss, loss)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info('Training time {}'.format(total_time_str))

    def train_one_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        self.data_loader_train.sampler.set_epoch(epoch)
        num_steps = len(self.data_loader_train)
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        scaler_meter = AverageMeter()

        start = time.time()
        end = time.time()
        for idx, (samples, targets) in enumerate(self.data_loader_train):
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            if self.mixup_fn is not None:
                samples, targets = self.mixup_fn(samples, targets)

            with torch.cuda.amp.autocast(enabled=self.config.AMP_ENABLE):
                outputs = self.model(samples)

            loss = self.criterion(outputs, targets)
            loss = loss / self.config.TRAIN.ACCUMULATION_STEPS

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order
            grad_norm = self.loss_scaler(loss, self.optimizer, clip_grad=self.config.TRAIN.CLIP_GRAD,
                                         parameters=self.model.parameters(), create_graph=is_second_order,
                                         update_grad=(idx + 1) % self.config.TRAIN.ACCUMULATION_STEPS == 0)

            if (idx + 1) % self.config.TRAIN.ACCUMULATION_STEPS == 0:
                self.optimizer.zero_grad()
                self.lr_scheduler.step_update((epoch * num_steps + idx) // self.config.TRAIN.ACCUMULATION_STEPS)
            loss_scale_value = self.loss_scaler.state_dict()["scale"]

            torch.cuda.synchronize()

            loss_meter.update(loss.item() * self.config.TRAIN.ACCUMULATION_STEPS, targets.size(0))
            if grad_norm is not None:  # loss_scaler return None if not update
                norm_meter.update(grad_norm)

            scaler_meter.update(loss_scale_value)
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % self.config.PRINT_FREQ == 0:
                lr = self.optimizer.param_groups[0]['lr']
                wd = self.optimizer.param_groups[0]['weight_decay']
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

        for idx, (images, _) in enumerate(self.data_loader_val):
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