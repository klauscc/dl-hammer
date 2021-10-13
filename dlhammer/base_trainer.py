# -*- coding: utf-8 -*-
#================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
#================================================================
import time
from tqdm import tqdm
import torch
from torch.utils.data.distributed import DistributedSampler

from core.metrics.metrics import AverageMeter

from .utils.nested import nested_call, nested_to_device


class BaseTrainer(object):
    """basic trainer"""

    def __init__(self,
                    cfg,
                    data_loaders,
                    models,
                    criterions,
                    optimizers,
                    schedulers,
                    checkpointer,
                    phases,
                    device,
                    local_rank=0,
                    writer=None,
                    logger=None):
        super(BaseTrainer, self).__init__()
        self.cfg = cfg
        self.models = models
        self.data_loaders = data_loaders
        self.criterions = criterions
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.checkpointer = checkpointer
        self.phases = phases
        self.device = device

        self.writer = writer
        self.logger = logger
        self.local_rank = local_rank

        self.epoch_logs = {phase: {} for phase in phases}

        try:
            self.world_size = cfg.DDP.WORLD_SIZE
        except Exception as e:
            self.world_size = 1

    def train_step(self, batch, batch_idx, global_step):
        pass

    def test_step(self, batch, batch_idx, global_step):
        pass

    def on_train_epoch_start(self, epoch):
        pass

    def on_train_epoch_end(self, epoch):
        pass

    def on_test_epoch_start(self, epoch):
        pass

    def on_test_epoch_end(self, epoch):
        pass

    def forward(self, x):
        raise NotImplemented('not implemented')

    def train(self, evaluate_freq=1):
        """train function

        Kwargs:
            evaluate_freq (int): evaluate frequence (epoch). Default 1. If -1, no evaluation.

        Returns: TODO

        """
        for epoch in range(self.cfg.SOLVER.START_EPOCH, self.cfg.SOLVER.NUM_EPOCHS + 1):

            # train
            losses = {phase: AverageMeter() for phase in self.phases}

            for phase in self.phases:
                training = phase == 'train'

                if (not training) and epoch % evaluate_freq != 0:
                    continue

                start = time.time()

                # on epoch start
                ## clean epoch_logs
                self.epoch_logs[phase] = {}

                ## set model and sampler
                if training:
                    nested_call(self.models, 'train')
                    ## set epoch to samplers
                    for data_loader in self.data_loaders.values():
                        sampler = data_loader.sampler
                        sampler.set_epoch(epoch) if sampler is not None and isinstance(
                            sampler, DistributedSampler) else None
                else:
                    nested_call(self.models, 'eval')
                self.on_train_epoch_start(epoch) if training else self.on_test_epoch_start(epoch)

                # epoch
                step_fn = self.train_step if training else self.test_step
                with torch.set_grad_enabled(training):
                    if self.local_rank == 0:
                        pbar = tqdm(self.data_loaders[phase],
                                    desc='{}ing epoch {}'.format(phase.capitalize(), epoch))
                    else:
                        pbar = self.data_loaders[phase]

                    # train batches
                    for batch_idx, batch in enumerate(pbar, start=1):
                        global_step = (epoch - 1) * len(self.data_loaders[phase]) + batch_idx

                        batch = nested_to_device(batch, self.device, non_blocking=True)

                        loss, batch_logs = step_fn(batch, batch_idx, global_step)

                        losses[phase].update(loss.item())

                        # log batch
                        if self.local_rank == 0:
                            self.log_batch(loss, batch_logs, pbar, global_step, phase)

                # on epoch end
                self.on_train_epoch_end(epoch) if training else self.on_test_epoch_end(epoch)

                end = time.time()

                epoch_loss = losses[phase].compute()
                self.epoch_logs[phase].update({'loss': epoch_loss})

                # log epoch
                if self.local_rank == 0:
                    log = f'Epoch {epoch:03d}'
                    log += f' | {phase.capitalize()}'
                    log += f' | {self.epoch_logs[phase]}'
                    log += f' | Time cost:{end-start} sec'
                    self.logger.info(log)

                    if self.writer is not None:
                        for k, v in self.epoch_logs[phase].items():
                            self.writer.add_scalar('epoch/' + k, v, global_step=epoch)

                    #save checkpoint
                    if phase == 'train':
                        self.checkpointer.save(epoch, self.models, self.optimizers)

    def log_batch(self, loss, batch_logs, pbar, global_step, phase):
        """log batch metrics.

        Args:
            loss (float): The loss.
            batch_logs (dict): The batch logs to show in pbar
            pbar (tqdm): progress bar
            global_step (int): global_step
            phase (string): The current phase.
            training (bool): whether is training

        Returns: TODO

        """
        training = phase == 'train'
        batch_logs[f'{phase}/loss'] = loss.item()
        pbar.set_postfix(batch_logs)

        ## log loss and lr to tensorboard
        if self.writer is not None:
            logs = batch_logs

            if training and self.schedulers is not None:
                lrs = nested_call(self.schedulers, 'get_last_lr')
                if isinstance(lrs, dict):
                    logs.update({'learning_rate/' + k: v[0] for k, v in lrs.items()})
                else:
                    logs['learning_rate'] = lrs[0]

            for k, v in logs.items():
                self.writer.add_scalar(k, v, global_step=global_step)
