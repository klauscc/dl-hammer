__all__ = ['setup_checkpointer']

import os
import os.path as osp

import torch


class Checkpointer(object):

    def __init__(self, cfg, phase):

        # Load pretrained checkpoint
        if not hasattr(cfg.MODEL, 'CHECKPOINT'):
            cfg.MODEL.CHECKPOINT = ''
        self.checkpoint = self._load_checkpoint(cfg.MODEL.CHECKPOINT)
        if self.checkpoint is not None:
            cfg.SOLVER.START_EPOCH = self.checkpoint.get(
                'epoch', 0) + 1 if 'train' in cfg.SOLVER.PHASES and phase == 'train' else self.checkpoint.get(
                    'epoch', 0)
        elif self.checkpoint is None and phase != 'train':
            raise RuntimeError('Cannot find checkpoint {}'.format(cfg.MODEL.CHECKPOINT))

        self.output_dir = osp.join(cfg.WORKSPACE, 'ckpts')
        os.makedirs(self.output_dir, exist_ok=True)

    def load_models(self, models):
        if self.checkpoint is None:
            return
        # load models
        if isinstance(models, dict):
            for name, model in models.items():
                model.load_state_dict(self.checkpoint[f'model_{name}_state_dict'])
        else:
            models.load_state_dict(self.checkpoint['model_state_dict'])

    def load_optimizers(self, optimizers):
        if self.checkpoint is None:
            return
        # load optimizers
        if optimizers is None:
            return
        if isinstance(optimizers, dict):
            for name, optimizer in optimizers.items():
                optimizer.load_state_dict(self.checkpoint[f'optimizer_{name}_state_dict'])
        else:
            optimizers.load_state_dict(self.checkpoint['optimizer_state_dict'])

    def load(self, models, optimizers=None):
        """
        Args:
            models (nn.Module, Dict(nn.Module)) :  model or nested model.
            optimizers (Optimizer, Dict(Optimizer)) :  optimizer or nested optimizer.
        """
        if self.checkpoint is None:
            return

        # load models
        if isinstance(models, dict):
            for name, model in models.items():
                model.load_state_dict(self.checkpoint[f'model_{name}_state_dict'])
        else:
            models.load_state_dict(self.checkpoint['model_state_dict'])

        # load optimizers
        if optimizers is None:
            return
        if isinstance(optimizers, dict):
            for name, optimizer in optimizers.items():
                optimizer.load_state_dict(self.checkpoint[f'optimizer_{name}_state_dict'])
        else:
            optimizers.load_state_dict(self.checkpoint['optimizer_state_dict'])

    def save(self, epoch, models, optimizers):
        """
        Args:
            models (nn.Module, Dict(nn.Module)) :  model or nested model.
            optimizers (Optimizer, Dict(Optimizer)) :  optimizer or nested optimizer.
        """
        save_dict = {'epoch': epoch}

        # save models
        get_model_state_dict = lambda model: model.module.state_dict() if hasattr(model, 'module'
                                                                                 ) else model.state_dict()
        if isinstance(models, dict):
            save_dict.update(
                {f'model_{name}_state_dict': get_model_state_dict(model) for name, model in models.items()})
        else:
            save_dict.update({'model_state_dict': get_model_state_dict(models)})

        # save optimizers
        if isinstance(optimizers, dict):
            save_dict.update({
                f'optimizer_{name}_state_dict': optimizer.state_dict()
                for name, optimizer in optimizers.items()
            })
        else:
            save_dict.update({f'optimizer_state_dict': optimizers.state_dict()})

        # do save
        torch.save(save_dict, osp.join(self.output_dir, 'epoch-{}.pth'.format(epoch)))

    def _load_checkpoint(self, checkpoint):
        if checkpoint != '' and not osp.isfile(checkpoint):
            raise ValueError(f'checkpoint not exist.Path:{checkpoint}')
        if osp.isfile(checkpoint):
            print(f'load checkpoint from {checkpoint}')
            return torch.load(checkpoint, map_location=torch.device('cpu'))
        return None


def setup_checkpointer(cfg, phase):
    return Checkpointer(cfg, phase)
