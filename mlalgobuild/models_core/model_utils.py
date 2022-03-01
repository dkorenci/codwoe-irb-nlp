import math
from pathlib import Path

import torch
from torch import nn as nn
from torch.nn import functional as F

from mlalgobuild.models_core.models_defmod import TransformerDefmod, RnnDefmod
from mlalgobuild.models_core.models_revdict import RevdictBase
from mlalgobuild.models_core.models_emb2emb import Embed2EmbedMLP

from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_schedule(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):
    """From Huggingface"""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def get_schedule_plateau(optimizer):
    return ReduceLROnPlateau(optimizer, factor=0.1, patience=5, threshold=0.001, cooldown=1)

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction="mean"):
    return (
        loss.mean()
        if reduction == "mean"
        else loss.sum()
        if reduction == "sum"
        else loss
    )


def get_model(model, *args, **kwargs):
    """ Get model by name or filepath. With args set as dict or filpath to hyperparameters - ToDO. """
    mlo = str(model).strip().lower()
    if mlo in MODELS.keys():
        return MODELS[mlo](*args, **kwargs)
    else:
        location = kwargs['device'] if 'device' in kwargs.keys() else 'cuda'
        print(location)
        if Path(model).is_file:
            return torch.load(model, map_location=location)
    raise ValueError("Given model does not exists: {}".format(model))

#  Implementation of Label smoothing with CrossEntropy and ignore_index
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction="mean", ignore_index=-100):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index
        )
        return linear_combination(loss / n, nll, self.epsilon)


MODELS = {
    'defmod-transformer':TransformerDefmod,
    'defmod-rnn': RnnDefmod,
    'revdict-base':RevdictBase,
    'embed2embed-mlp':Embed2EmbedMLP,
}