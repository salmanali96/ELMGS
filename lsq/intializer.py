import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .qnn import *


def initialize(model):
    state = model.state_dict()
    for name, module in model.named_modules():
        if type(module) in [QuantConv2d, QuantLinear]:
            init_weight_step(state, name, module)
        elif type(module) in [QuantReLU]:
            init_activation_step(state, name, module, act='relu')
        elif type(module) in [QuantReLU6]:
            init_activation_step2(state, name, module, act='act')


def init_weight_step(state, name, module):
    step = state[name + '.quantizer.step']
    do_init = state[name + '.quantizer.do_init']

    if do_init == 1:
        return

    step.data.copy_(
        2. * module.weight.abs().mean() / math.sqrt(module.q_p)
    )
    do_init.fill_(1)


def init_activation_step(state, name, module, act='relu'):
    step = state[name + '.quantizer.step']
    do_init = state[name + '.quantizer.do_init']

    if do_init == 1:
        return

    bn_weight = state[name.replace(act, 'bn') + '.weight'].abs().cpu().numpy()
    bn_bias = state[name.replace(act, 'bn') + '.bias'].cpu().numpy()
    
    # gen a virtual matrix
    nb = int(50000 / bn_weight.shape[0])
    mat = np.zeros([bn_weight.shape[0], nb])
    for i in range(bn_weight.shape[0]):
        mat[i, :] = np.random.normal(bn_bias[i], bn_weight[i], nb)
    mat = np.where(mat > 0., mat, 0.)
    
    step.fill_(
        2. * np.mean(np.absolute(mat)) / math.sqrt(module.q_p)
    )
    do_init.fill_(1)


def init_activation_step2(state, name, module, act='act'):
    step = state[name + '.quantizer.step']
    do_init = state[name + '.quantizer.do_init']

    if do_init == 1:
        return

    bn_weight = state[name.replace(act, 'bn') + '.weight'].abs().cpu().numpy()
    bn_bias = state[name.replace(act, 'bn') + '.bias'].cpu().numpy()
    
    # gen a virtual matrix
    nb = int(50000 / bn_weight.shape[0])
    mat = np.zeros([bn_weight.shape[0], nb])
    for i in range(bn_weight.shape[0]):
        mat[i, :] = np.random.normal(bn_bias[i], bn_weight[i], nb)
    mat = np.clip(mat, 0., 6.)
    
    step.fill_(
        2. * np.mean(np.absolute(mat)) / math.sqrt(module.q_p)
    )
    do_init.fill_(1)
    