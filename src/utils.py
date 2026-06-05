# -*- coding: utf-8 -*-
# Shared IRT (3PL) utility functions used across all DQN/DRQN/DBQN training scripts.

from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar


def RESPOND(item_para, theta, D=1):
    a = item_para[:, 0]
    b = item_para[:, 1]
    c = item_para[:, 2]
    p = (1 - c) / (1 + np.exp(-D * a * (theta - b))) + c
    resp = (np.random.rand(1) <= p).astype(int)
    return resp


def FI(item_para, theta, D=1):
    a = item_para[:, 0]
    b = item_para[:, 1]
    c = item_para[:, 2]
    info = (
        D**2
        * a**2
        * (1 - c)
        / (c + np.exp(D * a * (theta - b)))
        / (1 + np.exp(-D * a * (theta - b))) ** 2
    )
    return info


def MLE(item_paras, resp, D=1):
    a = item_paras[:, 0]
    b = item_paras[:, 1]
    c = item_paras[:, 2]

    def mins_likelihood(x):
        logl = 0
        for i in range(len(resp)):
            p = (1 - c[i]) / (1 + np.exp(-D * a[i] * (x - b[i]))) + c[i]
            p = np.clip(p, 1e-10, 1 - 1e-10)
            logl -= resp[i] * np.log(p) + (1 - resp[i]) * np.log((1 - p))
        return logl

    result = cast(Any, minimize_scalar(mins_likelihood, bounds=(-4, 4), method="bounded"))
    theta = result.x
    return np.array(theta).reshape(1,)


def MLE_TEST(item_paras, resp, D=1):
    def mins_likelihood(x):
        logl = 0
        for i in range(resp_i.shape[0]):
            p = (1 - c[i]) / (1 + np.exp(-D * a[i] * (x - b[i]))) + c[i]
            p = np.clip(p, 1e-10, 1 - 1e-10)
            logl -= resp_i[i] * np.log(p) + (1 - resp_i[i]) * np.log((1 - p))
        return logl

    theta = np.zeros(resp.shape[1])
    for i in range(resp.shape[1]):
        resp_i = resp[:, i]
        a = item_paras[:, i, 0]
        b = item_paras[:, i, 1]
        c = item_paras[:, i, 2]
        result = cast(
            Any, minimize_scalar(mins_likelihood, bounds=(-4, 4), method="bounded")
        )
        theta[i] = result.x
    return np.expand_dims(theta, axis=0)


def Apply_Positive_Constraint(model, min_value=0.0):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight.data = torch.clamp(module.weight.data, min=min_value)
            if module.bias is not None:
                module.bias.data = torch.clamp(module.bias.data, min=min_value)
