import os, sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn

from model.DSSM_modules.s4 import S4
from model.DSSM_modules.s4nd import S4ND

# fix the random seed
torch.manual_seed(42)
# use higher precision
torch.set_default_dtype(torch.float64)

import pytest
import itertools


def inference(
    d_model,
    d_state,
    data_shape,
    bidirectional,
    transposed,
    linear,
    model_cls=S4ND,
):

    *dims_spatial, length = data_shape

    kernel_args = {
        "mode": "diag",
        "measure": "diag-legs",  # diag-<scaling>
        "dt_min": 1,
        "dt_max": 1,
        "deterministic": False,
        "disc": "bilinear",
        "real_type": "none",
    }

    if model_cls == S4:
        model_args = {
            "bottleneck": None,
            "gate": None,
        }
    elif model_cls == S4ND:
        model_args = {
            "linear": linear,
            "contract_version": 1,
        }

    kwargs = {}
    kwargs.update(model_args)
    kwargs.update(kernel_args)

    dim = len(dims_spatial) + 1

    model = model_cls(
        d_model,
        d_state=d_state,
        l_max=None,
        dim=dim,
        channels=1,
        bidirectional=bidirectional,
        # Arguments for position-wise feedforward components
        activation=None,
        postact=None,
        initializer="xavier",
        hyper_act=None,
        dropout=0.0,
        tie_dropout=False,
        # bottleneck=None, # for S4 only
        # gate=None, # for S4 only
        transposed=transposed,
        verbose=True,
        # contract_version="not zero;)", # for S4ND only
        # SSM Kernel arguments
        **kwargs,
    )
    model.eval()

    batch_size = 1  # batch_size
    H = d_model
    # N = d_state // 2 # state size

    # input
    if isinstance(model, S4):
        if transposed:
            x = 1.0 * torch.randn(batch_size, H, length)
        else:
            x = 1.0 * torch.randn(batch_size, length, H)
        state = model.default_state(batch_size)
        state = torch.randn_like(state) * 10

    elif isinstance(model, S4ND):
        if transposed:
            x = torch.randn(batch_size, H, *dims_spatial, length)
        else:
            x = torch.randn(batch_size, *dims_spatial, length, H)
        state = model.default_state(batch_size, *dims_spatial)
        state = torch.randn_like(state) * 10

    # recurrent mode
    state_recur = state.clone()

    with torch.no_grad():

        # conv mode
        y_conv, state_conv = model.forward(x, state)

        # recurrent mode
        model.setup_step()  # call before recurrence

        y_recur = []
        for i in range(length):
            if transposed:
                x_i = x[..., i]
            else:
                x_i = x[..., i, :]

            y_i, state_recur = model.step(x_i, state_recur)

            if transposed:
                y_recur.append(y_i.unsqueeze(-1))
            else:
                y_recur.append(y_i.unsqueeze(-2))

        if transposed:
            y_recur = torch.cat(y_recur, -1)
        else:
            y_recur = torch.cat(y_recur, -2)

        # compute the relative differences between outputs of the conv and recu
        eps = 1e-16
        rdif = lambda a, b: torch.norm(a - b) / (torch.norm(b) + eps)

        diff_output = rdif(y_recur, y_conv).item()
        diff_state = rdif(state_recur, state_conv).item()
        return diff_output, diff_state


# Configuration groups

# (d_model, d_state)
model_configs = [
    (1, 4),
    (4, 4),
    (3, 6),
    (9, 8),
    (35, 64),
]

# (feature_dim1, ..., feature_dimN, time_dim)
data_configs = [
    (1, 1),
    (7, 1),
    (1, 7),
    (5, 7),
    (1, 1, 1, 1, 1),
    (2, 2, 2, 2),
    (2, 2, 2, 10),
    (1, 1, 10, 1, 1),
    (5, 7, 9),
    (5, 7, 9, 11),
    (61, 99),
]

bidirectional_configs = [
    (False,),
    (True,),
]

transposed_configs = [
    (False,),
    (True,),
]

linear_configs = [
    (False,),
    (True,),
]


# Make a Cartesian product of all configs
all_test_configs = list(
    itertools.product(
        model_configs,
        data_configs,
        bidirectional_configs,
        transposed_configs,
        linear_configs,
    )
)


@pytest.mark.parametrize(
    "d_model, d_state, data_shape, bidirectional, transposed, linear",
    [
        (d_model, d_state, data_shape, bidirectional, transposed, linear)
        for (
            (d_model, d_state),
            data_shape,
            (bidirectional,),
            (transposed,),
            (linear,),
        ) in all_test_configs
    ],
)
def test_inference(d_model, d_state, data_shape, bidirectional, transposed, linear):
    diff_out, diff_state = inference(
        d_model,
        d_state,
        data_shape,
        bidirectional,
        transposed,
        linear,
    )

    default_dtype = torch.get_default_dtype()

    if default_dtype == torch.float32:
        max_diff = 1e-4
    else:
        max_diff = 1e-13

    assert diff_out < max_diff
    assert diff_state < max_diff
