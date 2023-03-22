# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import numpy as np
from einops import repeat

from ...utils import clamp
from ..utils import rand_init_delta
from ...utils import clamp
from ...utils import normalize_by_pnorm
from ...utils import clamp_by_pnorm
from ...utils import batch_multiply
from ...utils import batch_clamp
from ...utils import batch_l1_proj

from ..iterative_projected_gradient import LinfPGDAttack

from .estimators import NESWrapper
from .utils import _flatten

# !Modified (Fixed bug!)
# predict function need output loss
def perturb_iterative(
    xvar,
    yvar,
    predict,
    nb_iter,
    eps,
    eps_iter,
    delta_init=None,
    minimize=False,
    ord=np.inf,
    clip_min=0.0,
    clip_max=1.0,
    l1_sparsity=None,
):
    """
    Iteratively maximize the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.

    :param xvar: input data.
    :param yvar: input labels.
    :param predict: forward pass function.
    :param nb_iter: number of iterations.
    :param eps: maximum distortion.
    :param eps_iter: attack step size.
    :param delta_init: (optional) tensor contains the random initialization.
    :param minimize: (optional bool) whether to minimize or maximize the loss.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param l1_sparsity: sparsity value for L1 projection.
                  - if None, then perform regular L1 projection.
                  - if float value, then perform sparse L1 descent from
                    Algorithm 1 in https://arxiv.org/pdf/1904.13000v1.pdf
    :return: tensor containing the perturbed input.
    """
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)

    delta.requires_grad_()
    for ii in range(nb_iter):
        loss = predict(xvar + delta)
        if minimize:
            loss = -loss

        loss.backward()
        if ord == np.inf:
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
            delta.data = batch_clamp(eps, delta.data)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max) - xvar.data

        elif ord == 2:
            grad = delta.grad.data
            grad = normalize_by_pnorm(grad)
            delta.data = delta.data + batch_multiply(eps_iter, grad)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max) - xvar.data
            if eps is not None:
                delta.data = clamp_by_pnorm(delta.data, ord, eps)

        elif ord == 1:
            grad = delta.grad.data
            abs_grad = torch.abs(grad)

            batch_size = grad.size(0)
            view = abs_grad.view(batch_size, -1)
            view_size = view.size(1)
            if l1_sparsity is None:
                vals, idx = view.topk(1)
            else:
                vals, idx = view.topk(int(np.round((1 - l1_sparsity) * view_size)))

            out = torch.zeros_like(view).scatter_(1, idx, vals)
            out = out.view_as(grad)
            grad = grad.sign() * (out > 0).float()
            grad = normalize_by_pnorm(grad, p=1)
            delta.data = delta.data + batch_multiply(eps_iter, grad)

            delta.data = batch_l1_proj(delta.data.cpu(), eps)
            delta.data = delta.data.to(xvar.device)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max) - xvar.data
        else:
            error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
            raise NotImplementedError(error)
        delta.grad.data.zero_()

    x_adv = clamp(xvar + delta, clip_min, clip_max)
    return x_adv


class NESAttack(LinfPGDAttack):
    """
    Implements NES Attack https://arxiv.org/abs/1804.08598

    Employs Natural Evolutionary Strategies for Gradient Estimation.
    Generates Adversarial Examples using Projected Gradient Descent.

    Disclaimer: Computations are broadcasted, so it is advisable to use
    smaller batch sizes when nb_samples is large.

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_samples: number of samples to use for gradient estimation
    :param fd_eta: step-size used for Finite Difference gradient estimation
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    # !Modified
    def __init__(
        self,
        predict,
        loss_fn,
        eps=0.3,
        nb_samples=100,
        fd_eta=1e-2,
        nb_iter=40,
        eps_iter=0.01,
        rand_init=True,
        clip_min=0.0,
        clip_max=1.0,
        targeted=False,
    ):

        super(NESAttack, self).__init__(
            predict=predict,
            loss_fn=loss_fn,
            eps=eps,
            nb_iter=nb_iter,
            eps_iter=eps_iter,
            rand_init=rand_init,
            clip_min=clip_min,
            clip_max=clip_max,
            targeted=targeted,
        )

        self.nb_samples = nb_samples
        self.fd_eta = fd_eta

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)
        shape, flat_x = _flatten(x)
        data_shape = tuple(shape[1:])

        def f(x):
            new_shape = (x.shape[0],) + data_shape
            input = x.reshape(new_shape)
            # !Modified
            nonlocal y
            if len(input) != len(y):
                y_ = repeat(y, "b d->(b n) d", n=self.nb_samples)
                return self.loss_fn(self.predict(input), y_)
            return self.loss_fn(self.predict(input), y).mean()

        f_nes = NESWrapper(f, nb_samples=self.nb_samples, fd_eta=self.fd_eta)

        delta = torch.zeros_like(flat_x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            rand_init_delta(
                delta, flat_x, self.ord, self.eps, self.clip_min, self.clip_max
            )
            delta.data = (
                clamp(flat_x + delta.data, min=self.clip_min, max=self.clip_max)
                - flat_x
            )
        # !Modified
        rval = perturb_iterative(
            flat_x,
            y,
            f_nes,
            nb_iter=self.nb_iter,
            eps=self.eps,
            eps_iter=self.eps_iter,
            minimize=self.targeted,
            ord=self.ord,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            delta_init=delta,
            l1_sparsity=None,
        )

        return rval.data.reshape(shape)
