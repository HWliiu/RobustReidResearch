"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import logging
import random
from pathlib import Path

import accelerate
import kornia as K
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision.utils import save_image
from tqdm.auto import tqdm

from pytorch_reid_models.reid_models.data import build_test_datasets
from pytorch_reid_models.reid_models.modeling import build_reid_model
from pytorch_reid_models.reid_models.utils import set_seed, setup_logger
from reid_attack.attacker_base import TransferAttackBase


class TIM:
    def __init__(
        self,
        agent_model,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        decay=1.0,
        len_kernel=15,
        nsig=3,
        resize_rate=0.9,
        diversity_prob=0.5,
        random_start=True,
    ):
        self.agent_model = agent_model
        self.agent_model.eval().requires_grad_(False)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.len_kernel = (len_kernel, len_kernel)
        self.nsig = (nsig, nsig)

        self.device = next(agent_model.parameters()).device

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(
            low=img_size, high=img_resize, size=(1,), dtype=torch.int32
        ).item()
        ratio = x.shape[2] / x.shape[3]
        rescaled = F.interpolate(
            x, size=[int(rnd * ratio), rnd], mode="bilinear", align_corners=False
        )
        h_rem = int((img_resize - rnd) * ratio)
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem, size=(1,), dtype=torch.int32).item()
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem, size=(1,), dtype=torch.int32).item()
        pad_right = w_rem - pad_left

        padded = F.pad(
            rescaled,
            [pad_left, pad_right, pad_top, pad_bottom],
            value=0,
        )

        return padded if torch.rand(1) < self.diversity_prob else x

    # def forward(self, images, matches):
    #     images = images.detach().to(self.device)
    #     matches = matches.detach().to(self.device)

    #     criterion = torch.nn.CosineEmbeddingLoss()

    #     momentum = torch.zeros_like(images).detach().to(self.device)

    #     adv_images = images.clone().detach()
    #     if self.random_start:
    #         # Starting at a uniformly random point
    #         adv_images = adv_images + torch.empty_like(adv_images).uniform_(
    #             -self.eps, self.eps
    #         )
    #         adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    #     for _ in range(self.steps):
    #         adv_images.requires_grad = True

    #         adv_feats = self.model(self.input_diversity(adv_images))
    #         matches_feats = torch.stack([self.model(match) for match in matches])

    #         # Calculate loss
    #         loss = criterion(
    #             repeat(adv_feats, "b d->(b n) d", n=matches_feats.shape[1]),
    #             rearrange(matches_feats, "b n d->(b n) d"),
    #             torch.ones(1, device=adv_feats.device),
    #         )

    #         # Update adversarial images
    #         grad = torch.autograd.grad(
    #             loss, adv_images, retain_graph=False, create_graph=False
    #         )[0]

    #         # depth wise conv2d
    #         grad = K.filters.gaussian_blur2d(
    #             grad, kernel_size=self.len_kernel, sigma=self.nsig
    #         )
    #         grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
    #         grad = grad + momentum * self.decay
    #         momentum = grad

    #         adv_images = adv_images.detach() + self.alpha * grad.sign()
    #         delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
    #         adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    #     return adv_images

    # def __call__(self, images, matches):
    #     return self.forward(images, matches)

    def forward(self, images):
        images = images.detach().to(self.device)

        criterion = torch.nn.CosineEmbeddingLoss()

        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        feats = self.agent_model(images)
        for _ in range(self.steps):
            adv_images.requires_grad = True

            adv_feats = self.agent_model(self.input_diversity(adv_images))

            # Calculate loss
            loss = criterion(
                adv_feats,
                feats,
                torch.ones(1, device=adv_feats.device),
            )

            # Update adversarial images
            grad = torch.autograd.grad(
                loss, adv_images, retain_graph=False, create_graph=False
            )[0]

            # depth wise conv2d
            grad = K.filters.gaussian_blur2d(
                grad, kernel_size=self.len_kernel, sigma=self.nsig
            )
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def __call__(self, images):
        return self.forward(images)


class TIMAttack(TransferAttackBase):
    def generate_adv(self, q_dataset, agent_model, g_dataset=None):
        attack = TIM(agent_model)

        all_adv_imgs, all_pids, all_camids = [], [], []
        q_dataloader = data.DataLoader(q_dataset, batch_size=32, num_workers=8)
        for imgs, pids, camids in tqdm(q_dataloader, desc="Generate adv", leave=False):
            imgs, pids, camids = imgs.cuda(), pids.cuda(), camids.cuda()
            adv_imgs = attack(imgs)
            all_adv_imgs.append(adv_imgs.cpu())
            all_pids.append(pids.cpu())
            all_camids.append(camids.cpu())
        all_adv_imgs = torch.cat(all_adv_imgs)
        all_pids = torch.cat(all_pids)
        all_camids = torch.cat(all_camids)

        return data.TensorDataset(all_adv_imgs, all_pids, all_camids)


def main():
    setup_logger(name="reid_models")
    setup_logger(name="__main__")

    set_seed(42)

    TIMAttack().run()


if __name__ == "__main__":
    main()
