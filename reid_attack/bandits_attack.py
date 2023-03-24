"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision.utils import save_image
from tqdm.auto import tqdm

from pytorch_reid_models.reid_models.utils import set_seed, setup_logger
from reid_attack.attacker_base import QueryAttackBase


class Bandits:
    def __init__(
        self,
        target_model,
        eps=8 / 255,
        fd_eta=0.1,
        exploration=1.0,
        online_lr=100,
        max_queries=4000,
        alpha=0.01,
        downsampling=4,
    ):
        self.target_model = target_model
        self.target_model.eval().requires_grad_(False)
        self.eps = eps
        self.fd_eta = fd_eta
        self.exploration = exploration
        self.online_lr = online_lr
        self.max_queries = max_queries
        self.alpha = alpha
        self.downsampling = downsampling

        self.device = next(target_model.parameters()).device

    def _target_model_forward(self, imgs, pids, camids):
        if self.target_model.name in ["vit_transreid", "deit_transreid"]:
            # cam_label starts from 0
            feats = self.target_model(imgs, cam_label=camids - 1)
        else:
            feats = self.target_model(imgs)

        return feats

    def prior_step(self, prior, est_grad, lr=100):
        real_prior = (prior + 1) / 2  # from 0 center to 0.5 center
        pos = real_prior * torch.exp(lr * est_grad)
        neg = (1 - real_prior) * torch.exp(-lr * est_grad)
        new_prior = pos / (pos + neg)
        return new_prior * 2 - 1

    def forward(self, imgs, pids, camids):
        imgs = imgs.detach().to(self.device)

        b, c, h, w = imgs.shape
        if self.downsampling is not None:
            d = self.downsampling
            prior_shape = (b, c, int(h / d), int(w / d))
            prior = imgs.new_zeros(prior_shape)
        else:
            prior = torch.zeros_like(imgs)

        adv_imgs = imgs.clone().detach()

        feats = self._target_model_forward(imgs, pids, camids)

        for _ in range(self.max_queries // 2):
            dim = prior.nelement() / imgs.shape[0]
            exp_noise = self.exploration * torch.randn_like(prior) / (dim**0.5)

            q1 = F.normalize(F.interpolate(prior + exp_noise, size=(h, w)).view(b, -1))
            input1 = adv_imgs + self.fd_eta * q1.view_as(imgs)
            adv1 = self._target_model_forward(input1, pids, camids)
            l1 = (F.normalize(adv1) * F.normalize(feats)).sum(dim=1)

            q2 = F.normalize(F.interpolate(prior - exp_noise, size=(h, w)).view(b, -1))
            input2 = adv_imgs + self.fd_eta * q2.view_as(imgs)
            adv2 = self._target_model_forward(input2, pids, camids)
            l2 = (F.normalize(adv2) * F.normalize(feats)).sum(dim=1)

            est_deriv = (l1 - l2) / (self.fd_eta * self.exploration)
            est_grad = est_deriv.view(-1, 1, 1, 1) * exp_noise

            prior = self.prior_step(prior, est_grad, self.online_lr)

            grad = F.interpolate(prior, size=(h, w))

            adv_imgs -= self.alpha * grad.sign()
            delta = torch.clamp(adv_imgs - imgs, min=-self.eps, max=self.eps)
            adv_imgs = torch.clamp(imgs + delta, min=0, max=1)

        return adv_imgs

    def __call__(self, imgs, pids, camids):
        return self.forward(imgs, pids, camids)


class BanditsAttack(QueryAttackBase):
    def generate_adv(self, q_dataset, target_model, g_dataset):
        target_model.eval().requires_grad_(False)

        attack = Bandits(target_model)

        all_adv_imgs, all_pids, all_camids = [], [], []
        q_dataloader = data.DataLoader(q_dataset, batch_size=32, num_workers=8)
        for imgs, pids, camids in tqdm(q_dataloader, desc="Generate adv", leave=False):
            imgs, pids, camids = imgs.cuda(), pids.cuda(), camids.cuda()
            adv_imgs = attack(imgs, pids, camids)
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

    BanditsAttack().run()


if __name__ == "__main__":
    main()
