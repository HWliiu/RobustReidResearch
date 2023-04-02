"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

from functools import partial

import kornia as K
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision.utils import save_image
from tqdm.auto import tqdm

from pytorch_reid_models.reid_models.data import build_train_dataset
from pytorch_reid_models.reid_models.utils import set_seed, setup_logger
from reid_attack.attacker_base import TransferAttackBase


class TIMUAP:
    def __init__(
        self,
        attacked_model,
        epoch=1,
        eps=8 / 255,
        alpha=1 / 255,
        decay=1.0,
        len_kernel=15,
        nsig=3,
        resize_rate=0.9,
        diversity_prob=0.5,
    ):
        self.attacked_model = attacked_model
        self.attacked_model.eval()
        self.eps = eps
        self.epoch = epoch
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.len_kernel = (len_kernel, len_kernel)
        self.nsig = (nsig, nsig)

        self.device = next(attacked_model.parameters()).device

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

    def __call__(self, t_dataset):
        uap = torch.zeros(
            (1, 3, *t_dataset[0][0].shape[-2:]), device=self.device
        ).uniform_(-1e-3, 1e-3)
        momentum = torch.zeros_like(uap)

        criterion = partial(
            torch.nn.CosineEmbeddingLoss(), target=torch.ones(1, device=self.device)
        )
        t_dataloader = data.DataLoader(t_dataset, batch_size=32)
        for e in range(1, self.epoch + 1):
            for imgs, _, _ in tqdm(
                t_dataloader, desc=f"Train UAP [{e}/{self.epoch}]", leave=False
            ):
                imgs = imgs.to(self.device)

                feats = self.attacked_model(imgs)
                uap.requires_grad_(True)
                adv_imgs = torch.clamp(imgs + uap, 0, 1)
                adv_feats = self.attacked_model(self.input_diversity(adv_imgs))

                loss = criterion(adv_feats, feats)

                grad = torch.autograd.grad(loss, uap)[0]

                grad = K.filters.gaussian_blur2d(
                    grad, kernel_size=self.len_kernel, sigma=self.nsig
                )
                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                grad = grad + momentum * self.decay
                momentum = grad

                uap = uap.detach() + self.alpha * grad.sign()
                uap.clamp_(-self.eps, self.eps)

        return uap


class MIUAP:
    def __init__(self, attacked_model, epoch=1, eps=8 / 255, alpha=1 / 255, decay=1.0):
        self.attacked_model = attacked_model
        self.attacked_model.eval()
        self.eps = eps
        self.epoch = epoch
        self.decay = decay
        self.alpha = alpha

        self.device = next(attacked_model.parameters()).device

    def __call__(self, t_dataset):
        uap = torch.zeros(
            (1, 3, *t_dataset[0][0].shape[-2:]), device=self.device
        ).uniform_(-1e-3, 1e-3)
        momentum = torch.zeros_like(uap)

        criterion = partial(
            torch.nn.CosineEmbeddingLoss(), target=torch.ones(1, device=self.device)
        )
        t_dataloader = data.DataLoader(t_dataset, batch_size=32)
        for e in range(1, self.epoch + 1):
            for imgs, _, _ in tqdm(
                t_dataloader, desc=f"Train UAP [{e}/{self.epoch}]", leave=False
            ):
                imgs = imgs.to(self.device)

                feats = self.attacked_model(imgs)
                uap.requires_grad_(True)
                adv_imgs = torch.clamp(imgs + uap, 0, 1)
                adv_feats = self.attacked_model(adv_imgs)

                loss = criterion(adv_feats, feats)

                grad = torch.autograd.grad(loss, uap)[0]

                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                grad = grad + momentum * self.decay
                momentum = grad

                uap = uap.detach() + self.alpha * grad.sign()
                uap.clamp_(-self.eps, self.eps)

        return uap


class OPTIMUAP:
    def __init__(
        self,
        attacked_model,
        epoch=20,
        eps=8 / 255,
        lr=1e-4,
        len_kernel=15,
        nsig=3,
        resize_rate=0.9,
        diversity_prob=0.5,
    ):
        self.attacked_model = attacked_model
        self.attacked_model.eval()
        self.eps = eps
        self.epoch = epoch
        self.lr = lr
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.len_kernel = (len_kernel, len_kernel)
        self.nsig = (nsig, nsig)

        self.uap = None
        self.momentum = None
        self.device = next(attacked_model.parameters()).device

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

    def convert2uap(self, w):
        # return torch.tanh(w) * self.eps
        return torch.clamp(w, -self.eps, self.eps)

    def __call__(self, t_dataset):
        w = torch.nn.Parameter(
            torch.randn((1, 3, *t_dataset[0][0].shape[-2:]), device=self.device)
        )
        # w.register_hook(
        #     lambda grad: K.filters.gaussian_blur2d(
        #         grad, kernel_size=self.len_kernel, sigma=self.nsig
        #     )
        # )
        optim = torch.optim.Adam([w], lr=self.lr)

        criterion = partial(
            torch.nn.CosineEmbeddingLoss(), target=-torch.ones(1, device=self.device)
        )
        t_dataloader = data.DataLoader(t_dataset, batch_size=32)
        for e in range(1, self.epoch + 1):
            for imgs, pids, camids in tqdm(
                t_dataloader, desc=f"Train UAP[{e}/{self.epoch}]", leave=False
            ):
                imgs = imgs.to(self.device)

                feats = self.attacked_model(imgs)
                adv_imgs = torch.clamp(imgs + self.convert2uap(w), 0, 1)
                # adv_feats = self.attacked_model(self.input_diversity(adv_imgs))
                adv_feats = self.attacked_model(adv_imgs)

                loss = criterion(adv_feats, feats)

                optim.zero_grad(True)
                loss.backward()
                optim.step()

        return self.convert2uap(w.detach())


class UAPAttack(TransferAttackBase):
    def generate_adv(self, q_dataset, agent_model):
        agent_model.eval().requires_grad_(False)

        attack = MIUAP(agent_model)
        # attack = OPTIMUAP(agent_model)

        all_adv_imgs, all_pids, all_camids = [], [], []

        t_dataset = build_train_dataset([q_dataset.name], per_dataset_num=800)
        uap = attack(t_dataset)

        q_dataloader = data.DataLoader(q_dataset, batch_size=32, num_workers=8)
        for imgs, pids, camids in q_dataloader:
            imgs, pids, camids = imgs.cuda(), pids.cuda(), camids.cuda()
            adv_imgs = torch.clamp(imgs + uap, 0, 1)
            all_adv_imgs.append(adv_imgs.cpu())
            all_pids.append(pids.cpu())
            all_camids.append(camids.cpu())
        all_adv_imgs = torch.cat(all_adv_imgs)
        all_pids = torch.cat(all_pids)
        all_camids = torch.cat(all_camids)

        return data.TensorDataset(all_adv_imgs, all_pids, all_camids)


def main():
    setup_logger(name="pytorch_reid_models.reid_models")
    setup_logger(name="__main__")

    set_seed(42)

    UAPAttack("bagtricks_R50_ibn_fastreid", ("bagtricks_R50_ibn_fastreid",)).run()


if __name__ == "__main__":
    main()
