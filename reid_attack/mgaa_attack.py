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

from pytorch_reid_models.reid_models.utils import set_seed, setup_logger
from reid_attack.attacker_base import EnsTransferAttackBase


class MGAATIM:
    def __init__(
        self,
        attacked_models,
        eps=8 / 255,
        alpha=2 / 255,
        meta_test_step=10,
        meta_train_step=8,
        decay=1.0,
        len_kernel=15,
        nsig=3,
        resize_rate=0.9,
        diversity_prob=0.5,
        random_start=True,
    ):
        self.attacked_models = attacked_models
        for model in self.attacked_models:
            model.eval()
        self.eps = eps
        self.meta_test_step = meta_test_step
        self.meta_train_step = meta_train_step
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.len_kernel = (len_kernel, len_kernel)
        self.nsig = (nsig, nsig)

        self.device = next(attacked_models[0].parameters()).device

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

    def forward(self, images):
        images = images.detach().to(self.device)

        criterion = criterion = partial(
            torch.nn.CosineEmbeddingLoss(), target=torch.ones(1, device=self.device)
        )

        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        all_feats = [model(images) for model in self.attacked_models]
        for _ in range(self.meta_test_step):
            rand_idx = torch.randperm(len(self.attacked_models))
            meta_train_idx = rand_idx[:-1]
            meta_test_idx = rand_idx[-1]

            meta_train_models = [self.attacked_models[i] for i in meta_train_idx]
            meta_test_model = self.attacked_models[meta_test_idx]

            # meta train
            meta_train_adv_imgs = adv_images.clone()
            for _ in range(self.meta_train_step):
                meta_train_adv_imgs.requires_grad_(True)
                meta_train_adv_feats = [
                    model(self.input_diversity(meta_train_adv_imgs))
                    for model in meta_train_models
                ]

                meta_train_feats = [all_feats[i] for i in meta_train_idx]

                loss = sum(
                    [
                        criterion(adv_feats, feats)
                        for adv_feats, feats in zip(
                            meta_train_adv_feats, meta_train_feats
                        )
                    ]
                ) / len(meta_train_idx)

                grad = torch.autograd.grad(
                    loss, meta_train_adv_imgs, retain_graph=False, create_graph=False
                )[0]

                # depth wise conv2d
                grad = K.filters.gaussian_blur2d(
                    grad, kernel_size=self.len_kernel, sigma=self.nsig
                )
                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                grad = grad + momentum * self.decay
                momentum = grad

                meta_train_adv_imgs = (
                    meta_train_adv_imgs.detach() + self.alpha * grad.sign()
                )
                delta = torch.clamp(
                    meta_train_adv_imgs - images, min=-self.eps, max=self.eps
                )
                meta_train_adv_imgs = torch.clamp(images + delta, min=0, max=1).detach()

            # 元测试
            meta_test_adv_imgs = meta_train_adv_imgs.clone()
            meta_test_adv_imgs.requires_grad_(True)
            meta_test_adv_feats = meta_test_model(
                self.input_diversity(meta_test_adv_imgs)
            )
            meta_test_feats = all_feats[meta_test_idx]

            loss = criterion(
                meta_test_adv_feats,
                meta_test_feats,
                torch.ones(1, device=meta_test_adv_feats.device),
            )

            grad = torch.autograd.grad(
                loss, meta_test_adv_imgs, retain_graph=False, create_graph=False
            )[0]

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


class MGAATIMAttack(EnsTransferAttackBase):
    def generate_adv(self, q_dataset, agent_models):
        for model in self.agent_models:
            model.eval().requires_grad_(False)

        attack = MGAATIM(agent_models)

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
    setup_logger(name="pytorch_reid_models.reid_models")
    setup_logger(name="__main__")

    set_seed(42)

    MGAATIMAttack().run()


if __name__ == "__main__":
    main()
