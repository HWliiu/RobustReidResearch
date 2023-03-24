"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

from functools import partial

import kornia as K
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils import data
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm.auto import tqdm

from nips2017_attack.attacker_base import TransferAttackBase
from nips2017_attack.utils import set_seed, setup_logger


def build_train_dataset(dir_path, train_num):
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    train_dataset = ImageFolder(dir_path, transform)
    train_dataset = data.Subset(
        train_dataset,
        indices=torch.linspace(
            0, len(train_dataset) - 1, steps=train_num, dtype=torch.int64
        ).tolist(),
    )
    return train_dataset


class TIMUAP:
    def __init__(
        self,
        agent_model,
        epoch=10,
        eps=8 / 255,
        alpha=0.001,
        decay=1.0,
        len_kernel=15,
        nsig=3,
        resize_rate=0.9,
        diversity_prob=0.5,
    ):
        self.agent_model = agent_model
        self.agent_model.eval().requires_grad_(False)
        self.eps = eps
        self.epoch = epoch
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
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
        rescaled = F.interpolate(
            x, size=[rnd, rnd], mode="bilinear", align_corners=False
        )
        h_rem = w_rem = img_resize - rnd
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
        uap = torch.zeros((1, 3, *t_dataset[0][0].shape[-2:]), device=self.device)
        momentum = torch.zeros_like(uap)

        criterion = torch.nn.CrossEntropyLoss()
        t_dataloader = data.DataLoader(t_dataset, batch_size=32, shuffle=True)
        for e in range(1, self.epoch + 1):
            for imgs, lbls in tqdm(
                t_dataloader, desc=f"Train UAP [{e}/{self.epoch}]", leave=False
            ):
                imgs = imgs.to(self.device)
                lbls = lbls.to(self.device)

                uap.requires_grad_(True)
                adv_imgs = torch.clamp(imgs + uap, 0, 1)
                logits = self.agent_model(self.input_diversity(adv_imgs))

                loss = criterion(logits, lbls)

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


class UAPAttack(TransferAttackBase):
    def generate_adv(self, test_dataset, agent_model):
        agent_model.eval().requires_grad_(False)

        attack = TIMUAP(agent_model)

        all_adv_imgs, all_lbls = [], []

        t_dataset = build_train_dataset("datasets/val", train_num=800)
        uap = attack(t_dataset)

        test_dataloader = data.DataLoader(test_dataset, batch_size=32, num_workers=8)
        for imgs, lbls in test_dataloader:
            imgs, lbls = imgs.cuda(), lbls.cuda()
            adv_imgs = torch.clamp(imgs + uap, 0, 1)
            all_adv_imgs.append(adv_imgs.cpu())
            all_lbls.append(lbls.cpu())
        all_adv_imgs = torch.cat(all_adv_imgs)
        all_lbls = torch.cat(all_lbls)

        return data.TensorDataset(all_adv_imgs, all_lbls)


def main():
    setup_logger(name="__main__")

    set_seed(42)

    UAPAttack().run()


if __name__ == "__main__":
    main()
