"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import random
from copy import deepcopy
from pathlib import Path

import kornia as K
import torch
import torch.nn as nn
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import BatchHardMiner
from torch.utils import data
from torchvision.utils import save_image
from tqdm.auto import tqdm

from pytorch_reid_models.reid_models.data import build_train_dataloader
from pytorch_reid_models.reid_models.utils import set_seed, setup_logger
from reid_attack.attacker_base import TransferAttackBase

# 'third_party/torchattacks/wrappers/lgv.py' not available

class EnsMIFGSM:
    def __init__(
        self,
        random_ens_model,
        eps=8 / 255,
        alpha=1 / 255,
        steps=50,
        decay=1.0,
        # len_kernel=15,
        # nsig=3,
        # resize_rate=0.9,
        # diversity_prob=0.5,
        random_start=True,
    ):
        self.random_ens_model = random_ens_model
        self.random_ens_model.eval().requires_grad_(False)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        # self.resize_rate = resize_rate
        # self.diversity_prob = diversity_prob
        self.random_start = random_start
        # self.len_kernel = (len_kernel, len_kernel)
        # self.nsig = (nsig, nsig)

        self.device = next(random_ens_model.parameters()).device

    # def input_diversity(self, x):
    #     img_size = x.shape[-1]
    #     img_resize = int(img_size * self.resize_rate)

    #     if self.resize_rate < 1:
    #         img_size = img_resize
    #         img_resize = x.shape[-1]

    #     rnd = torch.randint(
    #         low=img_size, high=img_resize, size=(1,), dtype=torch.int32
    #     ).item()
    #     ratio = x.shape[2] / x.shape[3]
    #     rescaled = F.interpolate(
    #         x, size=[int(rnd * ratio), rnd], mode="bilinear", align_corners=False
    #     )
    #     h_rem = int((img_resize - rnd) * ratio)
    #     w_rem = img_resize - rnd
    #     pad_top = torch.randint(low=0, high=h_rem, size=(1,), dtype=torch.int32).item()
    #     pad_bottom = h_rem - pad_top
    #     pad_left = torch.randint(low=0, high=w_rem, size=(1,), dtype=torch.int32).item()
    #     pad_right = w_rem - pad_left

    #     padded = F.pad(
    #         rescaled,
    #         [pad_left, pad_right, pad_top, pad_bottom],
    #         value=0,
    #     )

    #     return padded if torch.rand(1) < self.diversity_prob else x

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

        for _ in range(self.steps):
            adv_images.requires_grad = True

            feats_list = self.random_ens_model(images)
            # adv_feats_list = self.random_ens_model(
            #     self.input_diversity(adv_images), use_last_models=True
            # )
            adv_feats_list = self.random_ens_model(adv_images, use_last_models=True)

            # Calculate loss
            loss = sum(
                [
                    criterion(
                        adv_feats,
                        feats,
                        torch.ones(1, device=self.device),
                    )
                    for adv_feats, feats in zip(adv_feats_list, feats_list)
                ]
            )

            # Update adversarial images
            grad = torch.autograd.grad(
                loss, adv_images, retain_graph=False, create_graph=False
            )[0]

            # depth wise conv2d
            # grad = K.filters.gaussian_blur2d(
            #     grad, kernel_size=self.len_kernel, sigma=self.nsig
            # )
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def __call__(self, images):
        return self.forward(images)


class RandomEnsModels(torch.nn.Module):
    def __init__(self, models_list, ens_num=1):
        super().__init__()
        self.models = torch.nn.ModuleList(models_list)
        self.ens_num = ens_num
        # Calculating clean features and adversarial features requires
        # ensuring that the models are the same
        self.last_indexes = None

    def forward(self, x, use_last_models=False):
        if use_last_models:
            assert self.last_indexes is not None
            indexes = self.last_indexes
        else:
            indexes = random.sample(range(len(self.models)), self.ens_num)
            self.last_indexes = indexes

        feats_list = [self.models[i](x) for i in indexes]

        return feats_list


class LGVAttack(TransferAttackBase):
    def _random_start(self, imgs, eps):
        imgs = imgs + torch.empty_like(imgs).uniform_(-eps, eps)
        imgs = torch.clamp(imgs, min=0, max=1).detach()
        return imgs

    def generate_adv(self, q_dataset, agent_model, g_dataset=None):
        collect_models = self.lgv_collect_models(
            agent_model, train_dataset_name=q_dataset.name
        )
        for model in collect_models:
            model.eval().requires_grad_(False)

        random_ens_model = RandomEnsModels(collect_models, ens_num=1)

        eps = 8 / 255
        attack = EnsMIFGSM(
            random_ens_model, eps=eps, alpha=1 / 255, steps=50, decay=1.0
        )

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

    @staticmethod
    def lgv_collect_models(
        agent_model, train_dataset_name, epoch=10, nb_models_epoch=4, lr=5e-2
    ):
        models_list = []
        models_path = Path("/tmp/lgv_models.pth")
        if models_path.exists():
            state_dict_list = torch.load(models_path, map_location="cpu")
            assert len(state_dict_list) == epoch * nb_models_epoch
            for state in state_dict_list:
                model = deepcopy(agent_model)
                model.eval().requires_grad_(False)
                model.load_state_dict(state)
                models_list.append(model)
        else:
            # Construct training data
            train_dataloader = build_train_dataloader(
                dataset_names=[train_dataset_name],
                per_dataset_num=None,
                transforms=["randomflip"],
                batch_size=64,
                sampler="pk",
                num_instance=8,
            )

            # Fine-tuning model
            model = deepcopy(agent_model)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=1e-4,
            )

            # Loss function
            miner = BatchHardMiner()
            criterion_t = TripletMarginLoss(margin=0.3)
            criterion_x = nn.CrossEntropyLoss(label_smoothing=0.1)

            # Start fine-tuning
            for e in range(epoch):
                model.train()
                save_points = torch.linspace(
                    0,
                    len(train_dataloader) - 1,
                    steps=nb_models_epoch + 1,
                    dtype=torch.int64,
                )[1:]
                for i, (imgs, pids, _) in enumerate(
                    tqdm(train_dataloader, desc=f"Epoch {e}", leave=False)
                ):
                    imgs, pids = imgs.cuda(), pids.cuda()
                    logits, feats = model(imgs)

                    # FIXME: The pid relabel does not match the original model training code
                    # (We tried adding the correct matching version of xent loss and the result was worse)
                    # loss_x = criterion_x(logits, pids)
                    hard_pairs = miner(feats, pids)
                    loss_t = criterion_t(feats, pids, hard_pairs)
                    # loss = loss_x + loss_t
                    loss = loss_t

                    optimizer.zero_grad(True)
                    loss.backward()
                    optimizer.step()

                    if i in save_points:
                        save_model = deepcopy(model)
                        models_list.append(save_model)

            # Save model
            torch.save([m.state_dict() for m in models_list], models_path)

        return models_list


def main():
    setup_logger(name="reid_models")
    setup_logger(name="__main__")

    set_seed(42)

    LGVAttack().run()


if __name__ == "__main__":
    main()
