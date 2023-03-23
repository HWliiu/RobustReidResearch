"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

from functools import partial

import torch
from torch.utils import data
from torchvision.utils import save_image
from tqdm.auto import tqdm

from pytorch_reid_models.reid_models.utils import set_seed, setup_logger
from reid_attack.attacker_base import QueryAttackBase, TransferAttackBase
from third_party.advertorch import attacks


class TransferAttack(TransferAttackBase):
    def _random_start(self, imgs, eps):
        imgs = imgs + torch.empty_like(imgs).uniform_(-eps, eps)
        imgs = torch.clamp(imgs, min=0, max=1).detach()
        return imgs

    def generate_adv(self, q_dataset, agent_model):
        agent_model.eval().requires_grad_(False)

        loss_fn = partial(
            torch.nn.CosineEmbeddingLoss(), target=torch.ones(1, device="cuda")
        )
        eps = 8 / 255
        attack = attacks.LinfMomentumIterativeAttack(
            agent_model, loss_fn, eps=eps, eps_iter=1 / 255, nb_iter=50
        )

        # eps = 9.8
        # attack = attacks.L2MomentumIterativeAttack(
        #     agent_model, loss_fn, eps=eps, eps_iter=9.8 / 8, nb_iter=50
        # )

        all_adv_imgs, all_pids, all_camids = [], [], []
        q_dataloader = data.DataLoader(q_dataset, batch_size=32, num_workers=8)
        for imgs, pids, camids in tqdm(q_dataloader, desc="Generate adv", leave=False):
            imgs, pids, camids = imgs.cuda(), pids.cuda(), camids.cuda()
            y = agent_model(self._random_start(imgs, 1e-2))
            adv_imgs = attack.perturb(imgs, y)
            all_adv_imgs.append(adv_imgs.cpu())
            all_pids.append(pids.cpu())
            all_camids.append(camids.cpu())
        all_adv_imgs = torch.cat(all_adv_imgs)
        all_pids = torch.cat(all_pids)
        all_camids = torch.cat(all_camids)

        return data.TensorDataset(all_adv_imgs, all_pids, all_camids)


class QueryAttack(QueryAttackBase):
    def _random_start(self, imgs, eps):
        imgs = imgs + torch.empty_like(imgs).uniform_(-eps, eps)
        imgs = torch.clamp(imgs, min=0, max=1).detach()
        return imgs

    def generate_adv(self, q_dataset, target_model, g_dataset=None):
        target_model.eval().requires_grad_(False)

        loss_fn = partial(
            torch.nn.CosineEmbeddingLoss(reduction="none"),
            target=torch.ones(1, device="cuda"),
        )
        eps = 8 / 255
        # attack = attacks.BanditAttack(
        #     target_model,
        #     eps=eps,
        #     order=torch.inf,
        #     fd_eta=0.1,
        #     exploration=1.0,
        #     online_lr=100,
        #     loss_fn=loss_fn,
        #     nb_iter=2500,
        #     eps_iter=0.01,
        #     downsampling=True,
        # )
        attack = attacks.NESAttack(
            target_model,
            loss_fn=loss_fn,
            eps=eps,
            nb_samples=250,
            nb_iter=10,
            eps_iter=2 / 255,
        )

        all_adv_imgs, all_pids, all_camids = [], [], []
        q_dataloader = data.DataLoader(q_dataset, batch_size=32, num_workers=8)
        for imgs, pids, camids in tqdm(q_dataloader, desc="Generate adv", leave=False):
            imgs, pids, camids = imgs.cuda(), pids.cuda(), camids.cuda()
            y = target_model(self._random_start(imgs, 1e-2))
            adv_imgs = attack.perturb(imgs, y)
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

    # TransferAttack().run()
    QueryAttack().run()


if __name__ == "__main__":
    main()
