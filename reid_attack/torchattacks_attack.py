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
from third_party.torchattacks import (
    BIM,
    DIFGSM,
    FGSM,
    MIFGSM,
    NIFGSM,
    RFGSM,
    SINIFGSM,
    TIFGSM,
    VMIFGSM,
)


class TransferAttack(TransferAttackBase):
    def _random_start(self, imgs, eps):
        imgs = imgs + torch.empty_like(imgs).uniform_(-eps, eps)
        imgs = torch.clamp(imgs, min=0, max=1).detach()
        return imgs

    def generate_adv(self, q_dataset, agent_model):
        agent_model.eval().requires_grad_(False)

        eps = 8 / 255
        # attack = FGSM(agent_model, eps=eps)
        # attack = BIM(agent_model, eps=eps, alpha=2 / 255, steps=10)
        # attack = TIFGSM(agent_model, eps=eps, alpha=2 / 255, steps=10, decay=1)
        # attack = MIFGSM(agent_model, eps=eps, alpha=2 / 255, steps=10, decay=1)
        # attack = DIFGSM(agent_model, eps=eps, alpha=2 / 255, steps=10, decay=1)
        # attack = NIFGSM(agent_model, eps=eps, alpha=2 / 255, steps=10, decay=1)
        # attack = SINIFGSM(agent_model, eps=eps, alpha=2 / 255, steps=10, decay=1)
        attack = VMIFGSM(agent_model, eps=eps, alpha=2 / 255, steps=10, decay=1)

        all_adv_imgs, all_pids, all_camids = [], [], []
        q_dataloader = data.DataLoader(q_dataset, batch_size=32, num_workers=8)
        for imgs, pids, camids in tqdm(q_dataloader, desc="Generate adv", leave=False):
            imgs, pids, camids = imgs.cuda(), pids.cuda(), camids.cuda()
            y = agent_model(self._random_start(imgs, 1e-2))
            adv_imgs = attack.forward(imgs, y)
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

    TransferAttack().run()


if __name__ == "__main__":
    main()
