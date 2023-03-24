"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import torch
from torch.utils import data
from torchattacks import (
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
from torchvision.utils import save_image
from tqdm.auto import tqdm

from nips2017_attack.attacker_base import TransferAttackBase
from nips2017_attack.utils import set_seed, setup_logger


class TransferAttack(TransferAttackBase):
    def generate_adv(self, test_dataset, agent_model):
        agent_model.eval().requires_grad_(False)

        eps = 8 / 255
        # attack = FGSM(agent_model, eps=eps)
        # attack = BIM(agent_model, eps=eps, alpha=1 / 255, steps=50)
        # attack = TIFGSM(agent_model, eps=eps, alpha=1 / 255, steps=50, decay=1)
        # attack = MIFGSM(agent_model, eps=eps, alpha=1 / 255, steps=50, decay=1)
        attack = DIFGSM(agent_model, eps=eps, alpha=1 / 255, steps=50, decay=1)
        # attack = NIFGSM(agent_model, eps=eps, alpha=1 / 255, steps=50, decay=1)
        # attack = SINIFGSM(agent_model, eps=eps, alpha=1 / 255, steps=50, decay=1)
        # attack = VMIFGSM(agent_model, eps=eps, alpha=1 / 255, steps=50, decay=1)

        all_adv_imgs, all_lbls = [], []
        test_dataloader = data.DataLoader(test_dataset, batch_size=32, num_workers=8)
        for imgs, lbls in tqdm(test_dataloader, desc="Generate adv", leave=False):
            imgs, lbls = imgs.cuda(), lbls.cuda()
            adv_imgs = attack.forward(imgs, lbls)
            all_adv_imgs.append(adv_imgs.cpu())
            all_lbls.append(lbls.cpu())

        all_adv_imgs = torch.cat(all_adv_imgs)
        all_lbls = torch.cat(all_lbls)

        return data.TensorDataset(all_adv_imgs, all_lbls)


def main():
    setup_logger(name="__main__")

    set_seed(42)

    TransferAttack().run()


if __name__ == "__main__":
    main()
