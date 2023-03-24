"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import logging

import timm
import torch
from foolbox import PyTorchModel
from foolbox.attacks import HopSkipJumpAttack
from torch.utils import data
from torchvision.utils import save_image
from tqdm.auto import tqdm

from nips2017_attack.attacker_base import (
    QueryAttackBase,
    build_test_dataset,
    timer,
    timm_model_wrapper,
)
from nips2017_attack.utils import set_seed, setup_logger


class HJSAAttack(QueryAttackBase):
    def generate_adv(self, test_dataset, target_model):
        target_model.eval().requires_grad_(False)

        f_model = PyTorchModel(target_model, bounds=(0, 1))
        # about 2,000 queries per image
        attack = HopSkipJumpAttack(
            steps=10,
            max_gradient_eval_steps=180,
            constraint="linf",
        )

        eps = 8 / 255

        all_raw_adv_imgs, all_adv_imgs, all_lbls = [], [], []
        test_dataloader = data.DataLoader(test_dataset, batch_size=8, num_workers=8)
        for imgs, lbls in tqdm(test_dataloader, desc="Generate adv", leave=False):
            imgs, lbls = imgs.cuda(), lbls.cuda()

            no_clipped_advs, clipped_advs, success = attack(
                f_model, imgs, criterion=lbls, epsilons=eps
            )

            # raw_adv_imgs means delta may larger than epislon
            raw_adv_imgs = torch.clamp(no_clipped_advs, 0, 1)
            adv_imgs = torch.clamp(clipped_advs, 0, 1)
            all_raw_adv_imgs.append(raw_adv_imgs.cpu())
            all_adv_imgs.append(adv_imgs.cpu())
            all_lbls.append(lbls.cpu())

        all_raw_adv_imgs = torch.cat(all_raw_adv_imgs)
        all_adv_imgs = torch.cat(all_adv_imgs)
        all_lbls = torch.cat(all_lbls)

        return (
            data.TensorDataset(all_raw_adv_imgs, all_lbls),
            data.TensorDataset(all_adv_imgs, all_lbls),
        )

    def run(self):
        logger = logging.getLogger("__main__")

        test_dataset = build_test_dataset(self.test_num)

        for target_model_name in self.target_model_names:
            target_model = timm_model_wrapper(
                timm.create_model(target_model_name, pretrained=True).eval().cuda()
            )

            target_model = self.accelerator.prepare(target_model)
            (raw_adv_dataset, adv_dataset), spend_time = timer(self.generate_adv)(
                test_dataset, target_model
            )

            logger.info(f"Spend Time: {spend_time}")
            raw_vqe_results = self.evaluate_vqe(test_dataset, raw_adv_dataset)
            logger.info(f"No Clipped VQE Metrics:\t" + raw_vqe_results)
            vqe_results = self.evaluate_vqe(test_dataset, adv_dataset)
            logger.info(f"Clipped VQE Metrics:\t" + vqe_results)

            raw_reid_results = self.evaluate_imagenet(
                test_dataset, raw_adv_dataset, target_model
            )
            logger.info(
                f"No Clipped ImageNet Metrics: {target_model_name}\n" + raw_reid_results
            )
            reid_results = self.evaluate_imagenet(
                test_dataset, adv_dataset, target_model
            )
            logger.info(
                f"Clipped ImageNet Metrics: {target_model_name}\n" + reid_results
            )
            torch.cuda.empty_cache()


def main():
    setup_logger(name="reid_models")
    setup_logger(name="__main__")

    set_seed(42)

    HJSAAttack().run()


if __name__ == "__main__":
    main()
