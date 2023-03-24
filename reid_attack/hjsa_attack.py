"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import logging

import eagerpy as ep
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision.utils import save_image
from tqdm.auto import tqdm

from pytorch_reid_models.reid_models.evaluate import Matcher
from pytorch_reid_models.reid_models.modeling import build_reid_model
from pytorch_reid_models.reid_models.utils import set_seed, setup_logger
from reid_attack.attacker_base import QueryAttackBase, timer
from third_party.foolbox import PyTorchModel
from third_party.foolbox.attacks import (
    HopSkipJumpAttack,
    LinearSearchBlendedUniformNoiseAttack,
)
from third_party.foolbox.attacks.base import get_is_adversarial
from third_party.foolbox.criteria import Misclassification


class ReidMisCriterion(Misclassification):
    # For foolbox compatibility
    def __init__(self, q_pids, q_camids, g_pids, g_camids, g_feats_optimized, topk=10):
        self.q_pids = q_pids
        self.q_camids = q_camids
        self.g_pids = g_pids
        self.g_camids = g_camids
        self.g_feats_optimized = g_feats_optimized
        self.topk = topk

    def __repr__(self) -> str:
        return object.__repr__()

    def __call__(self, perturbed, outputs):
        outputs = ep.astensor(outputs)
        q_feats = F.normalize(outputs.raw)
        g_feats = self.g_feats_optimized
        sim_mat = torch.mm(q_feats, g_feats)
        # remove gallery samples that have the same pid and camid with query
        sim_mat[
            (self.q_pids.view(-1, 1) == self.g_pids)
            & (self.q_camids.view(-1, 1) == self.g_camids)
        ] = -1
        _, order = torch.topk(sim_mat, k=self.topk)
        matches = self.q_pids[:, None] == self.g_pids[order]
        is_adv = ~matches.any(dim=-1)
        return ep.astensor(is_adv)


class HJSAAttack(QueryAttackBase):
    def generate_adv(self, q_dataset, target_model, g_dataset):
        target_model.eval().requires_grad_(False)
        # Only for getting gallery informations
        matcher = Matcher(target_model, g_dataset)
        g_feats_optimized = matcher.g_feats_optimized
        g_pids = matcher.g_pids
        g_camids = matcher.g_camids
        del matcher

        f_model = PyTorchModel(target_model, bounds=(0, 1))
        init_attack = LinearSearchBlendedUniformNoiseAttack(directions=10, steps=10)
        # about 2,000 queries per image
        attack = HopSkipJumpAttack(
            steps=10,
            max_gradient_eval_steps=180,
            constraint="linf",
        )

        eps = 8 / 255

        all_raw_adv_imgs, all_adv_imgs, all_pids, all_camids = [], [], [], []
        q_dataloader = data.DataLoader(q_dataset, batch_size=8, num_workers=8)
        for imgs, pids, camids in tqdm(q_dataloader, desc="Generate adv", leave=False):
            imgs, pids, camids = imgs.cuda(), pids.cuda(), camids.cuda()

            criterion = ReidMisCriterion(
                pids, camids, g_pids, g_camids, g_feats_optimized, topk=10
            )

            starting_points = init_attack.run(f_model, imgs, criterion)
            # Dropping non adversary, to prevent starting points is not adversarial
            # is_adv = get_is_adversarial(criterion, f_model)(starting_points)
            # is_adv_idx = torch.where(is_adv.raw)[0]
            # imgs, pids, camids = (
            #     imgs[is_adv_idx].clone(),
            #     pids[is_adv_idx].clone(),
            #     camids[is_adv_idx].clone(),
            # )
            # starting_points = starting_points[is_adv_idx].clone()
            # criterion.q_pids = criterion.q_pids[is_adv_idx].clone()
            # criterion.q_camids = criterion.q_camids[is_adv_idx].clone()

            no_clipped_advs, clipped_advs, success = attack(
                f_model,
                imgs,
                criterion,
                epsilons=eps,
                starting_points=starting_points,
            )

            # raw_adv_imgs means delta may larger than epislon
            raw_adv_imgs = torch.clamp(no_clipped_advs, 0, 1)
            adv_imgs = torch.clamp(clipped_advs, 0, 1)
            all_raw_adv_imgs.append(raw_adv_imgs.cpu())
            all_adv_imgs.append(adv_imgs.cpu())
            all_pids.append(pids.cpu())
            all_camids.append(camids.cpu())

        all_raw_adv_imgs = torch.cat(all_raw_adv_imgs)
        all_adv_imgs = torch.cat(all_adv_imgs)
        all_pids = torch.cat(all_pids)
        all_camids = torch.cat(all_camids)

        return (
            data.TensorDataset(all_raw_adv_imgs, all_pids, all_camids),
            data.TensorDataset(all_adv_imgs, all_pids, all_camids),
        )

    def run(self):
        logger = logging.getLogger("__main__")
        for dataset_name, (q_dataset, g_dataset) in self.test_datasets.items():
            for target_model_name in self.target_model_names:
                target_model = build_reid_model(target_model_name, dataset_name).cuda()
                target_model = self.accelerator.prepare(target_model)

                (raw_adv_q_dataset, adv_q_dataset), spend_time = timer(
                    self.generate_adv
                )(q_dataset, target_model, g_dataset)

                logger.info(f"Spend Time: {spend_time}")

                raw_vqe_results = self.evaluate_vqe(q_dataset, raw_adv_q_dataset)
                logger.info(f"No Clipped VQE Metrics:\t" + raw_vqe_results)
                vqe_results = self.evaluate_vqe(q_dataset, adv_q_dataset)
                logger.info(f"Clipped VQE Metrics:\t" + vqe_results)

                raw_reid_results = self.evaluate_reid(
                    q_dataset, raw_adv_q_dataset, g_dataset, target_model
                )
                logger.info(
                    f"No Clipped ReID Metrics: {dataset_name} {target_model_name}\n"
                    + raw_reid_results
                )
                reid_results = self.evaluate_reid(
                    q_dataset, adv_q_dataset, g_dataset, target_model
                )
                logger.info(
                    f"Clipped ReID Metrics: {dataset_name} {target_model_name}\n"
                    + reid_results
                )
                torch.cuda.empty_cache()


def main():
    setup_logger(name="pytorch_reid_models.reid_models")
    setup_logger(name="__main__")

    set_seed(42)

    HJSAAttack().run()


if __name__ == "__main__":
    main()
