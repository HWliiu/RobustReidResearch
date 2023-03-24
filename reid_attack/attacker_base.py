"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""
import logging
import time
import warnings

import accelerate
import torch
from prettytable import PrettyTable
from torch.utils import data

from pytorch_reid_models.reid_models.data import build_test_datasets
from pytorch_reid_models.reid_models.evaluate import Estimator
from pytorch_reid_models.reid_models.modeling import build_reid_model
from reid_attack.vqe import VQE

warnings.filterwarnings("ignore")


class EvaluateReIDMixin:
    def _evaluate_reid(self, q_dataset, adv_q_dataset, g_dataset, target_model):
        q_dataloader = data.DataLoader(q_dataset, batch_size=128, num_workers=8)
        adv_q_dataloader = data.DataLoader(adv_q_dataset, batch_size=128, num_workers=8)
        g_dataloader = data.DataLoader(g_dataset, batch_size=128, num_workers=8)

        estimator = Estimator(target_model, g_dataloader)
        before_cmc, before_mAP, before_mINP = estimator(q_dataloader)
        after_cmc, after_mAP, after_mINP = estimator(adv_q_dataloader)

        return (
            (before_cmc, before_mAP, before_mINP),
            (after_cmc, after_mAP, after_mINP),
        )

    def evaluate_reid(
        self,
        q_dataset,
        adv_q_dataset,
        g_dataset,
        target_model,
    ):
        (
            (before_cmc, before_mAP, before_mINP),
            (after_cmc, after_mAP, after_mINP),
        ) = self._evaluate_reid(q_dataset, adv_q_dataset, g_dataset, target_model)

        results = PrettyTable(field_names=["", "top1", "top5", "mAP", "mINP"])
        results.add_row(
            [
                "before",
                f"{before_cmc[0]:.3f}",
                f"{before_cmc[4]:.3f}",
                f"{before_mAP:.3f}",
                f"{before_mINP:.3f}",
            ]
        )
        results.add_row(
            [
                "after",
                f"{after_cmc[0]:.3f}",
                f"{after_cmc[4]:.3f}",
                f"{after_mAP:.3f}",
                f"{after_mINP:.3f}",
            ]
        )
        return str(results)


class EvaluateVQEMixin:
    def evaluate_vqe(
        self,
        q_dataset,
        adv_q_dataset,
    ):
        l2_dist, lpips_value, ssim_value, psnr_value = VQE()(q_dataset, adv_q_dataset)
        return (
            f"l2↓ {l2_dist:.2f}\tlpips↓ {lpips_value:.2f}\t"
            f"ssim↑ {ssim_value:.2f}\tpsnr↑ {psnr_value:.2f}"
        )


class EvaluateMixin(EvaluateReIDMixin, EvaluateVQEMixin):
    pass


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_diff = end_time - start_time

        minutes = int(time_diff // 60)
        seconds = int(time_diff % 60)
        spend_time = f"{minutes}m{seconds}s"
        return result, spend_time

    return wrapper


class TransferAttackBase(EvaluateMixin):
    def __init__(
        self,
        agent_model_name="bagtricks_inception_v3_fastreid",
        target_model_names=(
            # Backbone
            "bagtricks_R50_fastreid",
            "bagtricks_osnet_x1_0_fastreid",
            "bagtricks_S50_fastreid",
            "bagtricks_SeR50_fastreid",
            "bagtricks_densenet121_fastreid",
            "bagtricks_mobilenet_v3_large_fastreid",
            "bagtricks_inception_resnet_v2_fastreid",
            "bagtricks_inception_v4_fastreid",
            # SOTA Reid
            "resnet50_abd",
            "resnet50_agw",
            "resnet50_ap",
            "osnet_x1_0_dpr",
            "osnet_ibn_x1_0_dpr",
            "resnet50_bot",
            "vit_transreid",
            # SBS
            "sbs_R50_fastreid",
            "sbs_R50_ibn_fastreid",
            "sbs_S50_fastreid",
        ),
        target_dataset_names=("dukemtmcreid", "market1501", "msmt17"),
        query_num=500,
    ):
        self.agent_model_name = agent_model_name
        self.target_model_names = target_model_names
        self.target_dataset_names = target_dataset_names
        self.query_num = query_num

        # It is only used to evaluate the results, and it will greatly affect the effect during training
        self.accelerator = accelerate.Accelerator(mixed_precision="fp16")

    def generate_adv(self, q_dataset, agent_model):
        raise NotImplementedError

    def run(self):
        logger = logging.getLogger("__main__")
        test_datasets = build_test_datasets(
            dataset_names=self.target_dataset_names, query_num=self.query_num
        )
        for dataset_name, (q_dataset, g_dataset) in test_datasets.items():
            agent_model = build_reid_model(self.agent_model_name, dataset_name).cuda()

            adv_q_dataset, spend_time = timer(self.generate_adv)(q_dataset, agent_model)

            logger.info(f"Spend Time: {spend_time}")
            vqe_results = self.evaluate_vqe(q_dataset, adv_q_dataset)
            logger.info(f"VQE Metrics:\t" + vqe_results)

            for target_model_name in self.target_model_names:
                target_model = build_reid_model(target_model_name, dataset_name).cuda()
                target_model = self.accelerator.prepare(target_model)

                reid_results = self.evaluate_reid(
                    q_dataset, adv_q_dataset, g_dataset, target_model
                )
                logger.info(
                    f"ReID Metrics: {dataset_name} {self.agent_model_name}-->{target_model_name}\n"
                    + reid_results
                )
                torch.cuda.empty_cache()


class EnsTransferAttackBase(EvaluateMixin):
    def __init__(
        self,
        agent_model_names=(
            "bagtricks_inception_v3_fastreid",
            "bagtricks_inception_v4_fastreid",
            "bagtricks_inception_resnet_v2_fastreid",
            "bagtricks_mobilenet_v3_large_fastreid",
        ),
        target_model_names=(
            # Backbone
            "bagtricks_R50_fastreid",
            "bagtricks_osnet_x1_0_fastreid",
            "bagtricks_SeR50_fastreid",
            "bagtricks_S50_fastreid",
            "bagtricks_densenet121_fastreid"
            # SOTA Reid
            "resnet50_abd",
            "resnet50_agw",
            "resnet50_ap",
            "osnet_x1_0_dpr",
            "osnet_ibn_x1_0_dpr",
            "resnet50_bot",
            "vit_transreid",
            # SBS
            "sbs_R50_fastreid",
            "sbs_R50_ibn_fastreid",
            "sbs_S50_fastreid",
        ),
        target_dataset_names=("dukemtmcreid", "market1501", "msmt17"),
        query_num=500,
    ):
        self.agent_model_names = agent_model_names
        self.target_model_names = target_model_names
        self.target_dataset_names = target_dataset_names
        self.query_num = query_num

        # only for evaluation
        self.accelerator = accelerate.Accelerator(mixed_precision="fp16")

    def generate_adv(self, q_dataset, agent_models):
        raise NotImplementedError

    def run(self):
        logger = logging.getLogger("__main__")
        test_datasets = build_test_datasets(
            dataset_names=self.target_dataset_names, query_num=self.query_num
        )
        for dataset_name, (q_dataset, g_dataset) in test_datasets.items():
            agent_models = [
                build_reid_model(agent_model_name, dataset_name).cuda()
                for agent_model_name in self.agent_model_names
            ]
            adv_q_dataset, spend_time = timer(self.generate_adv)(
                q_dataset, agent_models
            )

            logger.info(f"Spend Time: {spend_time}")

            vqe_results = self.evaluate_vqe(q_dataset, adv_q_dataset)
            logger.info(f"VQE Metrics:\t" + vqe_results)

            for target_model_name in self.target_model_names:
                target_model = build_reid_model(target_model_name, dataset_name).cuda()
                target_model = self.accelerator.prepare(target_model)

                reid_results = self.evaluate_reid(
                    q_dataset, adv_q_dataset, g_dataset, target_model
                )
                logger.info(
                    f"ReID Metrics: {dataset_name} {'|'.join(self.agent_model_names)}"
                    f"-->{target_model_name}\n" + reid_results
                )
                torch.cuda.empty_cache()


class QueryAttackBase(EvaluateMixin):
    def __init__(
        self,
        target_model_names=(
            # Backbone
            "bagtricks_R50_fastreid",
            "bagtricks_osnet_x1_0_fastreid",
            "bagtricks_S50_fastreid",
            "bagtricks_SeR50_fastreid",
            "bagtricks_densenet121_fastreid",
            "bagtricks_mobilenet_v3_large_fastreid",
            "bagtricks_inception_resnet_v2_fastreid",
            "bagtricks_inception_v4_fastreid",
            # SOTA Reid
            "resnet50_abd",
            "resnet50_agw",
            "resnet50_ap",
            "osnet_x1_0_dpr",
            "osnet_ibn_x1_0_dpr",
            "resnet50_bot",
            "vit_transreid",
            # SBS
            "sbs_R50_fastreid",
            "sbs_R50_ibn_fastreid",
            "sbs_S50_fastreid",
        ),
        target_dataset_names=("dukemtmcreid", "market1501", "msmt17"),
        query_num=100,
    ):
        self.target_model_names = target_model_names
        self.target_dataset_names = target_dataset_names
        self.query_num = query_num

        self.accelerator = accelerate.Accelerator(mixed_precision="fp16")

    def generate_adv(self, q_dataset, target_model, g_dataset):
        raise NotImplementedError

    def run(self):
        logger = logging.getLogger("__main__")
        test_datasets = build_test_datasets(
            dataset_names=self.target_dataset_names, query_num=self.query_num
        )
        for dataset_name, (q_dataset, g_dataset) in test_datasets.items():
            for target_model_name in self.target_model_names:
                target_model = build_reid_model(target_model_name, dataset_name).cuda()

                adv_q_dataset, spend_time = timer(self.generate_adv)(
                    q_dataset, target_model, g_dataset
                )

                target_model = self.accelerator.prepare(target_model)

                logger.info(f"Spend Time: {spend_time}")

                vqe_results = self.evaluate_vqe(q_dataset, adv_q_dataset)
                logger.info(f"VQE Metrics:\t" + vqe_results)

                reid_results = self.evaluate_reid(
                    q_dataset, adv_q_dataset, g_dataset, target_model
                )
                logger.info(
                    f"ReID Metrics: {dataset_name} {target_model_name}\n" + reid_results
                )
                torch.cuda.empty_cache()
