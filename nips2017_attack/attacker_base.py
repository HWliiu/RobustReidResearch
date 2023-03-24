"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""
import logging
import time
import warnings

import accelerate
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from kornia.enhance import Normalize
from prettytable import PrettyTable
from torch.utils import data

from nips2017_attack.evaluate import evaluate
from nips2017_attack.nips2017_dataset import NIPS2017Dataset
from nips2017_attack.vqe import VQE

warnings.filterwarnings("ignore")


class EvaluateImagenetMixin:
    def _evaluate_imagenet(self, test_dataset, adv_test_dataset, target_model):
        test_dataloader = data.DataLoader(test_dataset, batch_size=128, num_workers=8)
        adv_test_dataloader = data.DataLoader(
            adv_test_dataset, batch_size=128, num_workers=8
        )

        before_top1, before_top5 = evaluate(test_dataloader, target_model)
        after_top1, after_top5 = evaluate(adv_test_dataloader, target_model)

        return (
            (before_top1, before_top5),
            (after_top1, after_top5),
        )

    def evaluate_imagenet(
        self,
        test_dataset,
        adv_test_dataset,
        target_model,
    ):
        (
            (before_top1, before_top5),
            (after_top1, after_top5),
        ) = self._evaluate_imagenet(test_dataset, adv_test_dataset, target_model)

        results = PrettyTable(field_names=["", "top1", "top5"])
        results.add_row(["before", f"{before_top1:.3f}", f"{before_top5:.3f}"])
        results.add_row(["after", f"{after_top1:.3f}", f"{after_top5:.3f}"])

        return str(results)


class EvaluateVQEMixin:
    def evaluate_vqe(
        self,
        test_dataset,
        adv_test_dataset,
    ):
        l2_dist, lpips_value, ssim_value, psnr_value = VQE()(
            test_dataset, adv_test_dataset
        )
        return (
            f"l2↓ {l2_dist:.2f}\tlpips↓ {lpips_value:.2f}\t"
            f"ssim↑ {ssim_value:.2f}\tpsnr↑ {psnr_value:.2f}"
        )


class EvaluateMixin(EvaluateImagenetMixin, EvaluateVQEMixin):
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


def timm_model_wrapper(model):
    input_config = timm.data.resolve_data_config(model.pretrained_cfg)
    assert 200 < input_config["input_size"][2] < 300
    mean = input_config["mean"]
    std = input_config["std"]
    return nn.Sequential(Normalize(mean, std), model)


def build_test_dataset(test_num):
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    test_dataset = NIPS2017Dataset("datasets/archive", transform)
    test_dataset = data.Subset(
        test_dataset,
        indices=torch.linspace(
            0, len(test_dataset) - 1, steps=test_num, dtype=torch.int64
        ).tolist(),
    )
    return test_dataset


class TransferAttackBase(EvaluateMixin):
    def __init__(
        self,
        agent_model_name="inception_v3",
        target_model_names=(
            "resnet50",
            "densenet121",
            "inception_resnet_v2",
            "inception_v4",
            "adv_inception_v3",
            "mobilenetv3_large_100",
            "seresnet50",
        ),
        test_num=500,
    ):
        self.agent_model_name = agent_model_name
        self.target_model_names = target_model_names
        self.test_num = test_num

        self.accelerator = accelerate.Accelerator(mixed_precision="fp16")

    def generate_adv(self, test_dataset, agent_model):
        raise NotImplementedError

    def run(self):
        logger = logging.getLogger("__main__")

        test_dataset = build_test_dataset(self.test_num)
        agent_model = timm_model_wrapper(
            timm.create_model(self.agent_model_name, pretrained=True)
        ).cuda()

        adv_test_dataset, spend_time = timer(self.generate_adv)(
            test_dataset, agent_model
        )

        logger.info(f"Spend Time: {spend_time}")
        vqe_results = self.evaluate_vqe(test_dataset, adv_test_dataset)
        logger.info(f"VQE Metrics:\t" + vqe_results)

        for target_model_name in self.target_model_names:
            target_model = timm_model_wrapper(
                timm.create_model(target_model_name, pretrained=True).eval().cuda()
            )
            target_model = self.accelerator.prepare(target_model)

            imagenet_results = self.evaluate_imagenet(
                test_dataset, adv_test_dataset, target_model
            )
            logger.info(
                f"ImageNet Metrics: {self.agent_model_name}-->{target_model_name}\n"
                + imagenet_results
            )
            torch.cuda.empty_cache()


# TODO: Need To Modify
# class EnsTransferAttackBase(EvaluateMixin):
#     def __init__(
#         self,
#         agent_model_names=(
#             "bagtricks_inception_v3_fastreid",
#             "bagtricks_inception_v4_fastreid",
#             "bagtricks_inception_resnet_v2_fastreid",
#             "bagtricks_mobilenet_v3_large_fastreid",
#         ),
#         target_model_names=(
#             # Backbone
#             "bagtricks_R50_fastreid",
#             "bagtricks_osnet_x1_0_fastreid",
#             "bagtricks_S50_fastreid",
#             "bagtricks_densenet121_fastreid"
#             # SOTA Reid
#             "resnet50_abd",
#             "resnet50_agw",
#             "resnet50_ap",
#             "osnet_x1_0_dpr",
#             "osnet_ibn_x1_0_dpr",
#             "resnet50_bot",
#             "vit_transreid",
#             # SBS
#             "sbs_R50_fastreid",
#             "sbs_R50_ibn_fastreid",
#             "sbs_S50_fastreid",
#         ),
#         target_dataset_names=("dukemtmcreid", "market1501", "msmt17"),
#         query_num=500,
#     ):
#         self.agent_model_names = agent_model_names
#         self.target_model_names = target_model_names
#         self.test_datasets = build_test_datasets(
#             dataset_names=target_dataset_names, query_num=query_num
#         )
#         # only for evaluation
#         self.accelerator = accelerate.Accelerator(mixed_precision="fp16")

#     def generate_adv(self, q_dataset, agent_models):
#         raise NotImplementedError

#     def run(self):
#         logger = logging.getLogger("__main__")
#         for dataset_name, (q_dataset, g_dataset) in self.test_datasets.items():
#             agent_models = [
#                 build_reid_model(agent_model_name, dataset_name).cuda()
#                 for agent_model_name in self.agent_model_names
#             ]
#             adv_q_dataset, spend_time = timer(self.generate_adv)(
#                 q_dataset, agent_models
#             )

#             logger.info(f"Spend Time: {spend_time}")

#             vqe_results = self.evaluate_vqe(q_dataset, adv_q_dataset)
#             logger.info(f"VQE Metrics:\t" + vqe_results)

#             for target_model_name in self.target_model_names:
#                 target_model = build_reid_model(target_model_name, dataset_name).cuda()
#                 target_model = self.accelerator.prepare(target_model)

#                 reid_results = self.evaluate_reid(
#                     q_dataset, adv_q_dataset, g_dataset, target_model
#                 )
#                 logger.info(
#                     f"ReID Metrics: {dataset_name} {'|'.join(self.agent_model_names)}"
#                     f"-->{target_model_name}\n" + reid_results
#                 )
#                 torch.cuda.empty_cache()


class QueryAttackBase(EvaluateMixin):
    def __init__(
        self,
        target_model_names=(
            "resnet50",
            "densenet121",
            "inception_resnet_v2",
            "inception_v3",
            "inception_v4",
            "adv_inception_v3",
            "mobilenetv3_large_100",
            "seresnet50",
        ),
        test_num=100,
    ):
        self.target_model_names = target_model_names
        self.test_num = test_num

        self.accelerator = accelerate.Accelerator(mixed_precision="fp16")

    def generate_adv(self, test_dataset, target_model):
        raise NotImplementedError

    def run(self):
        logger = logging.getLogger("__main__")

        test_dataset = build_test_dataset(self.test_num)

        for target_model_name in self.target_model_names:
            target_model = timm_model_wrapper(
                timm.create_model(target_model_name, pretrained=True).eval().cuda()
            )

            adv_test_dataset, spend_time = timer(self.generate_adv)(
                test_dataset, target_model
            )

            logger.info(f"Spend Time: {spend_time}")
            vqe_results = self.evaluate_vqe(test_dataset, adv_test_dataset)
            logger.info(f"VQE Metrics:\t" + vqe_results)

            target_model = self.accelerator.prepare(target_model)

            imagenet_results = self.evaluate_imagenet(
                test_dataset, adv_test_dataset, target_model
            )
            logger.info(f"ImageNet Metrics: {target_model_name}\n" + imagenet_results)
            torch.cuda.empty_cache()
