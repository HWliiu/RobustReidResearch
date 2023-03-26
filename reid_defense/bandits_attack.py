"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import logging

import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision.utils import save_image
from tqdm.auto import tqdm

from pytorch_reid_models.reid_models.data import build_test_datasets
from pytorch_reid_models.reid_models.modeling import _build_reid_model, build_reid_model
from pytorch_reid_models.reid_models.utils import set_seed, setup_logger
from reid_attack.attacker_base import timer
from reid_attack.bandits_attack import BanditsAttack


class OverloadAttack(BanditsAttack):
    def run(self):
        logger = logging.getLogger("__main__")
        test_datasets = build_test_datasets(
            dataset_names=self.target_dataset_names, query_num=self.query_num
        )

        for dataset_name, (q_dataset, g_dataset) in test_datasets.items():
            num_classes_dict = {"dukemtmcreid": 702, "market1501": 751, "msmt17": 1041}
            weights_path_dict = {"bagtricks_R50_fastreid": {"dukemtmcreid": ""}}

            for target_model_name in self.target_model_names:
                target_model = _build_reid_model(
                    target_model_name, num_classes=num_classes_dict["dataset_name"]
                ).cuda()
                weight_path = weights_path_dict[target_model_name][dataset_name]
                target_model.load_state_dict(
                    torch.load(weight_path, map_location="cpu")
                )
                target_model = self.accelerator.prepare(target_model)

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


def main():
    setup_logger(name="pytorch_reid_models.reid_models")
    setup_logger(name="__main__")

    set_seed(42)

    OverloadAttack().run()


if __name__ == "__main__":
    main()
