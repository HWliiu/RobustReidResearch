"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import torch
from foolbox.distances import l2
from torch.utils.data import DataLoader
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class VQE:
    def __init__(self, device="cuda"):
        self.device = device

        self.lpips = LearnedPerceptualImagePatchSimilarity(
            reduction="sum", normalize=True
        ).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=1.0, reduction="sum"
        ).to(self.device)
        self.psnr = PeakSignalNoiseRatio(
            data_range=1.0, reduction="sum", dim=(1, 2, 3)
        ).to(self.device)

    @torch.no_grad()
    def __call__(self, q_dataset, adv_q_dataset):
        sum_l2_dist = 0.0
        sum_lpips_value = 0.0
        sum_ssim_value = 0.0
        sum_psnr_value = 0.0
        total = 0

        q_data_loader = DataLoader(q_dataset, batch_size=128, num_workers=8)
        adv_q_data_loader = DataLoader(adv_q_dataset, batch_size=128, num_workers=8)
        for (imgs, _, _), (adv_imgs, _, _) in zip(q_data_loader, adv_q_data_loader):
            imgs, adv_imgs = imgs.to(self.device), adv_imgs.to(self.device)
            l2_dist = l2(imgs, adv_imgs).sum()
            sum_l2_dist += l2_dist

            lpips_value = self.lpips(imgs, adv_imgs)
            sum_lpips_value += lpips_value

            ssim_value = self.ssim(imgs, adv_imgs)
            sum_ssim_value += ssim_value

            psnr_value = self.psnr(imgs, adv_imgs)
            sum_psnr_value += psnr_value

            total += len(imgs)

        l2_dist = sum_l2_dist / total
        lpips_value = sum_lpips_value / total
        ssim_value = sum_ssim_value / total
        psnr_value = sum_psnr_value / total

        return l2_dist.item(), lpips_value.item(), ssim_value.item(), psnr_value.item()
