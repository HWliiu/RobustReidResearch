"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import argparse
import os
from pathlib import Path

import accelerate
import torch
import torch.nn as nn
from pytorch_metric_learning.losses import TripletMarginLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image
from tqdm.auto import tqdm
from pytorch_reid_models.reid_models.data import (
    build_test_datasets,
    build_train_dataloader,
)
from pytorch_reid_models.reid_models.modeling import _build_reid_model
from pytorch_reid_models.reid_models.utils import set_seed, setup_logger
from reid_defense.eval_attack import test


class UAPManager:
    def __init__(
        self,
        eps=8 / 255,
        alpha=1 / 255,
        decay=1.0,
        len_kernel=7,
        nsig=1.5,
        size=(256, 128),
    ):
        self._eps = eps
        self._alpha = alpha
        self._decay = decay
        self._len_kernel = (len_kernel, len_kernel)
        self._nsig = (nsig, nsig)
        self._size = size
        self._device = "cuda"

        self._uap = torch.zeros((1, 3, *self._size), device=self._device)
        self._momentum = torch.zeros_like(self._uap)

        self.restart()

    @property
    def uap(self):
        self._uap.requires_grad_(True)
        return self._uap

    def zero_grad(self):
        self._uap.grad = None

    @torch.no_grad()
    def restart(self):
        self._uap.uniform_(-self._eps, self._eps)
        self._momentum.zero_()

    @torch.no_grad()
    def update(self):
        self._uap.detach_()
        grad = self._uap.grad
        # grad = K.filters.gaussian_blur2d(
        #     grad, kernel_size=self._len_kernel, sigma=self._nsig
        # )
        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        grad = grad + self._momentum * self._decay
        self._momentum = grad

        self._uap += self._alpha * grad.sign()
        self._uap.clamp_(-self._eps, self._eps)


def adv_train(
    accelerator,
    model,
    train_loader,
    optimizer,
    criterion_t,
    criterion_x,
    max_epoch,
    epoch,
    uap_manager,
    restaet_uap_freq,
    apply_uap_rate,
):
    model.train()
    bar = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch[{epoch}/{max_epoch}]",
        leave=False,
    )

    restart_uap_idx = torch.linspace(
        0, len(train_loader), restaet_uap_freq, dtype=torch.int64
    )

    for batch_idx, (imgs, pids, camids) in enumerate(bar):
        if batch_idx in restart_uap_idx:
            uap_manager.restart()

        imgs, pids, camids = imgs.cuda(), pids.cuda(), camids.cuda()

        masks = torch.bernoulli(torch.ones_like(pids) * apply_uap_rate)
        adv_imgs = torch.clamp(
            imgs + uap_manager.uap * masks[:, None, None, None], 0, 1
        )
        logits, feats = model(adv_imgs)

        loss_t = criterion_t(feats, pids)
        loss_x = criterion_x(logits, pids)

        loss = loss_t + loss_x
        optimizer.zero_grad(True)
        uap_manager.zero_grad()
        loss.backward()
        optimizer.step()
        uap_manager.update()

        acc = (logits.max(1)[1] == pids).float().mean()
        bar.set_postfix_str(f"loss:{loss.item():.1f} " f"acc:{acc.item():.1f}")
        bar.update()
    bar.close()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--restaet_uap_freq", type=int, default=1)
    parser.add_argument("--apply_uap_rate", type=float, default=0.8)
    args = parser.parse_args()

    setup_logger(name="pytorch_reid_models.reid_models")
    logger = setup_logger(name="__main__")

    seed = 42
    set_seed(seed)

    accelerator = accelerate.Accelerator(mixed_precision="fp16")

    dataset_name = "dukemtmcreid"
    test_dataset = build_test_datasets(dataset_names=[dataset_name], query_num=500)[
        dataset_name
    ]
    train_loader = build_train_dataloader(
        dataset_names=[dataset_name],
        transforms=["randomflip", "randomcrop", "rea"],
        batch_size=64,
        sampler="pk",
        num_instance=4,
    )

    model_name = "bagtricks_R50_fastreid"
    num_classes_dict = {"dukemtmcreid": 702, "market1501": 751, "msmt17": 1041}
    num_classes = num_classes_dict[dataset_name]
    # Make sure load pretrained model
    os.environ["pretrain"] = "1"
    model = _build_reid_model(
        model_name,
        num_classes=num_classes,
    ).cuda()
    model = accelerator.prepare(model)

    max_epoch = 60
    optimizer = torch.optim.Adam(model.parameters(), lr=3.5e-4, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=7e-7)

    criterion_t = TripletMarginLoss(margin=0.3)
    criterion_x = nn.CrossEntropyLoss(label_smoothing=0.1)

    save_dir = Path(f"logs/fast_uap_at")
    save_dir.mkdir(parents=True, exist_ok=True)

    uap_manager = UAPManager()

    for epoch in range(1, max_epoch + 1):
        adv_train(
            accelerator,
            model,
            train_loader,
            optimizer,
            criterion_t,
            criterion_x,
            max_epoch,
            epoch,
            uap_manager,
            restaet_uap_freq=args.restaet_uap_freq,
            apply_uap_rate=args.apply_uap_rate,
        )

        scheduler.step()

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                save_dir
                / f"{dataset_name}-{model_name}_{args.restaet_uap_freq}_{args.apply_uap_rate}.pth",
            )
            results = test(test_dataset, model)
            logger.info(f"Epoch {epoch:0>2} evaluate results:\n" + results)


if __name__ == "__main__":
    main()
