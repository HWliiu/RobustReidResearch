"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import os
import random
from functools import partial
from pathlib import Path

import accelerate
import kornia as K
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from prettytable import PrettyTable
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import BatchEasyHardMiner
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm.auto import tqdm

from pytorch_reid_models.reid_models.data import (
    build_test_dataloaders,
    build_train_dataloader,
    build_train_dataset,
)
from pytorch_reid_models.reid_models.evaluate import Estimator
from pytorch_reid_models.reid_models.modeling import _build_reid_model
from pytorch_reid_models.reid_models.utils import set_seed, setup_logger


class TIMUAP:
    def __init__(
        self,
        train_dataset_names,
        train_num=128,
        train_epoch=1,
        eps=4 / 255,
        alpha=0.01,
        decay=1.0,
        len_kernel=7,
        nsig=1.5,
        resize_rate=0.9,
        diversity_prob=0.5,
    ):
        self.train_dataset_names = train_dataset_names
        self.train_num = train_num
        self.train_epoch = train_epoch

        self.eps = eps
        self.alpha = alpha
        self.decay = decay
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.len_kernel = (len_kernel, len_kernel)
        self.nsig = (nsig, nsig)

        self.t_dataset = build_train_dataset(train_dataset_names)

    def get_train_dataloader(self):
        sub_t_dataset = Subset(
            self.t_dataset,
            indices=random.sample(range(len(self.t_dataset)), k=self.train_num),
        )
        return DataLoader(sub_t_dataset, batch_size=1, shuffle=True)

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(
            low=img_size, high=img_resize, size=(1,), dtype=torch.int32
        ).item()
        ratio = x.shape[2] / x.shape[3]
        rescaled = F.interpolate(
            x, size=[int(rnd * ratio), rnd], mode="bilinear", align_corners=False
        )
        h_rem = int((img_resize - rnd) * ratio)
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem, size=(1,), dtype=torch.int32).item()
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem, size=(1,), dtype=torch.int32).item()
        pad_right = w_rem - pad_left

        padded = F.pad(
            rescaled,
            [pad_left, pad_right, pad_top, pad_bottom],
            value=0,
        )

        return padded if torch.rand(1) < self.diversity_prob else x

    def __call__(self, model):
        model.eval()

        device = next(iter(model.parameters())).device

        uap = torch.zeros(
            (1, 3, *self.t_dataset[0][0].shape[-2:]), device=device
        ).uniform_(-1e-3, 1e-3)
        momentum = torch.zeros_like(uap)

        criterion = partial(
            torch.nn.CosineEmbeddingLoss(), target=torch.ones(1, device=device)
        )
        for e in range(1, self.train_epoch + 1):
            for imgs, _, _ in tqdm(
                self.get_train_dataloader(),
                desc=f"Train UAP [{e}/{self.train_epoch}]",
                leave=False,
            ):
                imgs = imgs.to(device)

                feats = model(imgs)
                uap.requires_grad_(True)
                adv_imgs = torch.clamp(imgs + uap, 0, 1)
                adv_feats = model(self.input_diversity(adv_imgs))

                loss = criterion(adv_feats, feats)

                grad = torch.autograd.grad(loss, uap)[0]

                grad = K.filters.gaussian_blur2d(
                    grad, kernel_size=self.len_kernel, sigma=self.nsig
                )
                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                grad = grad + momentum * self.decay
                momentum = grad

                uap = uap.detach() + self.alpha * grad.sign()
                uap.clamp_(-self.eps, self.eps)

        model.train()

        return uap


def adv_train(
    accelerator,
    model,
    train_loader,
    optimizer,
    miner,
    criterion_t,
    criterion_x,
    max_epoch,
    epoch,
    uap_attacker,
    update_uap_freq=1,
):
    model.train()
    bar = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch[{epoch}/{max_epoch}]",
        leave=False,
    )

    gen_uap_idx = torch.linspace(
        0, len(train_loader), update_uap_freq, dtype=torch.int64
    )

    for batch_idx, (imgs, pids, camids) in enumerate(bar):
        if batch_idx in gen_uap_idx:
            uap = uap_attacker(model)

        imgs, pids, camids = imgs.cuda(), pids.cuda(), camids.cuda()

        adv_imgs = torch.clamp(imgs + uap, 0, 1)
        logits, feats = model(adv_imgs)

        pairs = miner(feats, pids)
        loss_t = criterion_t(feats, pids, pairs)
        loss_x = criterion_x(logits, pids)

        loss = loss_t + loss_x
        optimizer.zero_grad(True)
        loss.backward()
        # accelerator.backward(loss)
        optimizer.step()

        acc = (logits.max(1)[1] == pids).float().mean()
        bar.set_postfix_str(f"loss:{loss.item():.1f} " f"acc:{acc.item():.1f}")
        bar.update()
    bar.close()


def main():
    setup_logger(name="pytorch_reid_models.reid_models")
    logger = setup_logger(name="__main__")

    seed = 42
    set_seed(seed)

    accelerator = accelerate.Accelerator(mixed_precision="fp16")

    test_dataset_names = ["market1501"]
    test_loaders = build_test_dataloaders(dataset_names=test_dataset_names)
    train_dataset_names = ["market1501"]
    train_loader = build_train_dataloader(
        dataset_names=train_dataset_names,
        transforms=["randomflip", "randomcrop", "rea"],
        batch_size=64,
        sampler="pk",
        num_instance=4,
    )

    model_name = "bagtricks_R50_fastreid"
    num_classes_dict = {"dukemtmcreid": 702, "market1501": 751, "msmt17": 1041}
    num_classes = sum([num_classes_dict[name] for name in train_dataset_names])
    # TODO: Make sure load pretrained model
    os.environ["pretrain"] = "1"
    model = _build_reid_model(
        model_name,
        num_classes=num_classes,
    )
    model = accelerator.prepare(model)

    save_dir = Path(f"logs/uap_at/{model_name}/{'_'.join(train_dataset_names)}")
    save_dir.mkdir(parents=True, exist_ok=True)

    max_epoch = 60
    optimizer = torch.optim.Adam(model.parameters(), lr=3.5e-4, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=7e-7)

    miner = BatchEasyHardMiner()
    criterion_t = TripletMarginLoss(margin=0.3)
    criterion_x = nn.CrossEntropyLoss(label_smoothing=0.1)

    uap_attacker = TIMUAP(train_dataset_names)

    for epoch in range(1, max_epoch + 1):
        adv_train(
            accelerator,
            model,
            train_loader,
            optimizer,
            miner,
            criterion_t,
            criterion_x,
            max_epoch,
            epoch,
            uap_attacker,
        )

        scheduler.step()

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                save_dir / f"{'_'.join(train_dataset_names)}-{model_name}.pth",
            )
            results = PrettyTable(
                field_names=[f"epoch {epoch:0>2}", "top1", "top5", "mAP", "mINP"]
            )
            for name, loader in test_loaders.items():
                cmc, mAP, mINP = Estimator(model, loader[1])(loader[0])
                results.add_row(
                    [
                        name,
                        f"{cmc[0]:.3f}",
                        f"{cmc[4]:.3f}",
                        f"{mAP:.3f}",
                        f"{mINP:.3f}",
                    ]
                )

            logger.info("\n" + str(results))


if __name__ == "__main__":
    main()
