"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import os
from pathlib import Path

import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def adv_train(
    model,
    train_loader,
    optimizer,
    criterion_t,
    criterion_x,
    max_epoch,
    epoch,
    # adv parameters
    adv_step,
    alpha=2 / 255,
    eps=4 / 255,
):
    model.train()
    bar = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch[{epoch}/{max_epoch}]",
        leave=False,
    )

    for batch_idx, (imgs, pids, camids) in enumerate(bar):
        imgs, pids, camids = imgs.cuda(), pids.cuda(), camids.cuda()

        model.eval()
        with torch.no_grad():
            feats = model(imgs)
        adv_imgs = imgs.clone() + torch.empty_like(imgs).uniform_(-1e-2, 1e-2)
        for _ in range(adv_step):
            adv_imgs.requires_grad_(True)
            adv_feats = model(adv_imgs)

            loss = (F.normalize(adv_feats) * F.normalize(feats)).sum(dim=1).mean()
            grad = torch.autograd.grad(loss, adv_imgs)[0]

            # Update adversaries
            adv_imgs = adv_imgs.detach() - alpha * grad.sign()
            delta = torch.clamp(adv_imgs - imgs, min=-eps, max=eps)
            adv_imgs = torch.clamp(imgs + delta, min=0, max=1).detach()
        model.train()

        logits, feats = model(adv_imgs)

        loss_t = criterion_t(feats, pids)
        loss_x = criterion_x(logits, pids)

        loss = loss_t + loss_x
        optimizer.zero_grad(True)
        loss.backward()
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

    accelerator = accelerate.Accelerator(mixed_precision="no")

    dataset_name = "market1501"
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

    adv_steps = 4
    max_epoch = 60
    optimizer = torch.optim.Adam(model.parameters(), lr=3.5e-4, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=7e-7)

    criterion_t = TripletMarginLoss(margin=0.3)
    criterion_x = nn.CrossEntropyLoss(label_smoothing=0.1)

    save_dir = Path(f"logs/pgd_at_cos")
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epoch + 1):
        adv_train(
            model,
            train_loader,
            optimizer,
            criterion_t,
            criterion_x,
            max_epoch,
            epoch,
            adv_steps,
        )

        scheduler.step()

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                save_dir / f"{dataset_name}-{model_name}.pth",
            )
            results = test(test_dataset, model)
            logger.info(f"Epoch {epoch:0>2} evaluate results:\n" + results)


if __name__ == "__main__":
    main()
