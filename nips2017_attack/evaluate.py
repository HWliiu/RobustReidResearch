"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import accelerate
import torch
import torch.nn as nn
import torchmetrics
import torchvision.models as models
import torchvision.transforms as T
from kornia.enhance import Normalize
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from nips2017_attack.nips2017_dataset import NIPS2017Dataset


def evaluate(data_loader, model):
    device = next(model.parameters()).device
    accuracy = torchmetrics.MetricCollection(
        {
            "top1": torchmetrics.Accuracy("multiclass", num_classes=1000, top_k=1),
            "top5": torchmetrics.Accuracy("multiclass", num_classes=1000, top_k=5),
        },
        compute_groups=False,
    ).to(device)

    for imgs, lbls in tqdm(data_loader, desc="Evaluate", leave=False):
        imgs = imgs.to(device)
        lbls = lbls.to(device)

        with torch.no_grad():
            preds = model(imgs)
            preds = nn.functional.softmax(preds, dim=1)
            accuracy.update(preds, lbls)
    results = {k: v.item() for k, v in accuracy.compute().items()}
    accuracy.reset()
    return list(results.values())


def main():
    accelerator = accelerate.Accelerator(mixed_precision="fp16")

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    data_loader = DataLoader(
        NIPS2017Dataset("datasets/archive", transform), batch_size=64, num_workers=8
    )

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).cuda()
    normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    model = nn.Sequential(normalize, model).eval()

    data_loader, model = accelerator.prepare(data_loader, model)

    top1, top5 = evaluate(data_loader, model)
    print(f"original top1:{top1:.2f} top5:{top5:.2f}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
