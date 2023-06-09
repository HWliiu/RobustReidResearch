import pathlib

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class NIPS2017Dataset(Dataset):
    # Download link https://www.kaggle.com/competitions/nips-2017-non-targeted-adversarial-attack/data
    def __init__(self, root_dir, transform=None):
        self.root_dir = pathlib.Path(root_dir)
        self.transform = transform
        self.categories = pd.read_csv(self.root_dir / "categories.csv")
        self.images = pd.read_csv(self.root_dir / "images.csv")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = (
            self.root_dir / "images" / (self.images.iloc[idx]["ImageId"] + ".png")
        )
        image = Image.open(img_name).convert("RGB")
        label = self.images.iloc[idx]["TrueLabel"]
        if self.transform:
            image = self.transform(image)
        return image, label - 1
