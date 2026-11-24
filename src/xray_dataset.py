"""xray_dataset.py - turns one row of data/xray/xray_studies.csv into what the model needs.

One item = (image as numbers, report as token ids, abnormal label 0/1).
Used by Step 4 (training), Step 5 (evaluation) and the demo app (image preparation).
"""
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from report_model import MAX_TOKENS

# Training images get small random changes so the model does not memorise them.
# We never flip a chest X-ray: left and right matter (the heart is on the left).
TRAIN_TRANSFORM = T.Compose([T.Resize((240, 240)), T.RandomCrop(224), T.RandomRotation(5),
                             T.ColorJitter(brightness=0.1, contrast=0.1), T.ToTensor(),
                             T.Normalize([0.5] * 3, [0.5] * 3)])
EVAL_TRANSFORM = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])


def load_image(path):
    return Image.open(path).convert("RGB")      # X-rays are grey; ViT expects three channels


class XrayDataset(Dataset):
    def __init__(self, studies, image_dir, tokenizer, transform):
        self.studies = studies.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        enc = tokenizer(list(self.studies["report_text"]), max_length=MAX_TOKENS, truncation=True,
                        padding="max_length", return_tensors="pt")
        self.labels = enc.input_ids
        self.labels[self.labels == tokenizer.pad_token_id] = -100     # padding is ignored by the loss

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        row = self.studies.iloc[i]
        pixel_values = self.transform(load_image(self.image_dir / row["image_file"]))
        return pixel_values, self.labels[i], torch.tensor(float(row["label_abnormal"]))
