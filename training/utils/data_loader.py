import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DeepfakeDataset(Dataset):
    """Dataset for deepfake detection training.
    
    Expected directory structure:
        data_dir/
            real/
                img001.jpg
                img002.jpg
            fake/
                img001.jpg
                img002.jpg
    """

    def __init__(self, data_dir: str, split: str = "train", input_size: int = 224):
        self.data_dir = data_dir
        self.input_size = input_size
        self.samples = []
        self.labels = []

        for label, class_name in enumerate(["real", "fake"]):
            class_dir = os.path.join(data_dir, split, class_name)
            if not os.path.exists(class_dir):
                class_dir = os.path.join(data_dir, class_name)

            if os.path.exists(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        self.samples.append(os.path.join(class_dir, fname))
                        self.labels.append(label)

        self.transform = self._get_transforms(split)

    def _get_transforms(self, split: str) -> A.Compose:
        if split == "train":
            return A.Compose([
                A.Resize(self.input_size, self.input_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                A.ImageCompression(quality_lower=50, quality_upper=100, p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.input_size, self.input_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)
        image_tensor = transformed["image"]

        return image_tensor, label
