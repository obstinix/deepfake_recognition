import torch
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_tta_transforms(img_size: int = 224) -> list[T.Compose]:
    """5 deterministic TTA augmentations for inference."""
    base = [T.Resize(int(img_size * 1.14)), T.CenterCrop(img_size), T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    flipped = [T.Resize(int(img_size * 1.14)), T.CenterCrop(img_size),
               T.RandomHorizontalFlip(p=1.0), T.ToTensor(),
               T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return [T.Compose(base), T.Compose(flipped)]
