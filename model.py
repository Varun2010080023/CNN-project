"""Small, randomly initialized CNN and shared image preprocessing."""
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms

CLASSES = ("apple", "banana")
IMAGE_SIZE = 96
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "fruit_cnn.pt"


class FruitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = 3
        for width in (16, 32, 64, 96):
            layers.extend([
                nn.Conv2d(channels, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2),
            ])
            channels = width
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(96 * 3 * 3, 64), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(64, len(CLASSES)),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def prepare_image(image):
    """Orient correctly, composite transparency, and preserve aspect ratio."""
    image = ImageOps.exif_transpose(image).convert("RGBA")
    background = Image.new("RGBA", image.size, "white")
    image = Image.alpha_composite(background, image).convert("RGB")
    return ImageOps.pad(
        image, (IMAGE_SIZE, IMAGE_SIZE),
        method=Image.Resampling.BILINEAR, color="white",
    )


def image_transform(training=False):
    steps = []
    if training:
        steps.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(25, translate=(0.08, 0.08), scale=(0.85, 1.1), fill=255),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        ])
    return transforms.Compose(steps + [
        transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])


def load_model(path=MODEL_PATH):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != list(CLASSES) or checkpoint["image_size"] != IMAGE_SIZE:
        raise ValueError("Checkpoint is incompatible with this app. Retrain with train.py.")
    model = FruitCNN()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def predict(model, image):
    tensor = image_transform()(prepare_image(image)).unsqueeze(0)
    probabilities = model(tensor).softmax(dim=1)[0].tolist()
    index = max(range(len(probabilities)), key=probabilities.__getitem__)
    return CLASSES[index], dict(zip(CLASSES, probabilities))
