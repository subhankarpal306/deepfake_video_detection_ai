import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        base_model = models.resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        self.feature_extractor = nn.Sequential(
            *list(base_model.children())[:-1]
        )
        self.feature_dim = 512

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def forward(self, frames):
        features = []

        for frame in frames:
            x = self.transform(frame).unsqueeze(0)
            with torch.no_grad():
                feat = self.feature_extractor(x)
            features.append(feat.squeeze())

        return torch.stack(features)
