import torch
import torch.nn as nn

class DeepfakeClassifier(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        video_feature = features.mean(dim=0)
        return self.classifier(video_feature)
