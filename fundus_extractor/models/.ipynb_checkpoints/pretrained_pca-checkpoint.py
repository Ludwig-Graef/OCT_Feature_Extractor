import numpy as np
from typing import List
import joblib

import torch
from torch import nn
from torchvision.models import resnet18, resnet50
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA as Sklearn_PCA

from fundus_extractor.utils.datasets import Fundus_Left_Right_Combined_Dataset


class ImageEncoder(nn.Module):
    def __init__(self, backbone_model: str, **kwargs):
        super().__init__(**kwargs)
        if backbone_model == 'resnet18':
            model = resnet18(pretrained=True)
        elif backbone_model == 'resnet50':
            model = resnet50(pretrained=True)
        else:
            raise NotImplementedError(f'Backbone model {backbone_model} not implemented!')
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def extract_features(model: nn.Module, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    features = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            output = model(inputs)
            features.append(output.cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features


def fit(train_dataloader: DataLoader, n_components: int, backbone_model: str, model_save_path: str,
        device: torch.device) -> None:
    model = ImageEncoder(backbone_model).to(device)

    features = extract_features(model, train_dataloader, device)

    pca = Sklearn_PCA(n_components=n_components)
    pca.fit(features)
    joblib.dump(pca, model_save_path)
