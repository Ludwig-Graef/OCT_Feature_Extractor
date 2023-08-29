import numpy as np
from typing import List
import joblib
from tqdm import tqdm

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
            model = resnet18(weights='ResNet18_Weights.DEFAULT')
        elif backbone_model == 'resnet50':
            model = resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
        else:
            raise NotImplementedError(f'Backbone model {backbone_model} not implemented!')
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def extract_features(model: nn.Module, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for inputs, file_names in tqdm(dataloader, desc='Process inputs'):
            output = model(inputs.to(device))
            features.append(output.cpu().numpy())
            labels.extend([f.split('.')[0] for f in file_names])
    return np.concatenate(features, axis=0).squeeze(), labels


def fit(train_dataloader: DataLoader, n_components: int, backbone_model: str, model_save_path: str,
        device: torch.device) -> None:
    model = ImageEncoder(backbone_model).to(device)

    features = extract_features(model, train_dataloader, device)
    features = features.squeeze()

    pca = Sklearn_PCA(n_components=n_components)
    pca.fit(features)
    joblib.dump(pca, model_save_path)
    return pca.transform(features)


def load_model(model_save_path: str) -> Sklearn_PCA:
    return joblib.load(model_save_path)
