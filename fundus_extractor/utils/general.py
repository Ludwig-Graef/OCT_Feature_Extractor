import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T


def imshow(img: torch.Tensor):
    """
   shows an imagenet-normalized image on the screen
   """
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.show()


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    image = image / 255
    return T.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def unnormalize_image(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    return T.functional.normalize(image, (-mean / std).tolist(), (1.0 / std).tolist())
