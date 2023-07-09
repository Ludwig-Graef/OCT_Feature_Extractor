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
