import os
import json
import shutil

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


def write_job_meta(job_dir: str, **args: dict) -> None:
    os.makedirs(job_dir, exist_ok=True)
    with open(os.path.join(job_dir, 'config.json'), 'w+') as fp:
        json.dump(args, fp)


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')