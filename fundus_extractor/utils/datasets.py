import os
from typing import Any, Callable, Optional, Tuple

from torchvision import datasets
import torchvision.transforms as T
from torchvision.datasets import DatasetFolder, ImageFolder


class STL10_Dataset(datasets.STL10):
    def __init__(self, save_dir: str, transform: Optional[Callable] = None) -> None:
        super().__init__(save_dir, transform=transform, download=True)


class OCT_2D_Dataset(DatasetFolder):
    def __init__(self, root: str, loader: Callable, extensions: str, transform: Optional[Callable] = None) -> None:
        super().__init__(root, loader, extensions, transform)


class Fundus_Left_Right_Combined_Dataset(ImageFolder):
    def __init__(self, root: str, loader: Callable[[str], Any], transform: Optional[Callable] = None) -> None:
        super().__init__(root, transform, loader=loader)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, os.path.basename(path)