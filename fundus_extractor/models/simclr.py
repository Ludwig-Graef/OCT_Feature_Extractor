import os
from typing import Tuple
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.nn import functional as F
import torchvision.models as models

from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from fundus_extractor.utils import general


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1), stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size), stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(nn.ReflectionPad2d(radias), self.blur_h, self.blur_v)

        self.pil_to_tensor = T.ToTensor()
        self.tensor_to_pil = T.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class ContrastiveLearningViewGenerator(nn.Module):
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform, n_views=2, **kwargs):
        super().__init__(**kwargs)
        self.base_transform = base_transform
        self.n_views = n_views

    def forward(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]


class SimCLRAugmentations(nn.Module):
    def __init__(self, image_size: int, s: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.image_size = image_size
        color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.transform = T.Compose([
            T.RandomResizedCrop(size=image_size),
            T.RandomHorizontalFlip(),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * image_size)),
            T.ToTensor()
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model: str, out_dim: int) -> None:
        super(ResNetSimCLR, self).__init__()
        resnet_dict = {
            "resnet_18": models.resnet18(weights=None, num_classes=out_dim),
            "resnet_50": models.resnet50(weights=None, num_classes=out_dim)
        }

        self.backbone = resnet_dict[base_model]
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def forward(self, x):
        return self.backbone(x)


class SimCLR(object):
    def __init__(self, backbone: nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler, device, batch_size: int, n_views: int,
                 temperature: float, similarity_fn: str, summary_writer: SummaryWriter, **kwargs):
        self.device = device
        self.backbone = backbone
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_views = n_views
        self.temperature = temperature
        self.summary_writer = summary_writer

        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        if similarity_fn == 'dot':
            self.similarity = lambda x: torch.matmul(x, x.T)
        elif similarity_fn == 'cosine':
            self.similarity = nn.CosineSimilarity()
        else:
            raise NotImplementedError(f'Similarity function {similarity_fn} not implemented.')

    def info_nce_loss(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        features = F.normalize(features, dim=1)

        similarity_matrix = self.similarity(features)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        # get class indices (first value is the one value that is "positive")
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)

        logits = logits / self.temperature
        return logits, labels

    def train(self, train_loader: DataLoader, backbone_architecture: str, epochs: int, fp16_precision: bool,
              log_every_n_steps: int, **kwargs) -> None:

        scaler = GradScaler(enabled=fp16_precision)

        n_iter = 0
        logging.info(f"Start SimCLR training for {epochs} epochs.")

        labels_input = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels_input = (labels_input.unsqueeze(0) == labels_input.unsqueeze(1)).float()
        labels_input = labels_input.to(self.device)

        with tqdm(total=epochs, position=0, unit="epoch") as tepoch:
            for epoch_counter in range(epochs):
                for images, labels in tqdm(train_loader, position=0, unit="iterations", leave=False, miniters=10):
                    images = torch.cat(images, dim=0)

                    images = images.to(self.device)

                    with autocast(enabled=fp16_precision):
                        features = self.backbone(images)
                        logits, labels = self.info_nce_loss(features, labels_input)
                        # logits[0, :] should be 1 (similar), logits[1:, :] should be 0 (not similar)
                        loss = self.criterion(logits, labels)

                    self.optimizer.zero_grad()

                    scaler.scale(loss).backward()

                    scaler.step(self.optimizer)
                    scaler.update()

                    if n_iter % log_every_n_steps == 0:
                        top1, top5 = accuracy(logits, labels, topk=(1, 5))
                        self.summary_writer.add_scalar('loss', loss, global_step=n_iter)
                        self.summary_writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                        self.summary_writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                        self.summary_writer.add_scalar('learning_rate',
                                                       self.scheduler.get_last_lr()[0], global_step=n_iter)

                    n_iter += 1

                # warmup for the first 10 epochs
                if epoch_counter >= 10:
                    self.scheduler.step()

                logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
                tepoch.set_postfix({'Loss': loss.item(), 'Top1 accuracy': top1[0].item()})
                tepoch.update(1)

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epochs)
        general.save_checkpoint(
            {
                'epoch': epochs,
                'backbone_architecture': backbone_architecture,
                'state_dict': self.backbone.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(self.summary_writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.summary_writer.log_dir}.")



def fit(train_dataloader: DataLoader, n_views: int, batch_size: int, img_size: int, out_dim: int,
        backbone_architecture: str, lr: float, weight_decay: float, **kwargs) -> None:
    assert n_views == 2, "Only two view training is supported. Please use --n-views 2."
    augmentations = SimCLRAugmentations(img_size)
    train_dataloader.dataset.transform = ContrastiveLearningViewGenerator(augmentations, n_views)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    backbone = ResNetSimCLR(base_model=backbone_architecture, out_dim=out_dim).to(device)

    optimizer = torch.optim.Adam(backbone.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader), eta_min=0, last_epoch=-1)

    simclr = SimCLR(backbone, optimizer, scheduler, device, n_views=n_views, batch_size=batch_size, **kwargs)
    simclr.train(train_dataloader, backbone_architecture, **kwargs)
