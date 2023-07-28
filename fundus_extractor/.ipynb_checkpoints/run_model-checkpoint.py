import os
import sys
import argparse
import logging
import tqdm
from distutils.util import strtobool

from torch.utils.tensorboard import SummaryWriter
from fundus_extractor.utils.datasets import Fundus_Left_Right_Combined_Dataset
from torch.utils.data import Dataset, DataLoader, Subset

from fundus_extractor.models import simclr, pretrained_pca
from fundus_extractor.utils import general


def get_dataloader(dir_to_dataset: str, batch_size: int, num_workers: int, **kwargs) -> Dataset:
    dataset = Fundus_Left_Right_Combined_Dataset(dir_to_dataset)
    # dataset = Subset(dataset, np.random.choice(range(len(dataset)), 1024))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                      drop_last=True)


def run_model(verbose: bool, seed: int, training_save_dir: str, model_type: str, **kwargs) -> None:
    os.makedirs(training_save_dir, exist_ok=True)
    general.set_seeds(seed)

    if training_save_dir.endswith('runs'):
        run_number = len([r for r in os.listdir(training_save_dir) if r.startswith('run_')])
        training_save_dir = os.path.join(training_save_dir, f'run_{run_number}')
    os.makedirs(training_save_dir, exist_ok=True)
    general.write_job_meta(training_save_dir, **kwargs)

    summary_writer = SummaryWriter(training_save_dir)
    summary_writer.add_hparams(kwargs, {})
    logging.basicConfig(filename=os.path.join(training_save_dir, 'training.log'), level=logging.DEBUG)
    if verbose is False:
        sys.stdout = open(os.path.join(training_save_dir, 'out.txt'), 'w+')
        sys.stderr = open(os.path.join(training_save_dir, 'err.txt'), 'w+')

    train_dataloader = get_dataloader(**kwargs)

    if model_type == 'sim_clr':
        simclr.fit(train_dataloader=train_dataloader, summary_writer=summary_writer, **kwargs)
    elif model_type == 'pretrained_pca':
        pretrained_pca.fit(train_dataloader=train_dataloader, **kwargs)
    else:
        raise NotImplementedError(f'Model type {model_type} not implemented.')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    parser.add_argument('--model_type', type=str, help='Which model type to use.')
    parser.add_argument('--dir_to_dataset', type=str, help='dir to dataset')
    parser.add_argument('--training_save_dir', type=str, help='Path of where to store the trained model and logs.')
    parser.add_argument('--seed', default=7, type=int, help='seed for initializing training. ')
    parser.add_argument('--backbone_architecture', type=str, default='resnet18')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--img_size', default=224, type=int, help='weight decay')
    parser.add_argument('--fp16_precision', default=False, type=lambda b: bool(strtobool(b)), help='16-bit precision.')
    parser.add_argument('--num_workers', default=-1, type=int, help='Number of workers')
    parser.add_argument('--out_dim', default=128, type=int, help='feature dimension (default: 128)')
    parser.add_argument('--log_every_n_steps', default=100, type=int, help='Log every n steps')
    parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
    parser.add_argument('--n_views', default=2, type=int, help='Number of views for contrastive learning training.')
    parser.add_argument('--similarity_fn', default='dot', type=str, help='Which SimCLR distance metric to use')
    parser.add_argument('--verbose', default=False, type=lambda b: bool(strtobool(b)))
    return parser.parse_args()


if __name__ == "__main__":
    kwargs = vars(parse_args())
    run_model(**kwargs)