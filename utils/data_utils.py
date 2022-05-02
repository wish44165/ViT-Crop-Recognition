import logging

import os
import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import pandas as pd
import PIL.Image as Image

logger = logging.getLogger(__name__)

class Dataset_SplitByCSV(torch.utils.data.Dataset):
    def __init__(self, root_dataset, path_csv, transform=None, return_file_name=False):
        super().__init__()
        self.className2idx = {
            'banana': 0, 
            'bareland': 1,
            'carrot': 2,
            'corn': 3,
            'dragonfruit': 4,
            'garlic': 5,
            'guava': 6,
            'inundated': 7,
            'peanut': 8,
            'pineapple': 9,
            'pumpkin': 10,
            'rice': 11,
            'soybean': 12,
            'sugarcane': 13,
            'tomato': 14 
        }

        df = pd.read_csv(path_csv, header=None)
        df.info()
        self.list_datapair = df.values
        self.transform = transform
        self.root_dataset = root_dataset
        self.len = len(self.list_datapair)
        self.return_file_name = return_file_name

    def __getitem__(self, index):
        data_path = os.path.join(self.root_dataset, self.list_datapair[index][1], self.list_datapair[index][0])
        img = Image.open(data_path)
        img.convert('RGB')
        if self.transform != None:
            img = self.transform(img)
        label = torch.tensor(self.className2idx[self.list_datapair[index][1]])
        if self.return_file_name:
            file_name_prefix = self.list_datapair[index][0][:self.list_datapair[index][0].rfind('.')]
            data = (img, label, self.list_datapair[index][1], file_name_prefix)
        else:
            data = (img, label)
        return data

    def __len__(self):
        return self.len

def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "Crop_CSV":
        trainset = Dataset_SplitByCSV(args.dir_dataset, path_csv=args.path_csv_train, transform=transform_train)
        testset = Dataset_SplitByCSV(args.dir_dataset, path_csv=args.path_csv_val, transform=transform_test)
    else:
        data_dir = args.dir_dataset
        trainset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
        testset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_test)
        
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader

def get_attn_loader(cfg):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = Dataset_SplitByCSV(cfg['directory']['data']['root-dir'], 
                                 cfg['directory']['data']['path-csv'],
                                 transform=transform,
                                 return_file_name=True)
    dataloader = DataLoader(dataset, 1, False)
    return dataloader