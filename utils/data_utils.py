import logging

import os
import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import pandas as pd
import PIL.Image as Image

import sys
sys.path.append('../')
from AttentionCrop.CroppingModelLoader import CroppingModelLoader

logger = logging.getLogger(__name__)

class Dataset_SplitByCSV(torch.utils.data.Dataset):
    def __init__(self, root_dataset, path_csv, transform=None):
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

    def __getitem__(self, index):
        data_path = os.path.join(self.root_dataset, self.list_datapair[index][1], self.list_datapair[index][0])
        img = Image.open(data_path)
        img.convert('RGB')
        if self.transform != None:
            img = self.transform(img)
        label = torch.tensor(self.className2idx[self.list_datapair[index][1]])
        data = (img, label)
        return data

    def __len__(self):
        return self.len

def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.use_cropping_model:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
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

    if args.use_cropping_model:
        train_loader = CroppingModelLoader(trainset, 
                                          args.cropping_model_checkpoint,
                                          args.device,
                                          args.train_batch_size,
                                          patch_len=args.img_size,
                                          positive_sample_threshold=args.cropping_model_positive_sample_threshold,
                                          list_downsample_rate=args.cropping_model_list_downsample_rate,
                                          hidden_activation=args.cropping_model_hidden_activation)
                                          
        test_loader = CroppingModelLoader(testset, 
                                          args.cropping_model_checkpoint,
                                          args.device,
                                          args.eval_batch_size,
                                          patch_len=args.img_size,
                                          positive_sample_threshold=args.cropping_model_positive_sample_threshold,
                                          list_downsample_rate=args.cropping_model_list_downsample_rate,
                                          hidden_activation=args.cropping_model_hidden_activation) if testset is not None else None
    else:
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