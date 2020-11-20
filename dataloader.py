import os
import random
from PIL import Image

from torch.utils import data

import torchvision
import torchvision.transforms as transforms


def create_dataloader(dataset='cifar10', batch_size=64, num_workers=1):
    if dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=transform, download=True)
        testset = torchvision.datasets.CIFAR10(root='./data/', train=False, transform=transform, download=True)

        trainloader = data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        testloader = data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elif dataset == 'summer2winter':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        trainset = Summer2WinterDataset(train=True, transform=transform)
        testset = Summer2WinterDataset(train=False, transform=transform)

        trainloader = data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        testloader = data.DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


class Summer2WinterDataset(data.Dataset):
    def __init__(self, train: bool=True, transform=None):
        self.transform = transform
        dataset_dir = './data/summer2winter_yosemite/'

        # Implement the dataset for unpaired image-to-image translation.
        # Check the dataset directory and implement the proper dataset.
        # This dataset have to load the train or test files depending on the 'train' option.

        ### YOUR CODE HERE (~ 10 lines)
        if train:
            self.dir_A = os.path.join(dataset_dir,'train'+'A')
            self.dir_B = os.path.join(dataset_dir,'train'+'B')
        else:
            self.dir_A = os.path.join(dataset_dir,'test'+'A')
            self.dir_B = os.path.join(dataset_dir,'test'+'B')

        self.A_paths = sorted(FolderDataset(self.dir_A).image_list)
        self.B_paths = sorted(FolderDataset(self.dir_B).image_list)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        ### END YOUR CODE

    def __getitem__(self, index):

        # The number of images in domain A and domain B are different.
        # You have to sample the index to load data from different pairs.

        ### YOUR CODE HERE (~ 2 lines)
        image_A = Image.open(os.path.join(self.dir_A,self.A_paths[index % self.A_size])).convert('RGB')
        image_B = Image.open(os.path.join(self.dir_B,self.B_paths[index % self.B_size])).convert('RGB')        
        # image_A = Image.open(self.A_paths[index % self.A_size]).convert('RGB')
        # image_B = Image.open(self.B_paths[index % self.B_size]).convert('RGB')

        ### END YOUR CODE

        return self.transform(image_A), self.transform(image_B)

    def __len__(self):
        #return len(self.image_list_A)
        return len(self.A_paths)


class FolderDataset(data.Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.image_list = os.listdir(folder)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.folder, self.image_list[index]))
        return self.transform(image)

    def __len__(self):
        return len(self.image_list)
