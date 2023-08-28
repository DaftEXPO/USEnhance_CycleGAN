import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
from .data_utils import *


def get_transforms(opt, input_nc):
    return get_transform(opt, grayscale=(input_nc == 1))


class ImageDataset(Dataset):
    def __init__(self, root, noise_level, count = None, transforms_1=None, transforms_2=None, unaligned=False, B2A=False):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        if B2A is not True:
            self.files_A = sorted(glob.glob("%s/A/*.png" % root))
            self.files_B = sorted(glob.glob("%s/B/*.png" % root))
        else:
            self.files_A = sorted(glob.glob("%s/B/*.png" % root))
            self.files_B = sorted(glob.glob("%s/A/*.png" % root))
        self.unaligned = unaligned
        self.noise_level =noise_level
        print(self.files_A[0])
        
    def __getitem__(self, index):
        if self.noise_level == 0:
            # if noise =0, A and B make same transform
            seed = np.random.randint(2147483647) # make a seed with numpy generator 
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            path_A = self.files_A[index % len(self.files_A)]
            item_A = Image.open(path_A).convert('L')
            item_A = self.transform2(item_A)
            #item_A = self.transform2(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            path_B = self.files_B[index % len(self.files_B)]
            item_B = Image.open(path_B).convert('L')
            item_B = self.transform2(item_B)
            #item_B = self.transform2(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))
        else:
            # if noise !=0, A and B make different transform
            path_A = self.files_A[index % len(self.files_A)]
            item_A = Image.open(path_A).convert('L')
            item_A = self.transform1(item_A)
            #item_A = self.transform1(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))
            path_B = self.files_B[index % len(self.files_B)]
            item_B = Image.open(path_B).convert('L')
            item_B = self.transform1(item_B)
            #item_B = self.transform1(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))
            
        if os.path.basename(path_A)!=os.path.basename(path_B):
            print(path_A, path_B)
            
        return {'A': item_A, 'B': item_B}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ValDataset(Dataset):
    def __init__(self, root,count = None,transforms_=None, unaligned=False, B2A=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        if B2A is not True:
            self.files_A = sorted(glob.glob("%s/A/*.png" % root))
            self.files_B = sorted(glob.glob("%s/B/*.png" % root))
        else:
            self.files_A = sorted(glob.glob("%s/B/*.png" % root))
            self.files_B = sorted(glob.glob("%s/A/*.png" % root))
        print(self.files_A[0])
    def __getitem__(self, index):

        path_A = self.files_A[index % len(self.files_A)]
        item_A = Image.open(path_A).convert('L')
        item_A = self.transform(item_A)
        case_name = os.path.basename(path_A)
        if self.unaligned:
            path_B = self.files_B[random.randint(0, len(self.files_B) - 1)]
            item_B = Image.open(path_B).convert('L')
            item_B = self.transform(item_B)
            #item_B = self.transform(np.load(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            path_B = self.files_B[index % len(self.files_B)]
            item_B = Image.open(path_B).convert('L')
            item_B = self.transform(item_B)
            #item_B = self.transform(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))
        return {'A': item_A, 'B': item_B, 'case':case_name}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


    
class TestDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob("%s/*.png" % root))
    def __getitem__(self, index):

        path = self.files[index % len(self.files)]
        item= Image.open(path).convert('L')
        item = self.transform(item)
        case_name = os.path.basename(path)
        return {'A': item, 'case':case_name}
    def __len__(self):
        return len(self.files)
