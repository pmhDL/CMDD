""" Dataloader for all datasets. """
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class DatasetLoader(Dataset):
    """The class to load the dataset"""
    def __init__(self, setname, args):
        THE_PATH = osp.join(args.dataset_dir, 'feat-' + setname + '.npz')
        data0 = np.load(THE_PATH)
        data = data0['features']
        label = data0['targets']

        pathemb = osp.join(args.dataset_dir, 'few-shot-wordemb-' + setname + '.npz')
        wordemb = np.load(pathemb)['features']

        self.wordemb = wordemb
        self.data = data
        self.label = label
        self.num_class = len(np.unique(label))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image = self.data[i]
        label = self.label[i]
        return image, label, self.wordemb[label]