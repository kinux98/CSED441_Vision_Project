import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import torch

from os import listdir
from os.path import join
from PIL import Image

from data import common

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    #img = np.array(Image.open(filepath).convert('YCbCr'))
    img = np.array(Image.open(filepath).convert('RGB'))
    return img[:,:,:].astype(float)

class PIXDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def name(self):
        return "PIX"


    def __init__(self, opt):
        super(PIXDataset, self).__init__()
        self.opt = opt
        self.train = (opt['phase'] == 'train')
        self.split = 'train' if self.train else 'test'

        # read image list from image/binary files
        x_dir = self.opt['dataroot_LR']
        y_dir = self.opt['dataroot_HR']

        self.x_filenames = [join(x_dir, x) for x in listdir(x_dir) if is_image_file(x)]
        self.y_filenames = [join(y_dir, y) for y in listdir(y_dir) if is_image_file(y)]
        x_list = listdir(x_dir)
        y_list = listdir(y_dir)
        for i in range(len(y_list)):
            if y_list[i // 4] not in x_list[i]:
                print("dataset error : ", x_list[i], y_list[i // 4])

    def __getitem__(self, index):
        x_path = self.x_filenames[index]
        y_path = self.y_filenames[index // 4]
        x = load_img(x_path)
        y = load_img(y_path)

        x = Variable(torch.from_numpy(x).float()).permute(2, 0, 1)
        y = Variable(torch.from_numpy(y).float()).permute(2, 0, 1)

        return {'LR': x, 'HR': y, 'LR_path': x_path, 'HR_path': y_path}

    def __len__(self):
        return len(self.x_filenames)