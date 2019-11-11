# import torch
import mxnet
import numpy as np
import pandas as pd
# from torch.utils import data
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
from mxnet.gluon import data
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
import os
from PIL import Image

class chestx(Dataset):

    def __init__(self, paths, labels, root_dir='./', transform=None):
        self.labels = labels
        self.paths = paths
        self.transform = transform
        self.root_dir = root_dir   # Absolute/Relative path to directory from where data paths are being calculated from

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, self.paths[idx])).convert('RGB')
        return self.transform(img), self.labels[idx]


# def image_collate(batch):   # unsure if needed
#     images, labels = zip(*batch)
#     return torch.FloatTensor(images), torch.FloatTensor(labels) # Need to use FloatTensor for uncertainty label

def load_data(path, filename):  # Do this cleaning via Scala??
    raw_data = pd.read_csv(os.path.join(path, filename))
    frontal_data = raw_data[raw_data['Frontal/Lateral']=='Frontal']

    def remove_main_folder(path):
        path_folders = path.split('/')[1:]
        return '/'.join(path_folders)

    removed_unneeded = frontal_data.drop(columns=['Sex', 'Age', 'Frontal/Lateral', 'AP/PA']) # as per need
    removed_unneeded['Path'] = removed_unneeded['Path'].apply(remove_main_folder)

    final_df = removed_unneeded.set_index('Path')
    final_df.fillna(0, inplace=True)
    final_df.replace(-1, 0.5, inplace=True) # -1 is uncertain label. Maybe adjust value from 0.5, tunable parameter

    # print(final_df.values.shape)
    # for blah in final_df.values:
    #     print(blah.shape)

    return final_df.index.tolist(), final_df.values.astype("double")


# print("Hello")
train_trans = transforms.Compose([
    transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize([0.5421649, 0.5421649, 0.5421649], [0.229, 0.224, 0.225])
])

# PATH_TO_DATA = "./CheXpert-v1.0-small"
# train_data, train_labels = load_data(PATH_TO_DATA, 'train.csv')
# train_dataset = chestx(train_data, train_labels, PATH_TO_DATA, train_trans)
# train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=16)

# z = []
# n = 0
# z1 = []
# each_std = 0
# for i, (img, target) in enumerate(train_loader):
#     print(i)
#     mean = np.mean(img.numpy())
#     z.append(mean)

#     mean2 = np.mean((img.numpy())**2)
#     z1.append(mean2)

# print(np.mean(z))
# print((np.mean(z1) - np.mean(z)**2)**0.5)