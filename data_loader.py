import torch
import numpy as np
import pandas as pd
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import os

class chestx(Dataset):

    def __init__(self, paths, labels, root_dir='./', transform=None):
        self.labels = labels
        self.paths = paths
        self.root_dir = root_dir   # Absolute/Relative path to directory from where data paths are being calculated from

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.root_dir, self.paths[idx])).astype(np.float32)/255.0
        if(len(img.shape) == 3): # B/W vs color, verify
            image = cv2.cvtColor(img[:,:,::-1], cv2.COLOR_RGB2GRAY)
        else:
            image = img

        # Return Specific size
        sh = image.shape
        if sh[0] >= 300 and sh[0] <= 400 and sh[1] >= 300 and sh[1] <= 400:
            x = int((sh[0]-300)/2)
            y = int((sh[1]-300)/2)
            image = image[x:x+300, y:y+300]
        else:
            image = cv2.resize(image, (300, 300))

        image = np.reshape(image, (1,300,300))
        # can apply transfromations to images
        return image, self.labels[idx]


def image_collate(batch):   # unsure if needed
    images, labels = zip(*batch)
    return torch.FloatTensor(images), torch.FloatTensor(labels) # Need to use FloatTensor for uncertainty label

def load_data(path, filename):  # Do this cleaning via Scala??
    raw_data = pd.read_csv(os.path.join(path, filename))
    frontal_data = raw_data[raw_data['Frontal/Lateral']=='Frontal']
    # do additional filtering

    def remove_main_folder(path):
        path_folders = path.split('/')[1:]
        return '/'.join(path_folders)

    removed_unneeded = frontal_data.drop(columns=['Sex', 'Age', 'Frontal/Lateral', 'AP/PA']) # as per need
    removed_unneeded['Path'] = removed_unneeded['Path'].apply(remove_main_folder)

    final_df = removed_unneeded.set_index('Path')
    final_df.fillna(0, inplace=True)
    final_df.replace(-1, 0.5, inplace=True) # -1 is uncertain label. Maybe adjust value from 0.5, tunable parameter

    return final_df.index.tolist(), final_df.values.tolist()


# Hyperparams
n_epoch = 1
batch_size = 32
num_workers = 4
path = '/Users/abhishekmangal/Downloads/Data_vvsmall/'

dst = chestx(csv_file='data.csv',root_dir='data/faces/')
dataset = DataLoader(dst, batch_size=batch_size, shuffle=True, num_workers=num_workers)


for i in range(n_epoch):
    for j, (x,y) in enumerate(dataset):
        print("Epoch = ", i, " Data number = ", j)
        print(x.shape, y.shape)
