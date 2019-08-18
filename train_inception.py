import os
import sys
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

# from data_loader import *
from data_loader_resnet import *
import model_resnets
from models import *
from utils_inception import train, evaluate
import time
import pickle
from plots import plot_learning_curves

if __name__ == "__main__":
    torch.manual_seed(0)
    if torch.cuda.is_available():
        print("using cuda")
        torch.cuda.manual_seed(0)
    else:
        print("no cuda")

    start_time = time.time()

    PATH_TO_DATA = "./CheXpert-v1.0-small"   # pass as argument
    if len(sys.argv) > 1:
        PATH_TO_DATA = str(sys.argv[1])
        # print(sys.argv[1])
    BATCH_SIZE = 4*32
    NUM_WORKERS = 16
    NUM_EPOCHS = 2
    USE_CUDA = True
    train_trans = transforms.Compose([
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(), #AP/PA??
        transforms.ToTensor(),
        transforms.Normalize([0.5421649, 0.5421649, 0.5421649], [0.27066961, 0.27066961, 0.27066961])
    ])

    valid_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.5421649, 0.5421649, 0.5421649], [0.27066961, 0.27066961, 0.27066961])
    ])
    train_data, train_labels = load_data(PATH_TO_DATA, 'train.csv')
    valid_data, valid_labels = load_data(PATH_TO_DATA, 'valid.csv')

    train_dataset = chestx(train_data, train_labels, PATH_TO_DATA, train_trans)
    valid_dataset = chestx(valid_data, valid_labels, PATH_TO_DATA, valid_trans)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True) #image_collate
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=10, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
    valid_loader_batch_1 = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
    # model = Network()
    # model = model_resnets.feature_extractor()
    model = model_resnets.inception_model()
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
    model.to(device)
    criterion.to(device)

    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []
    batch_times_train = []
    data_times = []
    batch_times_valid = []
    for epoch in range(NUM_EPOCHS):
        train_loss, train_accuracy, batch_times_train, data_times = train(model, device, valid_loader, criterion, optimizer, epoch, 10, batch_times_train, data_times)
        valid_loss, valid_accuracy_avg, valid_results, batch_times_valid = evaluate(model, device, valid_loader_batch_1, criterion, 10, batch_times_valid)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy_avg)

        torch.save(model, "{}_{}.pth".format(start_time, epoch))

    print(batch_times_valid)
    print(batch_times_train)
    print(data_times)

    # Plotting Batch Times for training
    plt.plot(batch_times_valid, label='Validation Batch Time')
    plt.plot(data_times, label='Train Data Load Time')
    plt.plot(batch_times_train, label = 'Training Batch Time')
    plt.legend(loc="best")
    plt.xlabel("Epochs")
    plt.ylabel("Time")
    plt.show()

    print(train_accuracies)
    print(valid_accuracies)

    plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

    with open('metrics_{}.pkl'.format(start_time), 'wb') as f:
        pickle.dump([train_losses, valid_losses, train_accuracies, valid_accuracies], f)
