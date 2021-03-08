from PIL import Image
import numpy as np
import torch
import cv2
import time


class MyDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, dtype, train, transform):
        self.dataset = dataset
        self.dtype = dtype
        self.train = train
        self.transform = transform
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        
        if self.train:
            data = np.load(f"datasets/{self.dataset}/baseline/X_train.npy")
            [self.X_train.append(d) for d in data]

            data = np.load(f"datasets/{self.dataset}/baseline/y_train.npy")
            [self.y_train.append(torch.tensor(int(d))) for d in data]

            if self.dtype == 'augmented':
                data = np.load(f"datasets/{self.dataset}/augmented/X_train.npy")
                [self.X_train.append(d) for d in data]

                data = np.load(f"datasets/{self.dataset}/augmented/y_train.npy")
                [self.y_train.append(torch.tensor(int(d))) for d in data]

            if self.dtype == 'augmented_FGSM':
                data = np.load(f"datasets/{self.dataset}/augmented_FGSM/X_train_FGSM.npy")
                [self.X_train.append(d.astype('float64')) for d in data]

                data = np.load(f"datasets/{self.dataset}/augmented_FGSM/y_train_FGSM.npy")
                [self.y_train.append(torch.tensor(int(d))) for d in data]

            if self.dtype == 'augmented_PGD':
                data = np.load(f"datasets/{self.dataset}/augmented_PGD/X_train_PGD.npy")
                [self.X_train.append(d.astype('float64')) for d in data]

                data = np.load(f"datasets/{self.dataset}/augmented_PGD/y_train_PGD.npy")
                [self.y_train.append(torch.tensor(int(d))) for d in data]

        else:
            data = np.load(f"datasets/{self.dataset}/baseline/X_test.npy")
            [self.X_test.append(d) for d in data]

            data = np.load(f"datasets/{self.dataset}/baseline/y_test.npy")
            [self.y_test.append(torch.tensor(int(d))) for d in data]

    def __getitem__(self, index):
        if self.train:
            img = self.X_train[index]
            label = self.y_train[index]
        else:
            img = self.X_test[index]
            label = self.y_test[index]

        if self.dataset == 'cifar10' or self.dataset == 'coil20unproc' or self.dataset == 'coil20proc':
            img = Image.fromarray(img, 'RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if self.train:
            return len(self.X_train)
        else:
            return len(self.X_test)
