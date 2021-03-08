from keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_data(dataset):
    (X_train, y_train), (X_test, y_test) = eval(dataset).load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train_mod = []
    X_test_mod = []

    for x in X_train:
        if dataset == 'mnist' or dataset == 'fashion_mnist':
            x = x.reshape(28, 28, 1)
        X_train_mod.append(x)

    for x in X_test:
        if dataset == 'mnist' or dataset == 'fashion_mnist':
            x = x.reshape(28, 28, 1)
        X_test_mod.append(x)

    print(dataset)
    print(len(X_train_mod))
    print(X_train_mod[0].shape)
    print(len(X_test_mod))
    print(X_test_mod[0].shape)

    np.save(f"datasets/{dataset}/baseline/X_train.npy", X_train_mod)
    np.save(f"datasets/{dataset}/baseline/y_train.npy", y_train)
    np.save(f"datasets/{dataset}/baseline/X_test.npy", X_test_mod)
    np.save(f"datasets/{dataset}/baseline/y_test.npy", y_test)


def create_gtsrb_data():
    # This csv files are Daniel's
    X_train = pd.read_csv("datasets/gtsrb/x_train_gr_smpl.csv")
    X_test = pd.read_csv("datasets/gtsrb/x_test_gr_smpl.csv")
    y_train = pd.read_csv("datasets/gtsrb/y_train_smpl.csv")
    y_test = pd.read_csv("datasets/gtsrb/y_test_smpl.csv")

    X_train_mod = []
    X_test_mod = []
    y_train_mod = []
    y_test_mod = []

    for i in range(X_train.shape[0]):
        img = np.uint8(X_train.iloc[i])
        img = img / 255.0
        X_train_mod.append(img.reshape((48, 48, 1)))

    for i in range(X_test.shape[0]):
        img = np.uint8(X_test.iloc[i])
        img = img / 255.0
        X_test_mod.append(img.reshape((48, 48, 1)))

    for i in range(y_train.shape[0]):
        label = np.uint8(y_train.iloc[i])
        y_train_mod.append(label[0])

    for i in range(y_test.shape[0]):
        label = np.uint8(y_test.iloc[i])
        y_test_mod.append(label[0])

    np.save(f"datasets/gtsrb/baseline/X_train.npy", X_train_mod)
    np.save(f"datasets/gtsrb/baseline/y_train.npy", y_train_mod)
    np.save(f"datasets/gtsrb/baseline/X_test.npy", X_test_mod)
    np.save(f"datasets/gtsrb/baseline/y_test.npy", y_test_mod)


if __name__ == "__main__":
    create_data("mnist")
    create_data("fashion_mnist")
    create_data("cifar10")
    create_gtsrb_data()
