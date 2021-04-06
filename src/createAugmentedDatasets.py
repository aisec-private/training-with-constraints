import numpy as np
import argparse
import random
import time


def mod_data(distance, X, Y, n):
    X_mod = []
    Y_mod = []
    t = time.time()

    for i in range(len(X)):
        x_mod = []
        y_mod = []

        # Create 'n' new images per image 'x'
        while len(x_mod) < n:
            x = X[i].copy()
            # Randomly modify numbers until a certain euclidean distance from the original image is reached
            while True:
                # Random positive modification number
                d = random.uniform(-0.1, 0.1)
                # Random pixel
                px = random.randint(0, x.shape[0] - 1)
                py = random.randint(0, x.shape[1] - 1)
                pz = random.randint(0, x.shape[2] - 1)
                # Modify the pixel if it stays between [0, 1]
                if x[px][py][pz] + d >= 0 and x[px][py][pz] + d <= 1:
                    x[px][py][pz] += d
                # Calculating the euclidean distance
                euclidean_distance = 0
                for j in range(x.shape[0]):
                    for k in range(x.shape[1]):
                        for z in range(x.shape[2]):
                            euclidean_distance += (X[i][j][k][z] - x[j][k][z]) ** 2
                euclidean_distance = np.sqrt(euclidean_distance)
                # Break when we reach the desired distance
                if euclidean_distance > distance:
                    x_mod.append(x)
                    y_mod.append(Y[i])
                    break

        for x in x_mod:
            X_mod.append(x)
        for y in y_mod:
            Y_mod.append(y)

        # Calculating the percentage of completed work
        completion_percentage = (i + 1) / len(X) * 100

        # Calculating the elapsed time
        elapsed_time = time.time() - t
        e_hours, e_rem = divmod(elapsed_time, 3600)
        e_minutes, e_seconds = divmod(e_rem, 60)

        # Calculating an estimation of the total time required
        total_time = elapsed_time * 100 / completion_percentage
        t_hours, t_rem = divmod(total_time, 3600)
        t_minutes, t_seconds = divmod(t_rem, 60)

        print(f'\rCompletion: {completion_percentage:.2f}% - Elapsed time: {int(e_hours)}:{int(e_minutes)}:{e_seconds:.2f} - Expected time: {int(t_hours)}:{int(t_minutes)}:{t_seconds:.2f}', end='', flush=True)

    print()
    return X_mod, Y_mod


def create_dataset(dataset, distance, n, xname, yname):
    print(f"\nLoading {dataset} dataset.")

    X_train = np.load(f"datasets/{dataset}/baseline/X_train.npy")
    y_train = np.load(f"datasets/{dataset}/baseline/y_train.npy")

    print(f"Baseline dataset length: {len(X_train)}.")
    print(f"Creating new {dataset} dataset.")

    X_train_mod, y_train_mod = mod_data(distance, X_train, y_train, n)

    print(f"Augmented dataset length: {len(X_train_mod)}")
    print(f"Saving new {dataset} dataset.")

    np.save(f"datasets/{dataset}/augmented/{xname}.npy", X_train_mod)
    np.save(f"datasets/{dataset}/augmented/{yname}.npy", y_train_mod)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--distance', type=float, default=0.1)
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--xname', type=str, default='X_train')
    parser.add_argument('--yname', type=str, default='y_train')
    args = parser.parse_args()

    print(f'Dataset: {args.dataset} - Distance: {args.distance} - N: {args.n} - Xname: {args.xname} - Yname: {args.yname}')
    create_dataset(args.dataset, args.distance, args.n, args.xname, args.yname)
