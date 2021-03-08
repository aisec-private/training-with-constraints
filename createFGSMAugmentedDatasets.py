from dl2.models import MnistNet, GTSRBNet, ResNet18, FASHIONSmall, GTSRBSmall
from dl2.getDatasets import MyDataset
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import json


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def test(model, device, test_loader, epsilons):
    X_train = []
    y_train = []
    e_count = 0

    for epsilon in epsilons:
        print(e_count, '/', len(epsilons))
        e_count += 1

        # Loop over all examples in test set
        for data, target in test_loader:
            # Send the data and label to the device
            data, target = data.to(device), target.to(device)
            data = data.float()
            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True
            # Forward pass the data through the model
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            # If the initial prediction is wrong, dont bother attacking, just move on
            if init_pred.item() != target.item():
                continue
            # Calculate the loss
            loss = F.nll_loss(output, target)
            # Zero all existing gradients
            model.zero_grad()
            # Calculate gradients of model in backward pass
            loss.backward()
            # Collect datagrad
            data_grad = data.grad.data
            # Call FGSM Attack
            perturbed_data = fgsm_attack(data, epsilon, data_grad)
            # Re-classify the perturbed image
            output = model(perturbed_data)
            # Check for success
            final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            if final_pred.item() != target.item():
                X_train.append(np.reshape(perturbed_data.detach().cpu().numpy(), (data.shape[2], data.shape[3], data.shape[1])))
                y_train.append(target.item())

    X_train = np.array(X_train)
    return X_train, y_train


def create_dataset(dataset, model, device, test_loader, epsilons):
    X_train, y_train = test(model, device, test_loader, epsilons)

    print(f"Augmented dataset length: {len(X_train)}")
    print(f"Saving new {dataset} dataset.")

    np.save(f"datasets/{dataset}/augmented_FGSM/X_train_FGSM.npy", X_train)
    np.save(f"datasets/{dataset}/augmented_FGSM/y_train_FGSM.npy", y_train)


if __name__ == "__main__":
    datasets = ['fashion_mnist', 'gtsrb']
    model_names_dict = {
        'mnist': [],
        'fashion_mnist': [
            'baseline/fashion_mnist_baseline_100Epochs_50Samples_TrueRobustness_94.pth'
        ],
        'gtsrb': [
            'baseline/gtsrb_baseline_100Epochs_50Samples_TrueRobustness_83.pth'
        ],
        'cifar10': []
    }
    epsilons_dict = {
        'mnist': [.05, .1],
        'fashion_mnist': [.05, .1],
        'gtsrb': [.05, .1],
        'cifar10': [.05, .1],
    }

    for dataset in datasets:
        model_names = model_names_dict.get(dataset)
        epsilons = epsilons_dict.get(dataset)

        for model_name in model_names:
            model_path = f'models/{dataset}/{model_name}'

            use_cuda = torch.cuda.is_available()
            if use_cuda:
                torch.cuda.empty_cache()
            device = torch.device("cuda" if use_cuda else "cpu")
            kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

            if dataset == 'mnist' or dataset == 'fashion_mnist':
                transform_test = transforms.Compose([transforms.ToTensor()])
                Xy_test = MyDataset(dataset=dataset, dtype='baseline', train=False, transform=transform_test)
                model = FASHIONSmall(dim=1).to(device)

            elif dataset == 'gtsrb':
                transform_test = transforms.Compose([transforms.ToTensor()])
                Xy_test = MyDataset(dataset=dataset, dtype='baseline', train=False, transform=transform_test)
                model = GTSRBSmall(dim=1).to(device)

            elif dataset == 'cifar10':
                transform_test = transforms.Compose([transforms.ToTensor()])
                Xy_test = torchvision.datasets.CIFAR10('../data/cifar10', train=False, download=True, transform=transform_test)
                model = ResNet18(dim=3).to(device)

            test_loader = torch.utils.data.DataLoader(Xy_test, shuffle=True, batch_size=1, **kwargs)

            model.load_state_dict(torch.load(model_path))
            model.eval()

            create_dataset(dataset, model, device, test_loader, epsilons)
