from dl2.models import MnistNet, MnistNetTanh, GTSRBNet, ResNet18, FASHIONSmall, GTSRBSmall
from torchvision import transforms
from dl2.getDatasets import MyDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
import torch
import json
import numpy as np
from PIL import Image


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def test(model, device, test_loader, epsilon, delta, dataset, dtype):
    # Accuracy counter
    correct = 0
    pseudo_correct = 0
    output_array = []
    mean_distance = 0
    mean_adversarial_distance = 0
    max_distance = 0
    max_adversarial_distance = 0
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
        output_fgsm = model(perturbed_data)
        # Check for success
        final_pred = output_fgsm.max(1, keepdim=True)[1] # get the index of the max log-probability

        # output = F.softmax(output, dim=1)
        # output_fgsm = F.softmax(output_fgsm, dim=1)

        # output -= output.min(1, keepdim=True)[0]
        # output /= output.max(1, keepdim=True)[0]
        # output_fgsm -= output_fgsm.min(1, keepdim=True)[0]
        # output_fgsm /= output_fgsm.max(1, keepdim=True)[0]

        output = torch.clamp(output, -100, 100)
        output_fgsm = torch.clamp(output_fgsm, -100, 100)

        output_distance = distance_output(output, output_fgsm)
        output_array.append(output_fgsm.detach().cpu().numpy()[0].tolist())

        distance = distance_image(data, perturbed_data)
        mean_distance += distance
        if distance > max_distance:
            max_distance = distance

        if final_pred.item() == target.item():
            pseudo_correct += 1
        if output_distance <= delta:
            correct += 1
        else:
            mean_adversarial_distance += distance
            if distance > max_adversarial_distance:
                max_adversarial_distance = distance
           
    # Calculate robustness and pseudo-robustness for this epsilon
    rob = correct/(len(test_loader))
    pseudo_rob = pseudo_correct/(len(test_loader))
    mean_distance /= (len(test_loader))
    if (len(test_loader)) - correct != 0:
        mean_adversarial_distance /= ((len(test_loader)) - correct)
    else:
        mean_adversarial_distance = 0
    print(f"        Eps: {epsilon}\tRobustness: {rob:.4f} ({correct}/{(len(test_loader))})\tPseudo-Robustness: {pseudo_rob:.4f} ({pseudo_correct}/{(len(test_loader))})\tMean D.: {mean_distance:.4f}\tMean Adv. D.: {mean_adversarial_distance:.4f}\tMax D.: {max_distance:.4f}\tMax Adv. D.: {max_adversarial_distance:.4f}")
    # Return the accuracy and an adversarial example
    return rob, pseudo_rob, output_array, mean_distance, mean_adversarial_distance, max_distance, max_adversarial_distance


def distance_output(output1, output2):
    out1 = output1.detach().cpu().numpy()
    out2 = output2.detach().cpu().numpy()
    # distance = np.linalg.norm(out1 - out2, ord=None)
    distance = np.max(abs(out1 - out2))
    return float(distance)


def distance_image(image1, image2):
    img1 = image1.detach().cpu().numpy()
    img2 = image2.detach().cpu().numpy()
    # distance = np.linalg.norm(img1 - img2, ord=None)
    distance = np.max(abs(img1 - img2))
    return float(distance)


if __name__ == "__main__":
    datasets = ['fashion_mnist', 'gtsrb']
    model_names_dict = {
        'fashion_mnist': [
            'baseline/fashion_mnist_baseline_100Epochs_50Samples_TrueRobustness_94.pth',
            'augmented/fashion_mnist_augmented_100Epochs_50Samples_TrueRobustness_95.pth',
            'augmented_FGSM/fashion_mnist_augmented_FGSM_100Epochs_50Samples_TrueRobustness_98.pth',
            'baseline/fashion_mnist_baseline_100Epochs_96_small_adversarial_FGSM.pth',
            'dl2/fashion_mnist_dl2_100Epochs_50Samples_TrueRobustness_94.pth',
            'dl2/fashion_mnist_dl2_100Epochs_50Samples_FGSM_94.pth',
        ],
        'gtsrb': [
            'baseline/gtsrb_baseline_100Epochs_50Samples_TrueRobustness_83.pth',
            'augmented/gtsrb_augmented_100Epochs_50Samples_TrueRobustness_92.pth',
            'augmented_FGSM/gtsrb_augmented_FGSM_100Epochs_50Samples_TrueRobustness_99.pth',
            'baseline/gtsrb_baseline_100Epochs_95_small_adversarial_FGSM.pth',
            'dl2/gtsrb_dl2_100Epochs_50Samples_TrueRobustness_100.pth',
            'dl2/gtsrb_dl2_100Epochs_50Samples_FGSM_83.pth',
        ]
    }
    epsilons_dict = {
        'fashion_mnist': [0, .02, .04, .06, .08, .1],
        'gtsrb': [0, .02, .04, .06, .08, .1],
    } 
    delta_dict = {
        'fashion_mnist': [10],
        'gtsrb': [10],
    }
    report_file = f'reports/FGSM_Attack_Small.json'
    data_dict = []
    outputs_report_file = f'reports/FGSM_Attack_Outputs_Small.json'
    outputs_data_dict = []

    for dataset in datasets:
        model_names = model_names_dict.get(dataset)
        epsilons = epsilons_dict.get(dataset)
        deltas = delta_dict.get(dataset)

        for delta in deltas:
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

                print(f'{model_name} (delta: {delta}):')
                dtype = model_name.rpartition('/')[0]
                model_dict = {
                    'model_name': model_name,
                    'dataset': dataset,
                    'dtype': dtype,
                    'delta': delta,
                    'attacks': []
                }
                outputs_dict = {
                    'model_name': model_name,
                    'delta': delta,
                    'attacks': []
                }

                for eps in epsilons:
                    rob, pseudo_rob, output_array, mean_distance, mean_adversarial_distance, max_distance, max_adversarial_distance = test(model, device, test_loader, eps, delta, dataset, dtype)
                    eps_dict = {
                        'epsilon': eps,
                        'robustness': rob,
                        'pseudo_robustness': pseudo_rob,
                        'mean_distance': mean_distance,
                        'mean_adversarial_distance': mean_adversarial_distance,
                        'max_distance': max_distance,
                        'max_adversarial_distance': max_adversarial_distance
                    }
                    model_dict['attacks'].append(eps_dict)
                    outputs_eps_dict = {
                        'epsilon': eps,
                        'outputs': output_array,
                    }
                    outputs_dict['attacks'].append(outputs_eps_dict)
                
                data_dict.append(model_dict)
                outputs_data_dict.append(outputs_dict)

                with open(report_file, 'w') as fou:
                    json.dump(data_dict, fou, indent=4)

                with open(outputs_report_file, 'w') as fee:
                    json.dump(outputs_data_dict, fee, indent=4)
