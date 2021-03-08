from dl2.models import MnistNet, MnistNetTanh, GTSRBNet, ResNet18, Coil20Net, FASHIONSmall, GTSRBSmall
from torchvision import transforms, datasets
from dl2.getDatasets import MyDataset
from dl2.oracles import DL2_Oracle
from dl2.constraints import *
import torch.optim as optim
import numpy as np
import dl2.args as dl2
import argparse
import torch
import json
import time
import os


def RobustnessG(eps, delta):
    return lambda model, use_cuda, network_output: RobustnessConstraint(model, eps, delta, use_cuda, network_output=network_output)


def RobustnessG1Class(eps, delta):
    return lambda model, use_cuda, network_output: RobustnessConstraint1Class(model, eps, delta, use_cuda, network_output=network_output)


def TrueRobustness(eps, delta):
    return lambda model, use_cuda, network_output: TrueRobustnessConstraint(model, eps, delta, use_cuda)


def FGSM(eps, delta):
    return lambda model, use_cuda, network_output: FGSMConstraint(model, eps, delta, use_cuda)


def PGD(eps, alpha, iters):
    return lambda model, use_cuda, network_output: PGDConstraint(model, eps, alpha, iters, use_cuda)


def test(args, oracle, model, device, test_loader):
    loss = torch.nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct, constr, num_steps, pgd_ok = 0, 0, 0, 0
    
    for data, target in test_loader:
        print(f'\r{num_steps}/{len(test_loader)}', end='', flush=True)
        num_steps += 1
        data = data.float()
        x_batch, y_batch = data.to(device), target.to(device)
        
        n_batch = int(x_batch.size()[0])
        x_batches, y_batches = [], []
        k = n_batch // oracle.constraint.n_tvars
        assert n_batch % oracle.constraint.n_tvars == 0, 'Batch size must be divisible by number of train variables!'

        for i in range(oracle.constraint.n_tvars):
            x_batches.append(x_batch[i:(i + k)])
            y_batches.append(y_batch[i:(i + k)])
        
        if oracle.constraint.n_gvars > 0:
            domains = oracle.constraint.get_domains(x_batches, y_batches)
            z_batches = oracle.general_attack(x_batches, y_batches, domains, num_restarts=1, num_iters=samples, args=args)
            _, dl2_batch_loss, constr_acc = oracle.evaluate(x_batches, y_batches, z_batches, args)
        else:
            _, dl2_batch_loss, constr_acc = oracle.evaluate(x_batches, y_batches, None, args)
        
        output = model(x_batch)
        output = torch.clamp(output, -100, 100)
        
        test_loss += loss(output, y_batch).item()  # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            
        correct += pred.eq(y_batch.view_as(pred)).sum().item()
        constr += constr_acc.item()

    test_loss /= len(test_loader.dataset)
    print(f"\nTest p_acc: {correct / len(test_loader.dataset):.4f}, Test c_acc: {(constr / float(num_steps)):.4f}, Average loss: {test_loss:.4f}")

    return correct / len(test_loader.dataset), constr / float(num_steps)


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
    report_file = f'reports/Constraint_Accuracy.json'
    data_dict = []
    # constraint = "TrueRobustness(eps=0.1, delta=10)"
    constraint = "RobustnessG(eps=0.1, delta=0.52)"
    samples = 50

    args = dl2.add_default_parser_args(argparse.ArgumentParser()).parse_args()

    for dataset in datasets:
        model_names = model_names_dict.get(dataset)

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

                test_loader = torch.utils.data.DataLoader(Xy_test, shuffle=True, batch_size=128, **kwargs)
                model.load_state_dict(torch.load(model_path))

                constr = eval(constraint)(model, use_cuda, network_output='logits')
                oracle = DL2_Oracle(learning_rate=0.01, net=model, constraint=constr, use_cuda=use_cuda)
                
                model_dict = {
                    'model_name': model_name,
                    'samples': samples,
                    'name': constr.name,
                    'constraint_params': constr.params(),
                    'Test p_acc': [],
                    'Test c_acc': [],
                }

                print(f'{model_name}')
                p, c = test(args, oracle, model, device, test_loader)
                model_dict['Test p_acc'].append(p)
                model_dict['Test c_acc'].append(c)

                data_dict.append(model_dict)

                with open(report_file, 'w') as fou:
                    json.dump(data_dict, fou, indent=4)
