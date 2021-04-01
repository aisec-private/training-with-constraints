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


def TrueRobustness(eps, delta):
    return lambda model, use_cuda, network_output: TrueRobustnessConstraint(model, eps, delta, use_cuda)


def train(args, oracle, net, device, train_loader, optimizer, epoch):
    t1 = time.time()
    num_steps = 0
    avg_train_acc, avg_PGD_acc = 0, 0
    avg_ce_loss, avg_PGD_loss = 0, 0
    ce_loss = torch.nn.CrossEntropyLoss()
    ce_PGD_loss = torch.nn.CrossEntropyLoss()

    print('\nEpoch ', epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.float()
        x_batch, y_batch = data.to(device), target.to(device)
        n_batch = int(x_batch.size()[0])
        num_steps += 1

        x_outputs = net(x_batch)
        x_outputs = torch.clamp(x_outputs, -100, 100)
        ce_batch_loss = ce_loss(x_outputs, y_batch)
        x_correct = torch.mean(torch.argmax(x_outputs, dim=1).eq(y_batch).float())

        x_batches, y_batches = [], []
        k = n_batch // oracle.constraint.n_tvars
        assert n_batch % oracle.constraint.n_tvars == 0, 'Batch size must be divisible by number of train variables!'
        for i in range(oracle.constraint.n_tvars):
            x_batches.append(x_batch[i:(i + k)])
            y_batches.append(y_batch[i:(i + k)])

        net.eval()

        domains = oracle.constraint.get_domains(x_batches, y_batches)
        z_batches = oracle.general_attack(x_batches, y_batches, domains, num_restarts=1, num_iters=args.num_iters, args=args)
        z_inputs = [torch.cuda.FloatTensor(z_batch) for z_batch in z_batches]

        z_outputs = net(z_inputs[0])
        z_outputs = torch.clamp(z_outputs, -100, 100)
        PGD_batch_loss = ce_PGD_loss(z_outputs, y_batch)

        avg_train_acc += x_correct.item()
        avg_ce_loss += ce_batch_loss.item()
        avg_PGD_loss += PGD_batch_loss.item()

        if batch_idx % args.print_freq == 0:
            print(f'[{batch_idx}] Train p_acc: {x_correct.item():.4f}, CE loss: {ce_batch_loss.item():.4f}, PGD loss: {PGD_batch_loss.item():.4f}')

        net.train()
        optimizer.zero_grad()
        tot_batch_loss = args.alfa * ce_batch_loss + args.beta * PGD_batch_loss
        tot_batch_loss.backward()
        optimizer.step()
    t2 = time.time()
        
    avg_train_acc /= float(num_steps)
    avg_PGD_loss /= float(num_steps)
    avg_ce_loss /= float(num_steps)
    
    return avg_train_acc, avg_PGD_loss, avg_ce_loss, t1, t2


def test(args, model, device, test_loader):
    loss = torch.nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct, constr, num_steps, pgd_ok = 0, 0, 0, 0
    
    for data, target in test_loader:
        num_steps += 1
        data = data.float()
        x_batch, y_batch = data.to(device), target.to(device)

        output = model(x_batch)
        output = torch.clamp(output, -100, 100)
        
        test_loss += loss(output, y_batch).item()  # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            
        correct += pred.eq(y_batch.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f"\nTest p_acc: {correct / len(test_loader.dataset):.4f}, Average loss: {test_loss:.4f}")

    return correct / len(test_loader.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train NN with constraints.')
    parser = dl2.add_default_parser_args(parser)
    parser.add_argument('--batch-size', type=int, default=128, help='Number of samples in a batch.')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs to train for.')
    parser.add_argument('--alfa', type=float, default=1, help='Weight of ce loss.')
    parser.add_argument('--beta', type=float, default=0.2, help='Weight of ce PGD loss.')
    parser.add_argument('--num-iters', type=int, default=50, help='PGD number of iterations.')
    parser.add_argument('--dataset', type=str, required=True, help=['mnist', 'fashion_mnist', 'gtsrb', 'cifar10', 'coil20unproc', 'coil20proc'])
    parser.add_argument('--print-freq', type=int, default=10, help='Print frequency.')
    parser.add_argument('--network-output', type=str, choices=['logits', 'prob', 'logprob'], default='logits', help='Wether to treat the output of the network as logits, probabilities or log(probabilities) in the constraints.')
    parser.add_argument('--constraint', type=str, default="TrueRobustness(eps=0.1, delta=10)", help='the constraint to train with')
    parser.add_argument('--report-dir', type=str, default='reports', help='Directory where results should be stored')
    parser.add_argument('--dtype', type=str, default='baseline', choices=['baseline', 'augmented', 'augmented_PGD'])
    parser.add_argument('--net-size', type=str, default='small', choices=['big', 'small'],help='Whether to use the big or small network')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    
    dev_name = "CPU"
    if use_cuda:
        torch.cuda.empty_cache()
        dev_name = torch.cuda.get_device_name(0)

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    print(f'DEVICE: {dev_name}')

    if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
        Xy_train = MyDataset(dataset=args.dataset, dtype=args.dtype, train=True, transform=transform_train)
        Xy_test = MyDataset(dataset=args.dataset, dtype=args.dtype, train=False, transform=transform_test)
        if args.net_size == 'big':
            model = MnistNet(dim=1).to(device)
        elif args.net_size == 'small':
            model = FASHIONSmall(dim=1).to(device)

    elif args.dataset == 'gtsrb':
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
        Xy_train = MyDataset(dataset=args.dataset, dtype=args.dtype, train=True, transform=transform_train)
        Xy_test = MyDataset(dataset=args.dataset, dtype=args.dtype, train=False, transform=transform_test)
        if args.net_size == 'big':
            model = GTSRBNet(dim=1).to(device)
        elif args.net_size == 'small':
            model = GTSRBSmall(dim=1).to(device)

    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([transforms.ToTensor()])
        #Xy_train = datasets.CIFAR10('../data/cifar10', train=True, download=True, transform=transform_train)
        #Xy_test = datasets.CIFAR10('../data/cifar10', train=False, download=True, transform=transform_test)
        Xy_train = MyDataset(dataset=args.dataset, dtype=args.dtype, train=True, transform=transform_train)
        Xy_test = MyDataset(dataset=args.dataset, dtype=args.dtype, train=False, transform=transform_test)
        model = ResNet18(dim=3).to(device)

    elif args.dataset == 'coil20unproc' or args.dataset == 'coil20proc':
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
        Xy_train = MyDataset(dataset=args.dataset, dtype=args.dtype, train=True, transform=transform_train)
        Xy_test = MyDataset(dataset=args.dataset, dtype=args.dtype, train=False, transform=transform_test)
        model = Coil20Net(dim=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    train_loader = torch.utils.data.DataLoader(Xy_train, shuffle=True, batch_size=args.batch_size, **kwargs)
    test_loader = torch.utils.data.DataLoader(Xy_test, shuffle=True, batch_size=args.batch_size, **kwargs)

    constraint = eval(args.constraint)(model, use_cuda, network_output=args.network_output)
    oracle = DL2_Oracle(learning_rate=0.01, net=model, constraint=constraint, use_cuda=use_cuda)

    dtype = args.dtype
    report_file = os.path.join(args.report_dir, f"{args.dataset}_{dtype}_{args.num_epochs}Epochs_adversarial_PGD.json")
    data_dict = {
        'dataset': args.dataset,
        'dtype': dtype,
        'epochs': args.num_epochs,
        'name': 'PGD Loss',
        'ce_loss': [],
        'PGD_loss': [],
        'Train p_acc': [],
        'Test p_acc': [],
        'epoch_time': [],
        'test_time': [],
        'total_time': []
    }

    for epoch in range(1, args.num_epochs + 1):
        avg_train_acc, avg_PGD_loss, avg_ce_loss, t1, t2 = \
            train(args, oracle, model, device, train_loader, optimizer, epoch)
        data_dict['Train p_acc'].append(avg_train_acc)
        data_dict['ce_loss'].append(avg_ce_loss)
        data_dict['PGD_loss'].append(avg_PGD_loss)
        
        p_acc = test(args, model, device, test_loader)
        data_dict['Test p_acc'].append(p_acc)

        t3 = time.time()
        epoch_time = t2 - t1
        test_time = t3 - t2
        total_time = t3 - t1
        data_dict['epoch_time'].append(epoch_time)
        data_dict['test_time'].append(test_time)
        data_dict['total_time'].append(total_time)

        print(f'Epoch Time [s]: {epoch_time:.4f} - Test Time [s]: {test_time:.4f} - Total Time [s]: {total_time:.4f}')

        if epoch > (args.num_epochs - 25):
            torch.save(model.state_dict(), f"models/{args.dataset}/{dtype}/{args.dataset}_{dtype}_{args.num_epochs}Epochs_{epoch}_adversarial_PGD.pth")

    with open(report_file, 'w') as fou:
        json.dump(data_dict, fou, indent=4)
