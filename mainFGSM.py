from dl2.models import MnistNet, MnistNetTanh, GTSRBNet, ResNet18, Coil20Net, FASHIONSmall, GTSRBSmall
from torchvision import transforms, datasets
from dl2.getDatasets import MyDataset
import torch.optim as optim
import numpy as np
import argparse
import torch
import json
import time
import os


def fgsm_attack(model, x_batch, y_batch, epsilon, device):
    model.eval()
    correct = 0
    outputs = []
    ce_FGSM_loss = torch.nn.CrossEntropyLoss()
    # Loop over all examples in test set
    for i in range(len(x_batch)):
        data = torch.reshape(x_batch[i], (1, x_batch[i].shape[0], x_batch[i].shape[1], x_batch[i].shape[2]))
        target = torch.reshape(y_batch[i], (1, 1))[0]
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            outputs.append(output)
            continue
        # Calculate the loss
        loss = torch.nn.functional.nll_loss(output, target)
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()
        # Collect datagrad
        data_grad = data.grad.data
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_data = data + epsilon * sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        # Re-classify the perturbed image
        output = model(perturbed_data)
        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        outputs.append(output)

        if final_pred.item() == target.item():
            correct += 1

    outputs = torch.squeeze(torch.stack(outputs))
    outputs = torch.clamp(outputs, -100, 100)

    FGSM_acc = torch.squeeze(torch.Tensor([correct/(len(x_batch))]).to(device))
    FGSM_loss = ce_FGSM_loss(outputs, y_batch)

    model.train()

    return FGSM_acc, FGSM_loss


def train(args, net, device, train_loader, optimizer, epoch):
    t1 = time.time()
    num_steps = 0
    avg_train_acc, avg_FGSM_acc = 0, 0
    avg_ce_loss, avg_FGSM_loss = 0, 0
    ce_loss = torch.nn.CrossEntropyLoss()

    print('\nEpoch ', epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.float()
        x_batch, y_batch = data.to(device), target.to(device)
        n_batch = int(x_batch.size()[0])
        num_steps += 1

        x_outputs = net(x_batch)
        x_outputs = torch.clamp(x_outputs, -100, 100)

        x_correct = torch.mean(torch.argmax(x_outputs, dim=1).eq(y_batch).float())
        ce_batch_loss = ce_loss(x_outputs, y_batch)
        
        # FGSM attack
        FGSM_acc, FGSM_batch_loss = fgsm_attack(net, x_batch, y_batch, args.eps, device)

        avg_train_acc += x_correct.item()
        avg_FGSM_acc += FGSM_acc.item()
        avg_ce_loss += ce_batch_loss.item()
        avg_FGSM_loss += FGSM_batch_loss.item()

        if batch_idx % args.print_freq == 0:
            print(f'[{batch_idx}] Train p_acc: {x_correct.item():.4f}, Train c_acc: {FGSM_acc.item():.4f}, CE loss: {ce_batch_loss.item():.4f}, FGSM loss: {FGSM_batch_loss.item():.4f}')

        net.train()
        optimizer.zero_grad()
        tot_batch_loss = ce_batch_loss + args.fgsm_weight * FGSM_batch_loss
        tot_batch_loss.backward()
        optimizer.step()
    t2 = time.time()
        
    avg_train_acc /= float(num_steps)
    avg_FGSM_acc /= float(num_steps)
    avg_FGSM_loss /= float(num_steps)
    avg_ce_loss /= float(num_steps)
    
    return avg_train_acc, avg_FGSM_acc, avg_FGSM_loss, avg_ce_loss, t1, t2


def test(args, model, device, test_loader):
    loss = torch.nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct, constr, num_steps, pgd_ok = 0, 0, 0, 0
    
    for data, target in test_loader:
        num_steps += 1
        data = data.float()
        x_batch, y_batch = data.to(device), target.to(device)
        n_batch = int(x_batch.size()[0])
        
        # FGSM attack
        FGSM_acc, FGSM_batch_loss = fgsm_attack(model, x_batch, y_batch, args.eps, device)
        
        output = model(x_batch)
        output = torch.clamp(output, -100, 100)
        
        test_loss += loss(output, y_batch).item()  # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            
        correct += pred.eq(y_batch.view_as(pred)).sum().item()
        constr += FGSM_acc.item()

    test_loss /= len(test_loader.dataset)
    print(f"\nTest p_acc: {correct / len(test_loader.dataset):.4f}, Test c_acc: {(constr / float(num_steps)):.4f}, Average loss: {test_loss:.4f}")

    return correct / len(test_loader.dataset), constr / float(num_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train NN with constraints.')
    parser.add_argument('--batch-size', type=int, default=128, help='Number of samples in a batch.')
    parser.add_argument('--eps', type=float, required=True, help='FGSM epsilon.')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs to train for.')
    parser.add_argument('--fgsm-weight', type=float, default=0.2, help='Weight of FGSM loss.')
    parser.add_argument('--dataset', type=str, required=True, help=['mnist', 'fashion_mnist', 'gtsrb', 'cifar10', 'coil20unproc', 'coil20proc'])
    parser.add_argument('--print-freq', type=int, default=10, help='Print frequency.')
    parser.add_argument('--report-dir', type=str, default='reports', help='Directory where results should be stored')
    parser.add_argument('--dtype', type=str, default='baseline', choices=['baseline', 'augmented', 'augmented_FGSM', 'augmented_PGD'])
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

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    train_loader = torch.utils.data.DataLoader(Xy_train, shuffle=True, batch_size=args.batch_size, **kwargs)
    test_loader = torch.utils.data.DataLoader(Xy_test, shuffle=True, batch_size=args.batch_size, **kwargs)

    dtype = args.dtype
    report_file = os.path.join(args.report_dir, f"{args.dataset}_{dtype}_{args.num_epochs}Epochs_{args.net_size}_adversarial_FGSM.json")
    data_dict = {
        'dataset': args.dataset,
        'dtype': dtype,
        'epochs': args.num_epochs,
        'name': 'FGSM Loss',
        'ce_loss': [],
        'FGSM_loss': [],
        'Train p_acc': [],
        'Train FGSM_acc': [],
        'Test p_acc': [],
        'Test FGSM_acc': [],
        'epoch_time': [],
        'test_time': [],
        'total_time': []
    }

    for epoch in range(1, args.num_epochs + 1):
        avg_train_acc, avg_FGSM_acc, avg_FGSM_loss, avg_ce_loss, t1, t2 = \
            train(args, model, device, train_loader, optimizer, epoch)
        data_dict['Train p_acc'].append(avg_train_acc)
        data_dict['Train FGSM_acc'].append(avg_FGSM_acc)
        data_dict['ce_loss'].append(avg_ce_loss)
        data_dict['FGSM_loss'].append(avg_FGSM_loss)
        
        p_acc, FGSM_acc = test(args, model, device, test_loader)
        data_dict['Test p_acc'].append(p_acc)
        data_dict['Test FGSM_acc'].append(FGSM_acc)

        t3 = time.time()
        epoch_time = t2 - t1
        test_time = t3 - t2
        total_time = t3 - t1
        data_dict['epoch_time'].append(epoch_time)
        data_dict['test_time'].append(test_time)
        data_dict['total_time'].append(total_time)

        print(f'Epoch Time [s]: {epoch_time:.4f} - Test Time [s]: {test_time:.4f} - Total Time [s]: {total_time:.4f}')

        if epoch > (args.num_epochs - 25):
            torch.save(model.state_dict(), f"models/{args.dataset}/{dtype}/{args.dataset}_{dtype}_{args.num_epochs}Epochs_{epoch}_{args.net_size}_adversarial_FGSM.pth")

    with open(report_file, 'w') as fou:
        json.dump(data_dict, fou, indent=4)
