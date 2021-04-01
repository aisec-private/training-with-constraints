from dl2.domains import Box
import torch.nn.functional as F
import dl2.diffsat as dl2
import numpy as np
import torch


def transform_network_output(o, network_output):
    if network_output == 'logits':
        pass
    elif network_output == 'prob':
        o = [F.softmax(zo) for zo in o]
    elif network_output == 'logprob':
        o = [F.log_sofmtax(zo) for zo in o]
    return o


class Constraint:
    def eval_z(self, z_batches):
        if self.use_cuda:
            z_inputs = [torch.cuda.FloatTensor(z_batch) for z_batch in z_batches]
        else:
            z_inputs = [torch.FloatTensor(z_batch) for z_batch in z_batches]

        for z_input in z_inputs:
            z_input.requires_grad_(True)
        z_outputs = [self.net(z_input) for z_input in z_inputs]
        for z_out in z_outputs:
            z_out.requires_grad_(True)
        return z_inputs, z_outputs

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        assert False

    def loss(self, x_batches, y_batches, z_batches, args):
        if z_batches is not None:
            z_inp, z_out = self.eval_z(z_batches)
        else:
            z_inp, z_out = None, None

        constr = self.get_condition(z_inp, z_out, x_batches, y_batches)
        
        try:
            neg_losses = dl2.Negate(constr).loss(args)
        except:
            neg_losses = torch.FloatTensor([0])
        pos_losses = constr.loss(args)
        sat = constr.satisfy(args)
            
        return neg_losses, pos_losses, sat, z_inp


class RobustnessConstraint(Constraint):
    def __init__(self, net, eps, delta, use_cuda=True, network_output='logits'):
        self.net = net
        self.network_output = network_output
        self.eps = eps
        self.delta = delta
        self.use_cuda = use_cuda
        self.n_tvars = 1
        self.n_gvars = 1
        self.name = 'RobustnessG'

    def params(self):
        return {'eps': self.eps, 'delta': self.delta}

    def get_domains(self, x_batches, y_batches):
        assert len(x_batches) == 1
        n_batch = x_batches[0].size()[0]

        return [[Box(np.clip(x_batches[0][i].cpu().numpy() - self.eps, 0, 1),
                     np.clip(x_batches[0][i].cpu().numpy() + self.eps, 0, 1))
                for i in range(n_batch)]]

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        n_batch = x_batches[0].size()[0]
        z_out = transform_network_output(z_out, self.network_output)[0]

        pred = z_out[np.arange(n_batch), y_batches[0]]

        limit = torch.FloatTensor([0.3])
        if self.use_cuda:
            limit = limit.cuda()
        return dl2.GEQ(pred, torch.log(limit))


class LipschitzConstraint(Constraint):

    def __init__(self, net, eps, l, use_cuda=True, network_output='logits'):
        self.net = net
        self.eps = eps
        self.l = l
        self.use_cuda = use_cuda
        self.network_output = network_output
        self.n_tvars = 1
        self.n_gvars = 1
        self.name = 'LipschitzG'

    def params(self):
        return {'eps': self.eps, 'L': self.l}

    def get_domains(self, x_batches, y_batches):
        assert len(x_batches) == 1
        n_batch = x_batches[0].size()[0]

        return [[Box(np.clip(x_batches[0][i].cpu().numpy() - self.eps, 0, 1),
                     np.clip(x_batches[0][i].cpu().numpy() + self.eps, 0, 1))
                for i in range(n_batch)]]

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        n_batch = z_inp[0].size()[0]

        x_out = self.net(x_batches[0])
        z_out = z_out[0]

        x_out = torch.clamp(x_out, -100, 100)
        z_out = torch.clamp(z_out, -100, 100)

        out_norm = torch.norm(x_out - z_out, p=float("inf"), dim=1)
        inp_norm = torch.norm((x_batches[0] - z_inp[0]).view((n_batch, -1)), p=float("inf"), dim=1)

        return dl2.LEQ(out_norm, self.l * inp_norm)


class PseudoRobustnessConstraint(Constraint):
    def __init__(self, net, eps, use_cuda=True, network_output='logits'):
        self.net = net
        self.network_output = network_output
        self.eps = eps
        self.use_cuda = use_cuda
        self.n_tvars = 1
        self.n_gvars = 1
        self.name = 'PseudoRobustness'

    def params(self):
        return {'eps': self.eps}

    def get_domains(self, x_batches, y_batches):
        assert len(x_batches) == 1
        n_batch = x_batches[0].size()[0]

        return [[Box(np.clip(x_batches[0][i].cpu().numpy() - self.eps, 0, 1),
                     np.clip(x_batches[0][i].cpu().numpy() + self.eps, 0, 1))
                for i in range(n_batch)]]

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        n_batch = x_batches[0].size()[0]
        z_out = torch.clamp(z_out[0], -100, 100)
        pred_argmax = torch.argmax(z_out, dim=1)

        pred_labels = z_out[np.arange(n_batch), pred_argmax]
        true_labels = z_out[np.arange(n_batch), y_batches[0]]

        return dl2.GEQ(true_labels, pred_labels)


class TrueRobustnessConstraint(Constraint):

    def __init__(self, net, eps, delta, use_cuda=True):
        self.net = net
        self.eps = eps
        self.delta = delta
        self.use_cuda = use_cuda
        self.n_tvars = 1
        self.n_gvars = 1
        self.name = 'TrueRobustness'

    def params(self):
        return {'eps': self.eps, 'delta': self.delta}

    def get_domains(self, x_batches, y_batches):
        assert len(x_batches) == 1
        n_batch = x_batches[0].size()[0]

        return [[Box(np.clip(x_batches[0][i].cpu().numpy() - self.eps, 0, 1),
                     np.clip(x_batches[0][i].cpu().numpy() + self.eps, 0, 1))
                for i in range(n_batch)]]

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        x_out = self.net(x_batches[0])
        z_out = z_out[0]

        x_out = torch.clamp(x_out, -100, 100)
        z_out = torch.clamp(z_out, -100, 100)

        return dl2.LEQ(torch.norm(x_out - z_out, p=float("inf"), dim=1), self.delta)


class FGSMConstraint(Constraint):
    def __init__(self, net, eps, delta, use_cuda=True):
        self.net = net
        self.eps = eps
        self.delta = delta
        self.use_cuda = use_cuda
        self.n_tvars = 1
        self.n_gvars = 0
        self.name = 'FGSM'

    def params(self):
        return{'eps': self.eps, 'delta': self.delta}

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        number_of_samples_found = fgsm_attack(x_batches[0], y_batches[0], self.net, self.eps, self.delta)

        number_of_samples_found = torch.FloatTensor([number_of_samples_found])
        zero = torch.FloatTensor([0])

        if self.use_cuda:
            number_of_samples_found = number_of_samples_found.cuda()
            zero = zero.cuda()

        # This constraint penalizes the network if the number of samples found through the FGSM attack is not zero.
        return dl2.EQ(number_of_samples_found, zero)


def fgsm_attack(x_batches, y_batches, model, epsilon, delta):
    number_of_samples_found = 0

    for i in range(len(x_batches)):
        # Add uniform noise before the attack [-0.0025, +0.0025]
        data_original = torch.reshape(x_batches[i], (1, x_batches[i].shape[0], x_batches[i].shape[1], x_batches[i].shape[2]))
        uniform_noise = torch.Tensor(np.random.random_sample(size=data_original.shape) * 0.005 - 0.0025).to("cuda")
        data = torch.clamp(data_original + uniform_noise, min=0, max=1)
        target = torch.reshape(y_batches[i], (1, 1))[0]
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
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_data = data + epsilon * sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        # Re-classify the perturbed image
        output_p = model(perturbed_data)
        
        # # Check for success
        # final_pred = output_p.max(1, keepdim=True)[1] # get the index of the max log-probability
        #
        # if final_pred.item() != target.item():
        #     number_of_samples_found += 1
        
        # output = F.softmax(output, dim=1)
        # output_p = F.softmax(output_p, dim=1)

        # output -= output.min(1, keepdim=True)[0]
        # output /= output.max(1, keepdim=True)[0]
        # output_p -= output_p.min(1, keepdim=True)[0]
        # output_p /= output_p.max(1, keepdim=True)[0]

        output = torch.clamp(output, -100, 100)
        output_p = torch.clamp(output_p, -100, 100)

        distance = torch.norm(output - output_p, p=float("inf"), dim=1)
        if distance > delta:
            number_of_samples_found += 1

    return number_of_samples_found / x_batches.size()[0]


class PGDConstraint(Constraint):
    def __init__(self, net, eps, alpha, iters, use_cuda=True):
        self.net = net
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        self.use_cuda = use_cuda
        self.n_tvars = 1
        self.n_gvars = 0
        self.name = 'PGD'

    def params(self):
        return{'eps': self.eps}

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        number_of_samples_found = pgd_attack(x_batches[0], y_batches[0], self.net, self.eps, self.alpha, self.iters)

        number_of_samples_found = torch.FloatTensor([number_of_samples_found])
        zero = torch.FloatTensor([0])

        if self.use_cuda:
            number_of_samples_found = number_of_samples_found.cuda()
            zero = zero.cuda()

        # This constraint penalizes the network if the number of samples found through the FGSM attack is not zero.
        return dl2.EQ(number_of_samples_found, zero)


def pgd_attack(x_batches, y_batches, model, epsilon, alpha, iters):
    number_of_samples_found = 0

    for i in range(len(x_batches)):
        # Add uniform noise before the attack [-0.0025, +0.0025]
        data_original = torch.reshape(x_batches[i], (1, x_batches[i].shape[0], x_batches[i].shape[1], x_batches[i].shape[2]))
        uniform_noise = torch.Tensor(np.random.random_sample(size=data_original.shape) * 0.005 - 0.0025).to("cuda")
        data = torch.clamp(data_original + uniform_noise, min=0, max=1)
        target = torch.reshape(y_batches[i], (1, 1))[0]

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        original_image = data.data

        for _ in range(iters):
            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True
            # Forward pass the data through the model
            output = model(data)
            # Calculate the loss
            loss = F.nll_loss(output, target)
            # Zero all existing gradients
            model.zero_grad()
            # Calculate gradients of model in backward pass
            loss.backward()
            # Collect datagrad
            data_grad = data.grad.data
            # Collect the element-wise sign of the data gradient
            sign_data_grad = data_grad.sign()
            # Create the perturbed image by adjusting each pixel of the input image
            perturbed_data = data + alpha * sign_data_grad
            eta = torch.clamp(perturbed_data - original_image, min=-epsilon, max=epsilon)
            # Adding clipping to maintain [0,1] range
            data = torch.clamp(original_image + eta, min=0, max=1).detach()

        # Re-classify the perturbed image
        output = model(data)
        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        if final_pred.item() != target.item():
            number_of_samples_found += 1

    return number_of_samples_found


class RobustnessConstraint1Class(Constraint):
    def __init__(self, net, eps, delta, use_cuda=True, network_output='logits'):
        self.net = net
        self.network_output = network_output
        self.eps = eps
        self.delta = delta
        self.use_cuda = use_cuda
        self.n_tvars = 1
        self.n_gvars = 1
        self.name = 'RobustnessG1Class'

    def params(self):
        return {'eps': self.eps, 'network_output': self.network_output}

    def get_domains(self, x_batches, y_batches):
        assert len(x_batches) == 1
        n_batch = x_batches[0].size()[0]

        return [[Box(np.clip(x_batches[0][i].cpu().numpy() - self.eps, 0, 1),
                     np.clip(x_batches[0][i].cpu().numpy() + self.eps, 0, 1))
                for i in range(n_batch)]]

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        n_batch = x_batches[0].size()[0]
        z_out = transform_network_output(z_out, self.network_output)[0]

        pred = z_out[np.arange(n_batch), y_batches[0]]

        for i in range(len(y_batches[0])):
            if y_batches[0][i] != 6:
                pred[i] = 1

        limit = torch.FloatTensor([0.3])
        if self.use_cuda:
            limit = limit.cuda()
        return dl2.GEQ(pred, torch.log(limit))
