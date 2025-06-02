import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.optimize import minimize_scalar

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_samples = 50000
train_frac = 0.5
num_mixture_samples = int(train_frac * num_samples)
num_train_samples = int(train_frac * num_samples)
num_val_samples = num_samples - num_train_samples

train_batch_size = 5000
val_batch_size = 1*train_batch_size
num_subnets_arr = [2, 4, 8, 16]

num_runs = 300
num_trainings = 10
units = 32
latent_dim = 32

true_arr = torch.zeros((len(num_subnets_arr), num_trainings, num_runs, 6))
pred_arr = torch.zeros((len(num_subnets_arr), num_trainings, num_runs, 6))
unc_arr = torch.zeros((len(num_subnets_arr), num_trainings, num_runs, 6))
z_score_arr = torch.zeros((len(num_subnets_arr), num_trainings, num_runs, 6))


def model_with_weights_scaled(submodels, dataloader, w):
    dataset_length = len(dataloader.dataset)
    h_outs = torch.zeros(dataset_length, device=device)
    end_idx = 0
    for data in dataloader:
        start_idx = end_idx
        end_idx += len(data[0])
        h_outs[start_idx:end_idx] = submodels.submodel_all(data[0].to(device))@w
    return h_outs


def MLC(y_true, model_outputs, net_idx):
    w00 = torch.zeros(model_outputs.shape[-1], device=device)
    w00[net_idx] = 1.
    y_pred = (model_outputs@w00).unsqueeze(1)
    cont1 = -(y_true.unsqueeze(1) * y_pred)
    cont2 = -(1-y_true.unsqueeze(1)) * (1 - torch.exp(y_pred))
    cont3 = -(1-y_true.unsqueeze(1))*(-y_pred)
    cont4 = -(y_true.unsqueeze(1)) * (1 - torch.exp(-y_pred))
    return cont1+cont2+cont3+cont4


class Model(nn.Module):
    def __init__(self, num_subnets):
        super(Model, self).__init__()
        self.num_subnets = num_subnets

        self.layer1 = nn.Linear(1, units)
        self.layer2 = nn.Linear(units, units)
        self.layer3 = nn.Linear(units, 1)

        self.layer1_list = nn.ModuleList([nn.Linear(1, units) for i in range(self.num_subnets)])
        self.layer2_list = nn.ModuleList([nn.Linear(units, units) for i in range(self.num_subnets)])
        self.layer3_list = nn.ModuleList([nn.Linear(units, 1) for i in range(self.num_subnets)])

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x), negative_slope=0.2)
        x = F.leaky_relu(self.layer2(x), negative_slope=0.2)
        x = self.layer3(x)
        return x

    def submodel_all(self, x):
        x1 = [F.leaky_relu(self.layer1_list[i](x), negative_slope=0.2) for i in range(self.num_subnets)]
        x1 = [F.leaky_relu(self.layer2_list[i](x1[i]), negative_slope=0.2) for i in range(self.num_subnets)]
        x1 = [self.layer3_list[i](x1[i]) for i in range(self.num_subnets)]
        x1 = torch.cat(x1, axis=1)
        return x1


def train_func(model_to_train, epochs, num_subnets):
    train_losses = []
    val_losses = []
    opt = optim.Adam(model_to_train.parameters(), lr=1e-2)
    for start_idx in range(num_subnets):
        train_strap = np.random.randint(0, x_train.shape[0], x_train.shape[0])
        x_train_strap = x_train[train_strap]
        y_train_strap = y_train[train_strap]
        val_strap = np.random.randint(0, x_val.shape[0], x_val.shape[0])
        x_val_strap = x_val[val_strap]
        y_val_strap = y_val[val_strap]

        trainset_strap = torch.utils.data.TensorDataset(x_train_strap, y_train_strap)
        valset_strap = torch.utils.data.TensorDataset(x_val_strap, y_val_strap)
        trainloader_strap = torch.utils.data.DataLoader(trainset_strap, batch_size=train_batch_size, shuffle=True)
        valloader_strap = torch.utils.data.DataLoader(valset_strap, batch_size=val_batch_size, shuffle=False)
        for param in model_to_train.parameters():
            param.requires_grad = False
        for param in model_to_train.layer1_list[start_idx].parameters():
            param.requires_grad = True
        for param in model_to_train.layer2_list[start_idx].parameters():
            param.requires_grad = True
        for param in model_to_train.layer3_list[start_idx].parameters():
            param.requires_grad = True
        print(f"Training basis function {start_idx}", flush=True)
        min_val_loss = 1e10
        for epoch in range(epochs):
            running_loss = 0.0
            val_loss = 0.0
            batches = 0
            for i, data in enumerate(trainloader_strap):
                batches += 1
                opt.zero_grad()
                inputs = data[0].to(device)
                train_outputs = model_to_train.submodel_all(inputs)
                loss = MLC(data[1].to(device), train_outputs, start_idx).mean()
                loss.backward()
                opt.step()
                running_loss += loss.item()
            val_batches = 0
            with torch.no_grad():
                for i, data in enumerate(valloader_strap):
                    val_batches += 1
                    inputs = data[0].to(device)
                    val_outputs = model_to_train.submodel_all(inputs)
                    val_loss += MLC(data[1].to(device), val_outputs, start_idx).mean().item()
            train_losses.append(running_loss/batches)
            val_losses.append(val_loss/val_batches)
            if val_loss/val_batches < min_val_loss:
                min_val_loss = val_loss/val_batches
                best_model = model_to_train.state_dict()
            print(f"Epoch {epoch+1} train loss: {running_loss/batches}, val loss: {val_loss/val_batches}", flush=True)
        model_to_train.load_state_dict(best_model)
    return train_losses, val_losses


def neg_maximum_likelihood_f(f, model_outputs):
    return -torch.log(F.softplus(torch.exp(model_outputs)*f + (1-f), beta=100)).sum()


def neg_maximum_likelihood_f_wrapper(*args):
    return neg_maximum_likelihood_f(*args).detach().cpu().numpy()


def calc_ai(f, model_outputs_all, w):
    model_outputs = model_outputs_all@w
    ai = (((torch.exp(model_outputs)/(torch.exp(model_outputs)*f + (1-f))**2)).unsqueeze(1)*model_outputs_all).sum(axis=0)
    return ai.detach()


def calc_second_deriv(f, model_outputs_all, w):
    model_outputs = model_outputs_all@w
    second_deriv = (-(torch.exp(model_outputs)-1)**2/(torch.exp(model_outputs)*f + (1-f))**2).sum()
    return second_deriv.detach()


def uncertainties(f, model_outputs_all, w, cov):
    ai = calc_ai(f, model_outputs_all, w)
    second_deriv = calc_second_deriv(f, model_outputs_all, w)
    return torch.abs(1/second_deriv) + 1/second_deriv**2 * (ai.double()@cov.double()@ai.double()).float(), 1 + torch.abs(1/second_deriv) * (ai.double()@cov.double()@ai.double()).float()


for subnet_idx, num_subnets in enumerate(num_subnets_arr):
    print("Number of subnets is", num_subnets, flush=True)
    for training in range(num_trainings):
        print("Starting training run", training, flush=True)
        qs = torch.randn(num_samples) + .1
        ps = torch.randn(num_samples) - .1
        qs_train = qs[0:num_train_samples]
        qs_val = qs[num_train_samples:]
        ps_train = ps[0:num_train_samples]
        ps_val = ps[num_train_samples:]

        data_train = torch.concatenate((qs_train, ps_train)).detach()
        data_val = torch.concatenate((qs_val, ps_val)).detach()
        train_perm_key = torch.randperm(2*num_train_samples).detach()
        val_perm_key = torch.randperm(2*num_val_samples).detach()
        x_train = data_train[train_perm_key].unsqueeze(1).detach()
        x_val = data_val[val_perm_key].unsqueeze(1).detach()
        y_train = torch.concatenate((torch.ones(num_train_samples), torch.zeros(num_train_samples)))[train_perm_key].detach()
        y_val = torch.concatenate((torch.ones(num_val_samples), torch.zeros(num_val_samples)))[val_perm_key].detach()

        model = Model(num_subnets)
        model.to(device)
        train_losses, val_losses = train_func(model, 150, num_subnets)
        for run in range(num_runs):
            if run % 10 == 0:
                print("Starting run", run, flush=True)

            w00 = torch.ones(num_subnets, device=device)/num_subnets
            w0 = w00
            cov_mat = torch.zeros((num_subnets, num_subnets), device=device)
            qs_test_mix = np.random.randn(num_mixture_samples) + .1
            ps_test_mix = np.random.randn(num_mixture_samples) - .1
            for f_ind, f in enumerate([0.01, 0.02, 0.05, 0.1, 0.2, 0.5]):
                test_mix_mask = np.random.choice([True, False], p=[f, 1-f], size=num_mixture_samples)
                test_mix = ps_test_mix.copy()
                test_mix[test_mix_mask] = qs_test_mix[test_mix_mask]
                test_mix_tensor = torch.tensor(test_mix).unsqueeze(1).to(device).float()
                test_mix_outputs_all = model.submodel_all(test_mix_tensor)
                test_mix_outputs = test_mix_outputs_all@w0

                res_min = minimize_scalar(neg_maximum_likelihood_f_wrapper, bounds=(-0.3, 1.), args=(test_mix_outputs), method='bounded')
                var, div = uncertainties(res_min.x, test_mix_outputs_all, w0, cov_mat)
                min_nll = neg_maximum_likelihood_f(res_min.x, test_mix_outputs)
                nll_star = neg_maximum_likelihood_f(f, test_mix_outputs)
                z_score_arr[subnet_idx, training, run, f_ind] = ((-2*(min_nll - nll_star)).abs()/div).sqrt().detach()
                true_arr[subnet_idx, training, run, f_ind] = f
                pred_arr[subnet_idx, training, run, f_ind] = res_min.x
                unc_arr[subnet_idx, training, run, f_ind] = var.sqrt()
                print("True f is", f, flush=True)
                print("Pred f is", res_min.x, flush=True)
                print("Naive z score is", np.abs(f - res_min.x)/var.sqrt(), flush=True)
                print("Z score is", z_score_arr[subnet_idx, training, run, f_ind], flush=True)
                print("Unc on f is", var.sqrt(), flush=True)
torch.save(pred_arr, 'YOUR_DIR_HERE/baseline_ensemble_pred_arr.pt')
torch.save(true_arr, 'YOUR_DIR_HERE/baseline_ensemble_true_arr.pt')
torch.save(unc_arr, 'YOUR_DIR_HERE/baseline_ensemble_unc_arr.pt')
torch.save(z_score_arr, 'YOUR_DIR_HERE/baseline_ensemble_z_score_arr.pt')
