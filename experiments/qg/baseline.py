import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from energyflow.datasets import qg_jets
from scipy.optimize import minimize_scalar

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_overall_samples = 2000000
num_train_pool_samples = num_overall_samples//2
num_test_pool_samples = num_overall_samples - num_train_pool_samples
num_samples = 40000
num_mixture_samples = 10000
train_frac = 0.5
num_train_samples = int(train_frac * num_samples)
num_val_samples = num_train_samples

num_subnets_arr = [32]
train_batch_size = int(num_train_samples/2)
val_batch_size = 1*train_batch_size

num_runs = 300
num_trainings = 10
units = 32
latent_dim = 32

true_arr = torch.zeros((len(num_subnets_arr), num_trainings, num_runs, 6))
pred_arr = torch.zeros((len(num_subnets_arr), num_trainings, num_runs, 6))
bias_corr_pred_arr = torch.zeros((len(num_subnets_arr), num_trainings, num_runs, 6))
unc_arr = torch.zeros((len(num_subnets_arr), num_trainings, num_runs, 6))
z_score_arr = torch.zeros((len(num_subnets_arr), num_trainings, num_runs, 6))
z_score_bc_arr = torch.zeros((len(num_subnets_arr), num_trainings, num_runs, 6))


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


def bias_correction_f(f_biased, w0, cov_mat, GSvar, model_outputs_all):
    """
    Implement higher order asymptotic bias correction in f
    Assumes model_outputs_all is shape [N, shape[w0]]
    """
    N = model_outputs_all.shape[0]
    M = model_outputs_all.shape[1]
    r = torch.exp(model_outputs_all@w0)
    fs = (r-1)/(f_biased*r + (1-f_biased))
    fs_prime = -fs**2
    fs_double_prime = (2*fs**3).mean()
    fs_i = torch.zeros(M)
    fs_prime_i = torch.zeros(M)
    fs_ij = torch.zeros((M, M))
    for i in range(M):
        fs_i[i] = (model_outputs_all[:, i]*r/(f_biased*r + (1-f_biased))).mean()
        fs_prime_i[i] = (-2*model_outputs_all[:, i]*r*(r-1)/(f_biased*r + (1-f_biased))**3).mean()
        for j in range(M):
            fs_ij[i, j] = -(model_outputs_all[:, i]*model_outputs_all[:, j]*r*(f_biased*r + (f_biased - 1))/(f_biased*r + (1-f_biased))**3).mean()
    fs_prime_mean = fs_prime.mean()
    bias_est = (fs*fs_prime).mean()/N/fs_prime_mean**2 - 1/2/fs_prime_mean*(fs_double_prime*GSvar - fs_i@cov_mat@fs_prime_i/fs_prime_mean + torch.einsum('ij,ij->', cov_mat, fs_ij))
    return bias_est


class TimeDistributed(nn.Module):
    def __init__(self, layer):
        super(TimeDistributed, self).__init__()
        self.layer = layer

    def forward(self, x):
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.layer(x_reshape)
        y_reshape = y.contiguous().view(x.size(0), -1, y.size(-1))
        return y_reshape


def model_with_weights_scaled(submodels, dataloader, w):
    with torch.no_grad():
        dataset_length = len(dataloader.dataset)
        h_outs = torch.zeros(dataset_length, device=device)
        end_idx = 0
        for data in dataloader:
            start_idx = end_idx
            end_idx += len(data[0])
            h_outs[start_idx:end_idx] = submodels.submodel_all(data[0].to(device))@w
        return h_outs.detach()


def submodels_on_dataloader(submodels, dataloader, w):
    with torch.no_grad():
        dataset_length = len(dataloader.dataset)
        h_outs = torch.zeros(dataset_length, len(w), device=device)
        end_idx = 0
        for data in dataloader:
            start_idx = end_idx
            end_idx += len(data[0])
            h_outs[start_idx:end_idx, :] = submodels.submodel_all(data[0].to(device))
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
        self.phi1 = TimeDistributed(nn.Linear(2, units))
        self.phi2 = TimeDistributed(nn.Linear(units, units))
        self.phi3 = TimeDistributed(nn.Linear(units, latent_dim))
        self.f1 = nn.Linear(latent_dim, units)
        self.f2 = nn.Linear(units, units)
        self.f3 = nn.Linear(units, units)
        self.f4 = nn.Linear(units, 1)

        self.rand_phi1_list = nn.ModuleList([TimeDistributed(nn.Linear(2, units)) for i in range(self.num_subnets)])
        self.rand_phi2_list = nn.ModuleList([TimeDistributed(nn.Linear(units, units)) for i in range(self.num_subnets)])
        self.rand_phi3_list = nn.ModuleList([TimeDistributed(nn.Linear(units, latent_dim)) for i in range(self.num_subnets)])

        self.rand_f1_list = nn.ModuleList([nn.Linear(latent_dim, units) for i in range(self.num_subnets)])
        self.rand_f2_list = nn.ModuleList([nn.Linear(units, units) for i in range(self.num_subnets)])
        self.rand_f3_list = nn.ModuleList([nn.Linear(units, units) for i in range(self.num_subnets)])
        self.rand_f4_list = nn.ModuleList([nn.Linear(units, 1) for i in range(self.num_subnets)])

    def forward(self, x):
        zs = x[:, :, 0]
        xs = x[:, :, 1:3] * 1000
        x = F.leaky_relu(self.phi1(xs), negative_slope=0.2)
        x = F.leaky_relu(self.phi2(x), negative_slope=0.2)
        x = F.leaky_relu(self.phi3(x), negative_slope=0.2)
        x = torch.sum(x*zs.unsqueeze(-1), axis=1)
        x = F.leaky_relu(self.f1(x), negative_slope=0.2)
        x = F.leaky_relu(self.f2(x), negative_slope=0.2)
        x = F.leaky_relu(self.f3(x), negative_slope=0.2)
        x = self.f4(x)
        return x

    def submodel_all(self, x):
        zs = x[:, :, 0]
        xs = x[:, :, 1:3] * 1000
        x1 = [F.leaky_relu(self.rand_phi1_list[i](xs), negative_slope=0.2) for i in range(self.num_subnets)]
        x1 = [F.leaky_relu(self.rand_phi2_list[i](x1[i]), negative_slope=0.2) for i in range(self.num_subnets)]
        x1 = [F.leaky_relu(self.rand_phi3_list[i](x1[i]), negative_slope=0.2) for i in range(self.num_subnets)]

        x1 = [torch.sum(x1[i] * zs.unsqueeze(-1), axis=1) for i in range(self.num_subnets)]
        x1 = [F.leaky_relu(self.rand_f1_list[i](x1[i]), negative_slope=0.2) for i in range(self.num_subnets)]
        x1 = [F.leaky_relu(self.rand_f2_list[i](x1[i]), negative_slope=0.2) for i in range(self.num_subnets)]
        x1 = [F.leaky_relu(self.rand_f3_list[i](x1[i]), negative_slope=0.2) for i in range(self.num_subnets)]
        x1 = torch.cat([self.rand_f4_list[i](x1[i]) for i in range(self.num_subnets)], axis=1)
        return x1


def train_func(model_to_train, epochs, num_subnets):
    train_losses = []
    val_losses = []
    for start_idx in range(num_subnets):
        opt = optim.Adam(model_to_train.parameters(), lr=2e-3)
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
        for param in model_to_train.rand_phi1_list[start_idx].parameters():
            param.requires_grad = True
        for param in model_to_train.rand_phi2_list[start_idx].parameters():
            param.requires_grad = True
        for param in model_to_train.rand_phi3_list[start_idx].parameters():
            param.requires_grad = True
        for param in model_to_train.rand_f1_list[start_idx].parameters():
            param.requires_grad = True
        for param in model_to_train.rand_f2_list[start_idx].parameters():
            param.requires_grad = True
        for param in model_to_train.rand_f3_list[start_idx].parameters():
            param.requires_grad = True
        for param in model_to_train.rand_f4_list[start_idx].parameters():
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
                best_epoch = epoch
            print(f"Epoch {epoch+1} train loss: {running_loss/batches}, val loss: {val_loss/val_batches}", flush=True)
            if best_epoch == epoch-10:
                print("Early stopping", flush=True)
                break
        model_to_train.load_state_dict(best_model)
    return train_losses, val_losses


for subnet_idx, num_subnets in enumerate(num_subnets_arr):
    print("Number of subnets is", num_subnets, flush=True)
    X, y = qg_jets.load(num_overall_samples)
    p_overall = np.arange(num_overall_samples)
    np.random.shuffle(p_overall)
    X = X[p_overall]
    y = y[p_overall]
    for x in X:
        mask = x[:, 0] > 0
        yphi_avg = np.average(x[mask, 1:3], weights=x[mask, 0], axis=0)
        x[mask, 1:3] -= yphi_avg
        x[mask, 0] /= x[:, 0].sum()
    X_train_pool = X[:num_train_pool_samples]
    y_train_pool = y[:num_train_pool_samples]
    X_test_pool = X[num_train_pool_samples:]
    y_test_pool = y[num_train_pool_samples:]
    for training in range(num_trainings):
        print("Starting training run", training, flush=True)
        p = np.arange(num_train_pool_samples)
        np.random.shuffle(p)
        X_samp = X_train_pool[p, :, :-1]
        y_samp = y_train_pool[p]
        x_train = torch.tensor(X_samp[:num_train_samples]).float().to(device)
        x_val = torch.tensor(X_samp[num_train_samples:num_samples]).float().to(device)
        y_train = torch.tensor(y_samp[:num_train_samples]).float().to(device)
        y_val = torch.tensor(y_samp[num_train_samples:num_samples]).float().to(device)
        print("Q tot train, Q tot val:", (y_train.sum(), y_val.sum()), flush=True)

        model = Model(num_subnets)
        model.to(device)
        train_losses, val_losses = train_func(model, 10000, num_subnets)
        for run in range(num_runs):
            if run % 10 == 0:
                print("Starting run", run, flush=True)
            p = np.arange(num_overall_samples)
            np.random.shuffle(p)
            X_samp = X[p, :, :-1]
            y_samp = y[p]
            x_train = torch.tensor(X_samp[:num_train_samples]).float().to(device)
            x_val = torch.tensor(X_samp[num_train_samples:num_samples]).float().to(device)
            y_train = torch.tensor(y_samp[:num_train_samples]).float().to(device)
            y_val = torch.tensor(y_samp[num_train_samples:num_samples]).float().to(device)

            p_test = np.arange(num_test_pool_samples)
            np.random.shuffle(p_test)
            X_samp_test = X_test_pool[p_test, :, :-1]
            y_samp_test = y_test_pool[p_test]
            x_test = torch.tensor(X_samp_test).float().to(device)
            y_test = torch.tensor(y_samp_test).float().to(device)
            x_test_quark = x_test[y_test == 1][:num_mixture_samples]
            x_test_gluon = x_test[y_test == 0][:num_mixture_samples]

            w00 = torch.ones(num_subnets, device=device)/num_subnets
            w0 = w00
            cov_mat = torch.zeros((num_subnets, num_subnets), device=device)
            for f_ind, f in enumerate([0.01, 0.02, 0.05, 0.1, 0.2, 0.5]):
                test_mix_mask = np.random.choice([True, False], p=[f, 1-f], size=num_mixture_samples)
                test_mix = x_test_gluon.clone()
                test_mix[test_mix_mask] = x_test_quark[test_mix_mask]
                test_mix_tensor = test_mix.to(device).float()
                testset = torch.utils.data.TensorDataset(test_mix_tensor)
                testloader = torch.utils.data.DataLoader(testset, batch_size=train_batch_size, shuffle=False)

                test_mix_outputs_all = submodels_on_dataloader(model, testloader, w0)
                test_mix_outputs = test_mix_outputs_all@w0

                res_min = minimize_scalar(neg_maximum_likelihood_f_wrapper, bounds=(-0.3, 1.), args=(test_mix_outputs), method='bounded')
                fhat = res_min.x
                var, div = uncertainties(res_min.x, test_mix_outputs_all, w0, cov_mat)
                min_nll = neg_maximum_likelihood_f(fhat, test_mix_outputs)
                nll_star = neg_maximum_likelihood_f(f, test_mix_outputs)
                z_score_arr[subnet_idx, training, run, f_ind] = ((-2*(min_nll - nll_star)).abs()/div).sqrt().detach()
                bias_est = bias_correction_f(res_min.x, w0, cov_mat.cpu(), var, test_mix_outputs_all).detach()
                true_arr[subnet_idx, training, run, f_ind] = f
                pred_arr[subnet_idx, training, run, f_ind] = fhat
                bias_corr_pred_arr[subnet_idx, training, run, f_ind] = fhat - bias_est
                min_nll_bc = neg_maximum_likelihood_f(fhat - bias_est, test_mix_outputs)
                z_score_bc_arr[subnet_idx, training, run, f_ind] = ((-2*(min_nll_bc - nll_star)).abs()/div).sqrt().detach()
                unc_arr[subnet_idx, training, run, f_ind] = var.sqrt()
                print("Predicted f is", fhat, "Bias corrected predicted f is", fhat - bias_est, "True f is", f, "Uncertainty is", var.sqrt(), "Chi2 z score is", z_score_arr[subnet_idx, training, run, f_ind], flush=True)

torch.save(pred_arr, 'YOUR_DIR_HERE/baseline_pred_arr.pt')
torch.save(bias_corr_pred_arr, 'YOUR_DIR_HERE/baseline_bias_corr_pred_arr.pt')
torch.save(true_arr, 'YOUR_DIR_HERE/baseline_true_arr.pt')
torch.save(unc_arr, 'YOUR_DIR_HERE/baseline_unc_arr.pt')
torch.save(z_score_arr, 'YOUR_DIR_HERE/baseline_z_score_arr.pt')
torch.save(z_score_bc_arr, 'YOUR_DIR_HERE/baseline_z_score_bc_arr.pt')
