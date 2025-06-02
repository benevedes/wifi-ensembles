import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import minimize
import numpy as np
from energyflow.datasets import qg_jets

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_overall_samples = 2000000
num_train_pool_samples = num_overall_samples//2
num_test_pool_samples = num_overall_samples - num_train_pool_samples
num_samples = 40000
num_mixture_samples = 10000
train_frac = 0.5
num_train_samples = int(train_frac * num_samples)
num_val_samples = num_train_samples

num_subnets = 32
train_batch_size = int(num_train_samples/2)
val_batch_size = 1*train_batch_size

units = 32
latent_dim = 32


def neg_maximum_likelihood_f(f, model_outputs):
    return -torch.log(torch.exp(model_outputs)*f + (1-f)).sum()


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
        fs_i[i] = (model_outputs_all[:,i]*r/(f_biased*r + (1-f_biased))).mean()
        fs_prime_i[i] = (-2*model_outputs_all[:,i]*r*(r-1)/(f_biased*r + (1-f_biased))**3).mean()
        for j in range(M):
            fs_ij[i,j] = -(model_outputs_all[:,i]*model_outputs_all[:,j]*r*(f_biased*r + (f_biased - 1))/(f_biased*r + (1-f_biased))**3).mean()
    fs_prime_mean = fs_prime.mean()
    bias_est = (fs*fs_prime).mean()/N/fs_prime_mean**2 - 1/2/fs_prime_mean*(fs_double_prime*GSvar - fs_i@cov_mat@fs_prime_i/fs_prime_mean + torch.einsum('ij,ij->',cov_mat,fs_ij))
    return bias_est


class TimeDistributed(nn.Module):
    def __init__(self, layer):
        super(TimeDistributed, self).__init__()
        self.layer = layer

    def forward(self, x):
        x_reshape = x.contiguous().view(-1, x.size(-1)) #reshape to (batch_size * timesteps, input_size per time step)
        y = self.layer(x_reshape)
        y_reshape = y.contiguous().view(x.size(0), -1, y.size(-1)) #reshape back to (batch_size, timesteps, output_size per time step)
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


def mlc_min(w, submodels, qloader, ploader):
    with torch.no_grad():
        w = torch.tensor(w).to(device).float()
        q_out = model_with_weights_scaled(submodels, qloader, w)
        p_out = model_with_weights_scaled(submodels, ploader, w)
        mlc1 = -(q_out.mean() - (torch.exp(p_out) - 1).mean())
        mlc2 = -(-p_out.mean() - (torch.exp(-q_out) - 1).mean())
        return (mlc1 + mlc2).cpu().numpy()


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


def mlc_grad(w, submodels, qloader, ploader):
    with torch.no_grad():
        w = torch.tensor(w).to(device).float()
        q_out_all = submodels_on_dataloader(submodels, qloader, w)
        p_out_all = submodels_on_dataloader(submodels, ploader, w)
        return (-q_out_all.mean(axis=0) + (p_out_all*torch.exp(p_out_all@w).unsqueeze(axis=1)).mean(axis=0) + p_out_all.mean(axis=0) - (q_out_all*torch.exp(-q_out_all@w).unsqueeze(axis=1)).mean(axis=0)).cpu().numpy()


def mlc_hess(w, submodels, qloader, ploader):
    with torch.no_grad():
        w = torch.tensor(w).to(device).float()
        q_out_all = submodels_on_dataloader(submodels, qloader, w)
        p_out_all = submodels_on_dataloader(submodels, ploader, w)
        q_out_mat = torch.einsum('ai,aj->aij', q_out_all, q_out_all)
        p_out_mat = torch.einsum('ai,aj->aij', p_out_all, p_out_all)
        return ((p_out_mat*torch.exp(p_out_all@w).unsqueeze(axis=1).unsqueeze(axis=2)).mean(axis=0) + (q_out_mat*torch.exp(-q_out_all@w).unsqueeze(axis=1).unsqueeze(axis=2)).mean(axis=0)).cpu().numpy()


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
        x1 = torch.cat([x1, torch.ones(x1.shape[0], device=device).unsqueeze(1)], axis=1)
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
        print(f"Training basis functions {start_idx}", flush=True)
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

qset = torch.utils.data.TensorDataset(x_train[y_train == 1])
qloader = torch.utils.data.DataLoader(qset, batch_size=train_batch_size, shuffle=False)
qset_val = torch.utils.data.TensorDataset(x_val[y_val == 1])
qloader_val = torch.utils.data.DataLoader(qset_val, batch_size=val_batch_size, shuffle=False)

pset = torch.utils.data.TensorDataset(x_train[y_train == 0])
ploader = torch.utils.data.DataLoader(pset, batch_size=train_batch_size, shuffle=False)
pset_val = torch.utils.data.TensorDataset(x_val[y_val == 0])
ploader_val = torch.utils.data.DataLoader(pset_val, batch_size=val_batch_size, shuffle=False)
w00 = torch.ones(num_subnets+1, device=device)/num_subnets
w00[-1] = 0.
res_root = minimize(mlc_min, x0=w00.cpu().numpy(), args=(model, qloader, ploader), method='trust-exact', jac=mlc_grad, hess=mlc_hess)
w0 = torch.tensor(res_root.x, device=device).float()
model_q_points = torch.zeros((num_subnets+1, qloader.dataset.tensors[0].shape[0]))
model_p_points = torch.zeros((num_subnets+1, ploader.dataset.tensors[0].shape[0]))
q_wgt_points = model_with_weights_scaled(model, qloader, w0).detach().cpu()
p_wgt_points = model_with_weights_scaled(model, ploader, w0).detach().cpu()

end_idx = 0
for data in ploader:
    model_prod_points_batch = model.submodel_all(data[0].to(device)).detach()
    end_idx += len(model_prod_points_batch)
    start_idx = end_idx - len(model_prod_points_batch)
    model_p_points[:, start_idx:end_idx] = model_prod_points_batch.cpu().T

end_idx = 0
for data in qloader:
    model_joint_points_batch = model.submodel_all(data[0].to(device)).detach()
    end_idx += len(model_joint_points_batch)
    start_idx = end_idx - len(model_joint_points_batch)
    model_q_points[:, start_idx:end_idx] = model_joint_points_batch.cpu().T

dijq = torch.zeros((num_subnets+1, num_subnets+1))
dijp = torch.zeros((num_subnets+1, num_subnets+1))
cijq = torch.zeros((num_subnets+1, num_subnets+1))
cijp = torch.zeros((num_subnets+1, num_subnets+1))

for i in range(num_subnets+1):
    for j in range(num_subnets+1):
        hi_q = model_q_points[i, :]*(1 + torch.exp(-q_wgt_points))
        hj_q = model_q_points[j, :]*(1 + torch.exp(-q_wgt_points))
        dijq[i, j] = (hi_q*hj_q).mean() - (hi_q).mean()*(hj_q).mean()

        hi_p = model_p_points[i, :]*(1 + torch.exp(p_wgt_points))
        hj_p = model_p_points[j, :]*(1 + torch.exp(p_wgt_points))
        dijp[i, j] = (hi_p*hj_p).mean() - (hi_p).mean()*(hj_p).mean()
        cijp[i, j] = -(model_p_points[i, :]*model_p_points[j, :]*torch.exp(p_wgt_points)).mean()
        cijq[i, j] = -(model_q_points[i, :]*model_q_points[j, :]*torch.exp(-q_wgt_points)).mean()
cij = cijq + cijp
dij = dijq + dijp
cov_mat = torch.linalg.solve(cij.double(), torch.linalg.solve(cij.double(), dij.double()).T).float()/end_idx
cov_mat = cov_mat.to(device)

torch.save(x_test_gluon.to(device), 'YOUR_DIR_HERE/gluon_inputs.pt')
torch.save(x_test_quark.to(device), 'YOUR_DIR_HERE/quark_inputs.pt')

torch.save(model.submodel_all(x_test_gluon.to(device)), 'YOUR_DIR_HERE/gluon_outputs.pt')
torch.save(model.submodel_all(x_test_quark.to(device)), 'YOUR_DIR_HERE/quark_outputs.pt')

torch.save(w0, 'YOUR_DIR_HERE/w0.pt')
torch.save(cov_mat, 'YOUR_DIR_HERE/cov_mat.pt')
