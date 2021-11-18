#%%
import matplotlib
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torchvision.transforms as transforms
import PIL
import torch.nn.functional as F
import core.amortised_models as amortised_models # type: ignore
import core.nonamortised_models as nonamortised_models # type: ignore
import core.utils as utils # type: ignore
from torch.distributions import Independent, Normal, MultivariateNormal
import hydra
from omegaconf import OmegaConf
import time


torch.set_num_threads(2)

def save_np(name, x):
    np.save(name, x)

xDIM = 32
yDIM = 3*32*32


@hydra.main(config_path='conf', config_name="seqVAE_amortised")
def main(cfg):
    device = cfg.device
    utils.save_git_hash(hydra.utils.get_original_cwd())
    root_dir = hydra.utils.get_original_cwd()

    class MyData(torch.utils.data.Dataset):
        def __init__(self, img_dir):
            transform = torchvision.transforms.ToTensor()
            self.offset = 55
            self.data = torch.zeros(4115-self.offset, 3, 32, 32)
            for i in tqdm(range(4115-self.offset)):
                img_path = os.path.join(img_dir, 'image_{}.png'.format(i+self.offset))
                pil_image = PIL.Image.open(img_path)
                image = transform(pil_image)
                self.data[i, :, :, :] = image

            self.data = self.data.to(device)

            self.data_mean = torch.mean(self.data, dim=0).to(device)
            self.data_std = torch.std(self.data, dim=0).to(device)

            self.data = (self.data - self.data_mean) / self.data_std


        def __len__(self):
            return 4115-self.offset

        def __getitem__(self, index):
            img = self.data[index]
            return img

    def unnormalize(x, dataset):
        return dataset.data_mean + x * dataset.data_std

    def imshow_ops(x):
        return unnormalize(x, dataset).cpu().detach().transpose(0,1).transpose(1,2)

    dataset = MyData(os.path.join(root_dir, cfg.dataset_path))
    flat_data = dataset.data.flatten(1)

    class ResBlock(nn.Module):
        def __init__(self, input_channels, channel):
            super().__init__()

            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, channel, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, input_channels, 1),
            )

        def forward(self, x):
            out = self.conv(x)
            out += x
            out = F.relu(out)
            return out

    class DeepMindDecoder(nn.Module):

        def __init__(self, n_init=32, n_hid=64, output_channels=3):
            super().__init__()

            self.net = nn.Sequential(
                nn.Conv2d(n_init, 2*n_hid, 3, padding=1),
                nn.ReLU(),
                ResBlock(2*n_hid, 2*n_hid//4),
                ResBlock(2*n_hid, 2*n_hid//4),
                nn.ConvTranspose2d(2*n_hid, n_hid, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(n_hid, output_channels, 4, stride=2, padding=1),
            )

        def forward(self, x):
            return self.net(x)

    class DeepMindEncoder(nn.Module):

        def __init__(self, input_channels=3, n_hid=64):
            super().__init__()

            self.net = nn.Sequential(
                nn.Conv2d(input_channels, n_hid, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_hid, 2*n_hid, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(2*n_hid, 2*n_hid, 3, padding=1),
                nn.ReLU(),
                ResBlock(2*n_hid, 2*n_hid//4),
                ResBlock(2*n_hid, 2*n_hid//4),
            )

            self.output_channels = 2 * n_hid
            self.output_stide = 4

        def forward(self, x):
            return self.net(x)

    class Decoder(nn.Module):
        def __init__(self, z_dim):
            super().__init__()
            self.z_dim = z_dim
            self.linear = nn.Linear(self.z_dim, 32*8*8)
            self.dmdecoder = DeepMindDecoder(output_channels=3)

        def forward(self, z):
            x1 = self.linear(z)
            x2 = x1.view(-1, 32, 8, 8)
            out = self.dmdecoder(x2)
            out = out.view(-1, yDIM)
            return out

    class Encoder(nn.Module):
        def __init__(self, z_dim):
            super().__init__()
            self.z_dim = z_dim
            self.nn = nn.Sequential(
                DeepMindEncoder(input_channels=3),
                nn.Flatten(),
                nn.Linear(128*8*8, 2*self.z_dim)
            )

        def forward(self, x):
            return self.nn(x)

    decoder = Decoder(xDIM).to(device)
    decoder.load_state_dict(torch.load(
        os.path.join(root_dir, cfg.decoder_path),
        map_location=torch.device(device)))

    class ResMLP(nn.Module):
        def __init__(self, in_dim, h, out_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Linear(h, out_dim)
            )
            self.register_parameter('scale', nn.Parameter(torch.tensor(cfg.init_res_scale)))
        def forward(self, x):
            out = self.net(x)
            out = out * self.scale
            return out + x

    class Transition(nn.Module):
        def __init__(self, xdim, num_layers, h):
            super().__init__()
            nnlist = [ResMLP(xdim, h, xdim)]
            for i in range(num_layers-1):
                nnlist += [ResMLP(xdim, h, xdim)]
            self.net = nn.Sequential(*nnlist)

        def forward(self, x):
            return self.net(x)

    transition = Transition(32, 4, 32).to(device)
    theta_dim = sum([p.numel() for p in transition.parameters()])
    print("theta dim", theta_dim)

    U = cfg.U_scalar * torch.eye(xDIM).to(device)
    V = cfg.V_scalar * torch.eye(yDIM).to(device)
    mean_0 = torch.zeros(xDIM).to(device)
    cov_0 = torch.eye(xDIM).to(device)

    class F_Module(nn.Module):
        def __init__(self, transition):
            super().__init__()
            self.transition = transition
            self.F_mean_fn = lambda x, t: self.transition(x)
            self.F_cov_fn = lambda x, t: U

        def forward(self, x, t=None):
            return Independent(Normal(self.F_mean_fn(x, t),
                torch.sqrt(torch.diag(U))), 1)

    class G_Module(nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder
            self.G_mean_fn = lambda x, t: self.decoder(x).flatten(1)

        def forward(self, x, t=None):
            return Independent(Normal(self.G_mean_fn(x, t),
                torch.sqrt(torch.diag(V))), 1)

    class p_0_dist_module(nn.Module):
        def __init__(self):
            super().__init__()
            self.mean_0 = mean_0
            self.cov_0 = cov_0

        def forward(self):
            return Independent(Normal(mean_0, torch.sqrt(torch.diag(self.cov_0))), 1)

    def net_constructor(net_dims):
        modules = []
        for in_dim, out_dim in zip(net_dims[:-2], net_dims[1:-1]):
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(net_dims[-2], net_dims[-1]))
        return nn.Sequential(*modules).to(device)


    class Normal_Net(nn.Module):
        def __init__(self, mlp_module):
            super().__init__()
            self.mlp_module = mlp_module

        def forward(self, x):
            x = self.mlp_module(x)
            mu, logsigma = torch.chunk(x, 2, dim=-1)
            sig = torch.exp(logsigma)
            return mu, sig

    def S_net_constructor():
        net = net_constructor([32, 256, 1024, theta_dim+1])
        return nonamortised_models.NN_Func_Polar(net, 32, theta_dim, 0.9).to(device)

    def KRR_constructor():
        return nonamortised_models.KernelRidgeRegressor(nonamortised_models.MaternKernel(
            sigma=2.0, lam=cfg.KRR_lambda, train_sigma=True, train_lam=False)).to(device)

    class ConvRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.RNN(32, cfg.q_rnn_hidden_dim, cfg.q_rnn_num_layers)
            self.conv = nn.Sequential(
                nn.Conv2d(3, 16, 4),
                nn.MaxPool2d(3),
                nn.ReLU(),
                nn.Conv2d(16, 32, 4),
                nn.MaxPool2d(3),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128, 32)
            )
            self.hidden_size = self.rnn.hidden_size

        def forward(self, y, h0=None):
            # Expected y shape (L, N, yDIM) with N=1
            assert len(y.shape) == 3
            assert y.shape[1] == 1
            assert y.shape[2] == yDIM
            if h0 is None:
                return self.rnn(
                    self.conv(y.view(y.shape[0], 3, 32, 32))\
                    .view(y.shape[0], 1, -1))
            else:
                return self.rnn(
                    self.conv(y.view(y.shape[0], 3, 32, 32))\
                    .view(y.shape[0], 1, -1), h0)


    mseloss = nn.MSELoss()
    cosloss = nn.CosineSimilarity()
    def dir_mag_loss(pred, true):
        pred_norm = torch.norm(pred, dim=1).unsqueeze(-1) + 1e-3
        true_norm = torch.norm(true, dim=1).unsqueeze(-1) + 1e-3
        dir_loss = -torch.mean(cosloss(pred, true))
        mag_loss = mseloss( torch.log(pred_norm), torch.log(true_norm))
        return dir_loss * cfg.dir_mag_loss_balance + \
            mag_loss * (1-cfg.dir_mag_loss_balance),\
            dir_loss, mag_loss

    F_fn = F_Module(transition).to(device)
    G_fn = G_Module(decoder).to(device)
    p_0_dist = p_0_dist_module().to(device)
    q_rnn_constructor = lambda: ConvRNN().to(device)






    q_t_net_constructor = lambda: Normal_Net(net_constructor([cfg.q_rnn_hidden_dim] + \
        list(cfg.q_hidden_dims) + [2 * xDIM])).to(device)
    cond_q_t_net_constructor = lambda: Normal_Net(net_constructor([xDIM + cfg.q_rnn_hidden_dim] + \
        list(cfg.q_hidden_dims) + [2 * xDIM])).to(device)

    class AggregateFuncs(nn.Module):
        def __init__(self):
            super().__init__()
            self.S = S_net_constructor()
            self.T = KRR_constructor()

        def update_normalization(self, inputs, S_outputs, T_outputs):
            self.S.update_normalization(inputs, S_outputs)

        def forward(self, x, t):
            # print("AggFunc forward t {}".format(t))
            if t > 0:
                net_outputs = [self.S(x), self.T(x)[0]]
            else:
                net_outputs = [
                    torch.zeros(x.shape[0], theta_dim).to(device),
                    self.T(x)[0]
                ]

            return net_outputs[0], net_outputs[1]

        def fit(self, train_input, *train_outputs):
            # train_outputs should be S targets then T targets
            # this function will only fit T
            assert len(train_outputs) == 2
            self.T.fit(train_input, train_outputs[1])

        @property
        def kernel(self):
            return self.T.kernel

    def aggregate_net_constructor():
        return AggregateFuncs()

    model = amortised_models.SeparateTimeKernelAmortisedModel(
        device, xDIM, yDIM, q_rnn_constructor, q_t_net_constructor,
        cond_q_t_net_constructor, F_fn, G_fn,
        p_0_dist, cfg.rnn_window_size, aggregate_net_constructor,
        cfg.approx_with_filter,
        'ST',
        'F'
    )

    def get_g_prob(T):
        model.rnn_forward(flat_data)
        x_samples, _ = model.sample_joint_q_t(32, 1 if T > 0 else 0)
        if len(x_samples) > 1:
            tm1_prob = model.G_fn(x_samples[0]).log_prob(flat_data[T-1, :]).mean().item()
            t_prob = model.G_fn(x_samples[1]).log_prob(flat_data[T, :]).mean().item()
        else:
            tm1_prob = 0
            t_prob = model.G_fn(x_samples[0]).log_prob(flat_data[T, :]).mean().item()

        return [tm1_prob, t_prob]




    theta_optim = torch.optim.Adam(model.F_fn.parameters(), lr=cfg.theta_lr)
    if not cfg.actually_make_nonamortized:
        phi_optim = torch.optim.Adam(model.get_phi_params(), lr=cfg.phi_lr)

    os.mkdir('transitions')

    theta_scales = []
    g_probs_all = []
    S_losses_all = []
    q_t_means = []
    q_t_stds = []
    q_tm1_means = []
    q_tm1_stds = []
    times = []

    for T in range(cfg.num_iters):
        start_time = time.time()
        print("T ", T)

        model.advance_timestep(flat_data)




        # ------------- phi optimization ---------------
        g_probs = []
        inner_q_t_means = []
        inner_q_t_stds = []
        inner_q_tm1_means = []
        inner_q_tm1_stds = []

        for i in range(cfg.phi_iters):
            # Update phi


            phi_optim.zero_grad()
            theta_optim.zero_grad()


            loss = model.populate_grads(flat_data, cfg.phi_minibatch_size)
            phi_optim.step()

            g_probs.append(get_g_prob(T))

            _, q_stats = model.sample_joint_q_t(1, 1)
            if T == 0:
                inner_q_t_means.append(q_stats[0][0].detach().cpu().numpy())
                inner_q_t_stds.append(q_stats[0][1].detach().cpu().numpy())
                inner_q_tm1_means.append(np.zeros(xDIM))
                inner_q_tm1_stds.append(np.ones(xDIM))
            else:
                inner_q_t_means.append(q_stats[1][0].detach().cpu().numpy())
                inner_q_t_stds.append(q_stats[1][1].detach().cpu().numpy())
                inner_q_tm1_means.append(q_stats[0][0][0, :].detach().cpu().numpy())
                inner_q_tm1_stds.append(q_stats[0][1][0, :].detach().cpu().numpy())


            # only update theta once
            if i == cfg.phi_iters - 1:
                theta_optim.step()
                theta_scales.append(model.F_fn.transition.net[0].scale.item())

        g_probs_all.append(np.array(g_probs))
        q_t_means.append(np.array(inner_q_t_means))
        q_t_stds.append(np.array(inner_q_t_stds))
        q_tm1_means.append(np.array(inner_q_tm1_means))
        q_tm1_stds.append(np.array(inner_q_tm1_stds))


        # ----------- Train function estimators ----------------
        train_x, S_targets, T_targets = model.generate_training_dataset(
            flat_data,
            cfg.training_dataset_size
        )
        # Fit kernel T
        model.update_func_t(train_x, S_targets, T_targets)

        # Fit net S

        # only fit for T >= 1 since T==0 doesn't depend on transition
        net_optim = torch.optim.Adam(model.approx_func_t.S.parameters(),
            lr=cfg.net_lr)
        S_net_losses = []
        for i in range(cfg.net_update_steps):
            net_optim.zero_grad()
            idx = torch.randint(0, cfg.training_dataset_size,
                (cfg.net_minibatch_size,))
            preds = model.approx_func_t.S(train_x[idx, :])
            loss, dir_loss, mag_loss = dir_mag_loss(preds, S_targets[idx, :])
            loss.backward()
            net_optim.step()
            S_net_losses.append([dir_loss.item(), mag_loss.item()])
        S_losses_all.append(S_net_losses)

        end_time = time.time() - start_time
        times.append(end_time)

        # ------------- Save data ------------
        if (T % cfg.save_interval == 0) or (T == (cfg.num_iters - 1)):
            torch.save(transition.state_dict(), 'transitions/transition_{}.pt'.format(T))
            np.save('theta_scales.npy', np.array(theta_scales))
            np.save('g_probs_all.npy', np.array(g_probs_all))
            np.save('q_t_means.npy', np.array(q_t_means))
            np.save('q_t_stds.npy', np.array(q_t_stds))
            np.save('q_tm1_means.npy', np.array(q_tm1_means))
            np.save('q_tm1_stds.npy', np.array(q_tm1_stds))
            np.save('S_losses_all.npy', np.array(S_losses_all))
            np.save('times.npy', np.array(times))



if __name__ == "__main__":
    main()
