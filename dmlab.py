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
import core.nonamortised_models as models
import core.utils as utils
from torch.distributions import Independent, Normal, MultivariateNormal
import hydra
from omegaconf import OmegaConf
import time

def save_np(name, x):
    np.save(name, x)

xDIM = 32
yDIM = 3*32*32

@hydra.main(config_path='conf', config_name="dmlab")
def main(cfg):
    device = cfg.device
    utils.save_git_hash(hydra.utils.get_original_cwd())

    theta_window_size = cfg.theta_window_size
    phi_window_size = cfg.phi_window_size

    class MyData(torch.utils.data.Dataset):
        def __init__(self, img_dir):
            transform = torchvision.transforms.ToTensor()
            self.data = torch.zeros(4115, 3, 32, 32)
            for i in tqdm(range(4115)):
                img_path = os.path.join(img_dir, 'image_{}.png'.format(i))
                pil_image = PIL.Image.open(img_path)
                image = transform(pil_image)
                self.data[i, :, :, :] = image

            self.data = self.data.to(device)

            self.data_mean = torch.mean(self.data, dim=0).to(device)
            self.data_std = torch.std(self.data, dim=0).to(device)

            self.data = (self.data - self.data_mean) / self.data_std


        def __len__(self):
            return 4115

        def __getitem__(self, index):
            img = self.data[index]
            return img

    def unnormalize(x, dataset):
        return dataset.data_mean + x * dataset.data_std

    def imshow_ops(x):
        return unnormalize(x, dataset).cpu().detach().transpose(0,1).transpose(1,2)

    dataset = MyData(os.path.join(hydra.utils.get_original_cwd(), cfg.dataset_path))
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
        os.path.join(hydra.utils.get_original_cwd(), cfg.decoder_path),
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

    F_fn = F_Module(transition).to(device)
    G_fn = G_Module(decoder).to(device)
    p_0_dist = p_0_dist_module().to(device)

    def cond_q_mean_net_constructor():
        net = nn.Sequential(
            nn.Linear(xDIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, xDIM)
        ).to(device)
        return net

    sigma = 2.0
    lam = cfg.KRR_lambda
    train_sigma = True
    train_lam = False

    def KRR_constructor():
        return models.KernelRidgeRegressor(models.MaternKernel(
            sigma=sigma, lam=lam, train_sigma=train_sigma, train_lam=train_lam)).to(device)

    time_store_size = max(phi_window_size, theta_window_size)+1

    phi_model = models.Vx_t_phi_t_Model(
        device, xDIM, yDIM, torch.zeros(xDIM, device=device),
        torch.zeros(xDIM, device=device),
        cond_q_mean_net_constructor,
        torch.zeros(xDIM, device=device),
        F_fn, G_fn, p_0_dist, 'last',
        phi_window_size, KRR_constructor, True, 1.0, True,
        time_store_size
    )

    def add_theta_grads_to_params(grads):
        culm_idx = 0
        for p in phi_model.F_fn.parameters():
            p.grad = p.grad + grads[culm_idx:culm_idx+p.numel()].view_as(p.grad)
            culm_idx += p.numel()
        assert culm_idx == theta_dim

    def theta_func_constructor():
        net = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, theta_dim+1)
        )
        return models.NN_Func_Polar(net, 32, theta_dim).to(device)

    mseloss = nn.MSELoss()
    cosloss = nn.CosineSimilarity()
    def dir_mag_loss(pred, true):
        pred_norm = torch.norm(pred, dim=1).unsqueeze(-1) + 1e-3
        true_norm = torch.norm(true, dim=1).unsqueeze(-1) + 1e-3
        dir_loss = -torch.mean(cosloss(pred, true))
        mag_loss = mseloss( torch.log(pred_norm), torch.log(true_norm))
        return dir_loss, mag_loss

    def get_model_parameters():
        return transition.parameters()

    theta_grad = models.ThetaGrad(
        device, phi_model, theta_func_constructor,
        theta_window_size, theta_dim, get_model_parameters, add_theta_grads_to_params
    )

    theta_optim = torch.optim.Adam(get_model_parameters(), lr=cfg.theta_lr)

    def get_g_prob():
        return phi_model.G_fn(phi_model.q_t_mean_list[T].reshape(1, xDIM))\
            .log_prob(flat_data[T, :])[0].item()
        
    def get_Tm1_g_prob():
        x_Tm1 = phi_model.cond_q_t_mean_net_list[T](\
            phi_model.q_t_mean_list[T].reshape(1,32))
        return phi_model.G_fn(x_Tm1).log_prob(flat_data[T-1, :])[0].item()

    def avg_f_prob(xs):
        # Given numpy array of xs, calculates mean of log f(x_k|x_{k-1})
        log_probs = []
        for i in range(1, len(xs)):
            x_km1 = torch.from_numpy(xs[i-1]).to(device).reshape(1,xDIM)
            x_k = torch.from_numpy(xs[i]).to(device).reshape(1,xDIM)
            log_prob = phi_model.F_fn(x_km1).log_prob(x_k)[0].item()
            log_probs.append(log_prob)
        log_probs = np.array(log_probs)
        return np.mean(log_probs)

    xs = []
    prev_xs = []
    thetas = []
    theta_scales = []
    avg_f_probs = []
    phi_iters_all = []
    g_probs_all = []
    Tm1_g_probs_all = []
    phi_lrs = []
    q_stds = []
    q_tm1_stds = []

    phi_T_lr = cfg.phi_T_lr
    phi_Tm1_lr = cfg.phi_Tm1_lr

    theta_net_losses_all = []
    valid_losses = []

    os.mkdir('transitions')

    for T in tqdm(range(cfg.num_iters)):

        phi_model.advance_timestep(flat_data[T, :])
        theta_grad.advance_timestep()

        phi_T_params = []
        phi_Tm1_params = []
        for t in range(max(0, T - phi_model.window_size + 1), T + 1):
            phi_T_params = phi_T_params + [phi_model.q_t_mean_list[t], \
                phi_model.q_t_log_std_list[t]]
            phi_Tm1_params = phi_Tm1_params + [\
                *phi_model.cond_q_t_mean_net_list[t].parameters(),
                phi_model.cond_q_t_log_std_list[t]]

        phi_T_optim = torch.optim.Adam(phi_T_params, lr=phi_T_lr)
        phi_Tm1_optim = torch.optim.Adam(phi_Tm1_params, lr=phi_Tm1_lr)

        g_probs = []
        Tm1_g_probs = []

        for i in range(cfg.phi_iters):
            phi_T_optim.zero_grad()
            phi_Tm1_optim.zero_grad()
            phi_model.populate_phi_grads(flat_data, 32)
            phi_T_optim.step()
            phi_Tm1_optim.step()
            g_probs.append(get_g_prob())
            Tm1_g_probs.append(get_Tm1_g_prob())

            if i > 20:
                if np.ptp(g_probs[-20:]) < cfg.g_thresh and \
                    np.ptp(Tm1_g_probs[-20:]) < cfg.g_thresh:
                    break


        phi_iters_all.append(i)
        g_probs_all = g_probs_all + g_probs
        Tm1_g_probs_all = Tm1_g_probs_all + Tm1_g_probs
        phi_lrs.append(phi_T_lr)


        xs.append(phi_model.q_t_mean_list[T].cpu().detach().clone().numpy())
        prev_xs.append(phi_model.cond_q_t_mean_net_list[T](\
            phi_model.q_t_mean_list[T].reshape(1,32)).cpu().detach().clone().numpy()
        )
        q_stds.append(phi_model.q_t_log_std_list[T].cpu().detach().clone().numpy())
        q_tm1_stds.append(phi_model.cond_q_t_log_std_list[T].cpu().detach().clone().numpy())

        if T >= phi_window_size - 1:
            phi_model.update_V_t(flat_data, 512)
            Vx_optim = torch.optim.Adam(phi_model.get_V_t_params(),
                lr=0.01)
            for k in range(25):
                Vx_optim.zero_grad()
                V_loss, _, _ = phi_model.V_t_loss(flat_data,
                    10)
                V_loss.backward()
                Vx_optim.step()
        
        if T >= theta_window_size:
            net_inputs, net_targets = theta_grad.generate_training_dataset(                
                cfg.theta_dataset_size, flat_data)
            net_inputs = net_inputs.detach()
            net_targets = net_targets.detach()

            valid_inputs, valid_targets = theta_grad.generate_training_dataset(
                128, flat_data
            )
            valid_inputs = valid_inputs.detach()
            valid_targets = valid_targets.detach()

            # Dont update norm on first since for learning F first grad is all
            # zeros
            if T > theta_window_size:
                theta_grad.theta_func_TmL.update_normalization(net_inputs, net_targets, 0.9)

            net_optim = torch.optim.Adam(theta_grad.theta_func_TmL.parameters(),
                lr=cfg.theta_net_lr)
            theta_net_losses = []
            for i in range(cfg.theta_net_update_steps):
                net_optim.zero_grad()
                idx = torch.randint(0, cfg.theta_dataset_size,
                    (cfg.theta_net_minibatch_size,))
                preds = theta_grad.theta_func_TmL(net_inputs[idx, :])
                dir_loss, mag_loss = dir_mag_loss(preds, net_targets[idx, :])
                loss = dir_loss * cfg.dir_mag_loss_balance + \
                    mag_loss * (1-cfg.dir_mag_loss_balance)
                loss.backward()
                net_optim.step()
                theta_net_losses.append([dir_loss.item(), mag_loss.item()])
            theta_net_losses_all.append(theta_net_losses)
            theta_grad.theta_func_TmL.eval()

            valid_preds = theta_grad.theta_func_TmL(valid_inputs)
            valid_dir_loss, valid_mag_loss = dir_mag_loss(valid_preds,
                valid_targets)
            valid_losses.append([valid_dir_loss.item(), valid_mag_loss.item()])

        if T > 1:
            theta_optim.zero_grad()
            theta_grad.populate_theta_grads(32, flat_data)
            theta_optim.step()
            theta_scales.append(phi_model.F_fn.transition.net[0].scale.item())
            avg_f_probs.append(avg_f_prob(xs[-4:]))


        if (T % cfg.save_interval == 0) or (T == (cfg.num_iters - 1)):
            torch.save(transition.state_dict(), 'transitions/transition_{}.pt'.format(T))
            np.save('thetas.npy', np.array(thetas))
            np.save('scales.npy', np.array(theta_scales))
            np.save('avg_f_probs.npy', np.array(avg_f_probs))
            np.save('phi_iters_all.npy', np.array(phi_iters_all))
            np.save('q_means.npy', np.array(xs))
            np.save('g_probs_all.npy', np.array(g_probs_all))
            np.save('Tm1_g_probs_all.npy', np.array(Tm1_g_probs_all))
            np.save('phi_lrs.npy', np.array(phi_lrs))
            np.save('q_tm1_stds.npy', np.array(q_tm1_stds))
            np.save('q_stds.npy', np.array(q_stds))
            np.save('prev_xs.npy', np.array(prev_xs))
            np.save('theta_net_losses_all.npy', np.array(theta_net_losses_all))
            np.save('theta_net_valid_losses.npy', np.array(valid_losses))


if __name__ == "__main__":
    main()





