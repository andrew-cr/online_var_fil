"""
To be interacted with as a script with a hydra config file
"""
import numpy as np
import torch
import torch.nn as nn
from core.data_generation import GaussianHMM, construct_HMM_matrices
from core.utils import save_git_hash
from tqdm import tqdm
import math
import subprocess
import hydra
import os


@hydra.main(config_path='conf', config_name="generate_data")
def main(cfg):
    save_git_hash()

    seed = cfg.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    DIM = cfg.dim

    F,G,U,V = construct_HMM_matrices(dim=DIM,
                                    F_eigvals=np.random.uniform(
                                        -cfg.F_max_abs_eigval,
                                        cfg.F_max_abs_eigval, (DIM)),
                                    G_eigvals=np.random.uniform(
                                        -cfg.G_max_abs_eigval,
                                        cfg.G_max_abs_eigval, (DIM)),
                                    U_std=cfg.U_std,
                                    V_std=cfg.V_std,
                                    diag=cfg.diagFG)


    data_gen = GaussianHMM(xdim=DIM, ydim=DIM, x_0=np.zeros(DIM), F=F, G=G, U=U, V=V)

    x_np, y_np = data_gen.generate_data(cfg.num_to_generate)

    np.save('datapoints.npy', np.stack((x_np, y_np)))

    # Save the true model params
    np.save('F.npy', F)
    np.save('G.npy', G)
    np.save('U.npy', U)
    np.save('V.npy', V)

if __name__ == "__main__":
    main()