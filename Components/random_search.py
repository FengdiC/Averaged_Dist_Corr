import numpy as np
import os
import torch

def random_search_cartpole(seed):
    rng = np.random.RandomState(seed=seed)
    gamma_coef = rng.randint(low=1, high=100)/10.0
    scale = rng.randint(low=1, high=50)
    lr = rng.randint(low=3, high=50) / 10000.0
    lr_weight = rng.randint(low=1, high=50)/1000.0
    gamma = rng.choice([0.9,0.95,0.99,0.995])
    hid = rng.choice([32,64,128])
    critic_hid = rng.choice([32,64,128])
    buffer = rng.choice([32,64,128])

    hyperparameters = {"gamma_coef":gamma_coef, "scale":scale, "lr":lr,"hid":hid,"buffer":buffer,
                       "lr_weight":lr_weight,"critic_hid":critic_hid,"gamma":gamma}

    return hyperparameters

def random_search_Reacher(seed):
    rng = np.random.RandomState(seed=seed)
    gamma_coef = rng.randint(low=5, high=2000)/100.0
    scale = rng.randint(low=1, high=150)
    lr = rng.randint(low=3, high=50) / 10000.0
    lr_weight = rng.randint(low=3, high=50)/10000.0
    gamma = rng.choice([0.8,0.9,0.95,0.99])
    hid = rng.choice([8,16,32,64])
    critic_hid = rng.choice([8,16,32,64])
    buffer = rng.choice([1,5,25,45])

    hyperparameters = {"gamma_coef":gamma_coef, "scale":scale, "lr":lr,"hid":hid,"buffer":buffer,
                       "lr_weight":lr_weight,"critic_hid":critic_hid,"gamma":gamma}

    return hyperparameters


def set_one_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)