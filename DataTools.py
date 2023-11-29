import numpy as np
import torch

import gc


def set_random_states(manual_seed):
    """
    Initializes the random number generators of NumPy, PyTorch, and PyTorch CUDA by passing the input seed.
    :param manual_seed: An integer to be passed to the random number generators.
    """
    np.random.seed(manual_seed)

    torch.manual_seed(manual_seed)

    torch.cuda.manual_seed(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_random_states():
    """
    Retrieves the current states of randomness of NumPy, PyTorch, and PyTorch CUDA.
    :return: Three states of randomness for NumPy, PyTorch, and PyTorch CUDA respectively.
    """
    np_random_state = np.random.get_state()
    torch_random_state = torch.random.get_rng_state()
    cuda_random_state = torch.cuda.random.get_rng_state()

    return np_random_state, torch_random_state, cuda_random_state


def reset_random_states(np_random_state, torch_random_state, cuda_random_state):
    """
    Sets the current states of randomness of NumPy, PyTorch, and PyTorch CUDA.
    :param np_random_state: The state at which the NumPy random generator will be set.
    :param torch_random_state: The state at which the PyTorch random generator will be set.
    :param cuda_random_state: The state at which the PyTorch CUDA random generator will be set.
    :return:
    """
    torch.cuda.empty_cache()

    np.random.set_state(np_random_state)
    torch.random.set_rng_state(torch_random_state)
    torch.cuda.random.set_rng_state(cuda_random_state)

    gc.collect()
    torch.cuda.empty_cache()
