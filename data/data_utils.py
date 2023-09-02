import random
import pickle
import torch

M = 2 ** 32 - 1

def sample(x, size):
    # https://gist.github.com/yoavram/4134617
    i = random.sample(range(x.shape[0]), size)
    return torch.tensor(x[i], dtype=torch.int16)
    # x = np.random.permutation(x)
    # return torch.tensor(x[:size])


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
