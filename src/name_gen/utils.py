import string

import numpy as np
import torch
import torch.nn.functional as F

tokens = " " + string.ascii_lowercase + "$"


def tokenize(char: str):
    return tokens.find(char)


def one_hot(letter):
    zero_tensor = torch.zeros(1, len(tokens))  # , requires_grad=True)
    zero_tensor[0][tokenize(letter)] = 1
    return zero_tensor


def tensor_to_char(tensor):
    """stochastically pick a letter based on the output of the model"""
    choice = tensor_to_index(tensor)
    return tokens[choice]


def tensor_to_index(tensor):
    """stochastically pick a token index based on the output of the model"""
    p = F.softmax(tensor, dim=1).data

    choice = np.random.choice(len(tokens), p=np.array(p[0]))

    return choice
