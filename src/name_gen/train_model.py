import os
import random
import string
import time
from datetime import datetime
from itertools import accumulate, pairwise

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from name_gen.model import RNN
from name_gen.run_model import sample
from name_gen.utils import one_hot, tensor_to_char, tokenize, tokens

n_epochs = 15000

batch_size = 10

num_layers = 2

drop_prob = 0.5

learning_rate = (
    0.01
)


def load_data():
    print("reading data")
    species_df = pd.read_csv("data/species.csv", header=None)  # , nrows=25000)

    print("adding delimeters")
    species = species_df.convert_dtypes()

    species = species + "$"  # add ending delimeter character
    species[0] = species[0].str.lower()

    print(species)
    return species
    # print("randomizing order")
    # fspecies = species.sample(frac=1).reset_index(drop=True) # randomize order


def target(tensor):
    i = tensor.argmax()
    return torch.tensor([i])


def train(letter, next_letter, rnn, criterion):
    hidden = rnn.initHidden()
    output, hidden = rnn(letter, hidden)

    output_letter = one_hot(tensor_to_char(output))
    loss = criterion(output, target(next_letter))

    return output, loss


def main():

    running_loss = 0
    avg_running_loss = 0
    batch_loss = 0

    rnn = RNN(len(tokens), 128, len(tokens))
    print(rnn)
    print(f"training {n_epochs} epochs")
    species = load_data()

    opt = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    rnn.train()
    for i in tqdm(range(n_epochs)):
        loss = 0

        opt.zero_grad()

        for j in range(batch_size):

            training_line = species.sample(1).squeeze()

            training_tensors = [one_hot(le) for le in training_line]

            tensor_pairs = pairwise(training_tensors)

            line_loss = 0
            for (letter, next_letter) in tensor_pairs:
                output, letter_loss = train(letter, next_letter, rnn, criterion)
                loss += letter_loss
                line_loss += letter_loss

            running_loss += line_loss.item() / len(training_line)

        loss.backward()
        opt.step()

        if i % 100 == 0:
            avg_running_loss = running_loss / 100
            running_loss = 0
            print(avg_running_loss)
            sample(rnn)
            rnn.train()

    torch.save(rnn.state_dict(), f"models/{datetime.now()}")


if __name__ == "__main__":
    main()
