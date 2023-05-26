import argparse
import random
import string

import torch

from name_gen.model import RNN
from name_gen.utils import one_hot, tensor_to_char, tokens

def sample(rnn):
    rnn.eval()
    stop = False

    test_string = f"{random.choice(string.ascii_lowercase)}"
    input = one_hot(test_string)
    hidden = rnn.initHidden()

    print("making string")
    n = 0
    while not stop:
        output, hidden = rnn(input, hidden)

        s = tensor_to_char(output)
        if s != "$":
            test_string += s
            if n == 34:
                stop = True
            else:
                n += 1
                input = one_hot(s)
        else:
            stop = True

    print(test_string)
    print(len(test_string))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    rnn = RNN(len(tokens), 128, len(tokens))
    rnn.load_state_dict(torch.load(args.filename))
    sample(rnn)
