from name_gen.train_model import one_hot, tokenize, tensor_to_char
from collections import Counter
import torch


def test_one_hot():
    space_tensor = one_hot(" ")
    space_tensor2 = torch.tensor(
        [
            [
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ]
    )
    y_tensor = one_hot("y")
    y_tensor2 = torch.tensor(
        [
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
            ]
        ]
    )
    assert torch.equal(space_tensor, space_tensor2)
    assert torch.equal(y_tensor, y_tensor2)


def test_tokenize():
    idx = tokenize("n")
    assert idx == 14


def test_tensor_to_char():
    letter_tensor = one_hot("l")
    # repeat a bunch of times because tensor_to_char is random
    predicted = [tensor_to_char(letter_tensor) for _ in range(10000)]
    c = Counter(predicted)
    assert c.most_common(1)[0][0] == "l"
