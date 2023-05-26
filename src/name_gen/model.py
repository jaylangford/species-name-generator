import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers=2, drop_prob=0.5
    ):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, dropout=drop_prob
        )
        self.out = nn.Linear(hidden_size, output_size)

        # randomize initial weights of the linear layer
        self.out.weight.data.uniform_(-1, 1)

        self.dropout = nn.Dropout(drop_prob)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.out(output)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return (
            torch.zeros(self.num_layers, self.hidden_size, requires_grad=True),
            torch.zeros(self.num_layers, self.hidden_size, requires_grad=True),
        )
