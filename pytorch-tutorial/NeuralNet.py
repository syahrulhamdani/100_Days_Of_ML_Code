import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Build feedforward neural network.

        parameters
        ----------
        input_size: int, size of the input
        output_size: int, size of the output layer
        hidden_layers: list of ints, size of the hidden layers
        drop_p: float ranging (0,1), dropout probability
        '''
        super().__init__()
        self.hidden_layers = nn.Linear(in_features=input_size, out_features=hidden_layers[0])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend( [nn.Linear(h1, h2) for h1, h2 in layer_sizes] )
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(drop_p)
    
    def forward(self, x):
        'forward pass `x` through the network, returns the output logits'
        for linear in self.layer_sizes:
            x = F.relu(linear(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)
