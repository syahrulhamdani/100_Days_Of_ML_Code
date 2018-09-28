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

def train(model, trainloader, testloader, criterion, optimizer, epochs=3):
    epoch_loss = 0
    print_every = 50
    steps = 0

    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            steps += 1

            images.resize_(images.shape[0], 784)
            # forward pass
            output = model.forward(images)
            loss = criterion(output, labels)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                test_loss, accuracy = _validation(model, testloader, criterion)
                print('Epoch: {}/{}..'.format(e+1, epochs),
                      'Training Loss: {:.3f}'.format(epoch_loss/print_every),
                      'Test Loss: {:.3f}'.format(test_loss/len(testloader)),
                      'Accuracy: {:.3f}'.format(accuracy/len(testloader)))
                epoch_loss = 0
                model.train()


def _validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        images.resize_(images.size()[0], 784)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        # compute accuracy
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy