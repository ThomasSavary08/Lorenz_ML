# Libraries
from torch import nn
from typing import List

# Definition of the model (GRU + classifier)
class Predictor(nn.Module):
    
    # Instanciate a network
    def __init__(self, input_size: int, hidden_size: int, n_layers: int, dropout: float, classifier_dim: List[int], n_classes: int):
        '''
        Instanciate a network for prediction.
            Parameters:
                intput_size (int): Dimension of the input.
                hidden_size (int): Dimension of the hidden state.
                n_layers (int): Number of stacked GRU.
                dropout (float): Dropout probability.
                classifier_dim (List[int]): List of hidden layers dimension for the classifier.
                n_classers (int): Number of classes to predict.
            Returns:
                A network for prediction as torch.nn.Module.
        '''
        super().__init__()
        self.input_size = input_size
        self.hidden_state = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.classifier_dim = classifier_dim
        self.n_classes = n_classes
        self.GRU = nn.GRU(input_size, hidden_size, n_layers, bias = True, batch_first = True, dropout = dropout, bidirectional = False)
        self.classifier = nn.ModuleList()
        if (len(classifier_dim) == 0):
            self.classifier.append(nn.Linear(hidden_size, n_classes))
        else:
            for i in range(len(classifier_dim)):
                if (i == 0):
                    self.classifier.append(nn.Linear(hidden_size, classifier_dim[0]))
                else:
                    self.classifier.append(nn.Linear(classifier_dim[i-1], classifier_dim[i]))
            self.classifier.append(nn.Linear(classifier_dim[-1], n_classes))
    
    # Forward a batch sequences through the network
    def forward(self, x):
        '''
        Instanciate a network for prediction.
            Parameters:
                x (torch.Tensor): input batch of dimension (batch_size, seq_length, input_size).
            Returns:
                h (torch.Tensor): logits of dimension (batch_size, n_classes)
        '''
        # Forward through the GRU
        _, h = self.GRU(x)

        # Forward through classifier
        for i in range(len(self.classifier) - 1):
            h = self.classifier[i](h)
            h = nn.functional.relu(h)
        h = self.classifier[-1](h)

        # Return logits
        return h.squeeze(0)