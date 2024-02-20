# Libraries
import math
import utils
import model
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

# Model parameters
input_size = 3
hidden_size = 64
n_layers = 1
dropout = 0.1
classifier_dim = [64,32]
n_classes = 3

# Training parameters
n_epochs = 20
batch_size = 64
init_lr = 3e-4
best_score = math.inf

# Load datasets
x_train_path, y_train_path = './Data/x_train.npy', './Data/y_train.npy'
x_val_path, y_val_path = './Data/x_test.npy', './Data/y_test.npy'
trainSet = utils.CreateDataset(x_train_path, y_train_path)
valSet = utils.CreateDataset(x_val_path, y_val_path)
n_train, n_val = trainSet.__len__(), valSet.__len__()

# Data loader
train_DL = torch.utils.data.DataLoader(trainSet, batch_size = batch_size, shuffle = True)
val_DL = torch.utils.data.DataLoader(valSet, batch_size = n_val, shuffle = False)

# Instanciate the network
network = model.Predictor(input_size, hidden_size, n_layers, dropout, classifier_dim, n_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = network.to(device)

# Loss, optimizer and scheduler
criterion = nn.CrossEntropyLoss(reduction = "mean")
optimizer = torch.optim.AdamW(network.parameters(), lr = init_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, 1e-6, verbose = False)
list_train_loss = []
list_val_loss = []

# Training
print("Beginning of the training...")
print("")

# Loop on epochs
for epoch in range(1, n_epochs + 1):

    # Initiate parameters to follow the evolution of training loss during the epoch
    n = 0
    n_val = 0
    running_loss = 0.
    running_loss_val = 0.
    tqdm_train = tqdm(train_DL, total = int(len(train_DL)))
    tqdm_train.set_description(f"Epoch {epoch}")

    # Loop on training batches
    for _, inputs in enumerate(tqdm_train):

        # Training mode
        network.train()

        # Clean gradients
        optimizer.zero_grad()

        # Compute loss
        x, y = inputs
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(True):
            y_hat = torch.nn.functional.softmax(network(x), dim = -1)
            loss = criterion(y_hat, y)
            loss.backward()

        # Update parameters
        optimizer.step()

        # Update running loss
        running_loss += loss.item() * x.size(0)
        n += x.size(0)
        tqdm_train.set_postfix(loss = (running_loss / n))

    list_train_loss.append(running_loss / n)

	# Loop on validation batches
    network.eval()
    with torch.no_grad():
        for inputs in val_DL:
            x, y = inputs
            x, y = x.to(device), y.to(device)
            y_hat = torch.nn.functional.softmax(network(x), dim = -1)
            loss = criterion(y_hat, y)
            running_loss_val += loss.item() * x.size(0)
            n_val += x.size(0)
    acc = (torch.sum(torch.argmax(y_hat, dim = -1) == y) / n_val).item() * 100
    print("Validation loss: {:.3f}".format(running_loss_val / n_val))
    print("Validation accuracy: {:.1f}".format(acc))
    print()
    list_val_loss.append(running_loss_val / n_val)
    
	# Update learning rate
    scheduler.step()
	
	# Save the best model
    if (list_val_loss[-1] < best_score):
        best_score = list_val_loss[-1]
        torch.save(network, './trainedModel.pt')

# Save the figure of losses evolution
assert len(list_val_loss) == len(list_train_loss)
x = np.asarray([i for i in range(1, len(list_val_loss) + 1)])
plt.figure(figsize = (10,6))
plt.plot(x, np.asarray(list_train_loss), color = 'blue', label = 'Training')
plt.plot(x, np.asarray(list_val_loss), color = 'orange', label = 'Evaluation')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()