# Libraries
import torch
import numpy as np

# Create Torch dataset
class CreateDataset(torch.utils.data.Dataset):
    
    # Instanciate a dataset
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path)
        self.y = np.load(y_path) 
        
    # Number of samples
    def __len__(self):
        assert self.x.shape[0] == self.y.shape[0]
        return self.x.shape[0]
    
    # Get an (x,y) couple from data
    def __getitem__(self, idx):
        x, y = self.x[idx,:,:], self.y[idx]
        x = torch.tensor(x, dtype = torch.float32, requires_grad = True)
        y = torch.tensor(y, dtype = torch.long)
        return x, y