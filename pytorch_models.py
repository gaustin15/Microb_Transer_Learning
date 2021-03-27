import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import MicroDataset
import pytorch_lightning as pl


### Pytorch Modules

class FFNN(torch.nn.Module):
    def __init__(self, 
                 dataset, 
                 layer_sizes = [128, 64, 32], 
                dropout = .2):
        super(FFNN, self).__init__()
        
        linear_layers = [ nn.Linear( dataset.n_taxa, layer_sizes[0]), 
                          nn.BatchNorm1d(layer_sizes[0]), 
                          nn.Dropout(dropout), 
                          nn.GELU()]
        
        for i in range(len(layer_sizes)-1):
            linear_layers += [ nn.Linear(layer_sizes[i], layer_sizes[i+1]), 
                               nn.BatchNorm1d(layer_sizes[i+1]), 
                               nn.Dropout(dropout), 
                               nn.GELU()]
            
        linear_layers += [ nn.Linear(layer_sizes[-1], 2), 
                           nn.Softmax() ]
        
        self.linear_net = nn.Sequential(*linear_layers)
        
    def forward(self, x):
        out = self.linear_net(x.float())
        return(out)
    
    
    
    
class SAE(torch.nn.Module):
    """
    The shallow autoencoder
    """
    def __init__(self, 
                 dataset, 
                 layer_dim = 64, 
                 dropout = .2):
        super(SAE, self).__init__()
        self.encoder = nn.Sequential(*[nn.Linear(dataset.n_taxa, layer_dim), nn.ReLU()])
        self.decoder = nn.Sequential(*[nn.Linear(layer_dim, dataset.n_taxa), nn.ReLU(), nn.Softmax()])
        
    def forward(self, x):
        out = self.decoder( self.encoder(x) )
        return(out)
    
    def encode(self, x):
        return( self.encoder(x) )
    
    
    
    
### Pytorch Lightning Modules -- Used for the training loop
### See the pytorch-lightning library's documentation for more details
###### (if anyone'e not familiar, it's kind of like keras for pytorch (but with a little more flexibility than keras)

class LitEncoderDecoder(pl.LightningModule):
    def __init__(self, 
                 train_df,
                 valid_df, 
                 layer_dim = 64,
                 dropout = .2,
                 learning_rate=1e-3, 
                 batch_size=50, 
                 is_marker=False):
        super().__init__()
        
        self.train_dataset = MicroDataset(train_df, is_marker=is_marker)
        self.valid_dataset = MicroDataset(valid_df, is_marker=is_marker)
        #self.test_dataset = MicroDataset(test_df)
        
        
        self.train_loader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size = batch_size, shuffle = False)
        #test_dataloader = DataLoader(self.test_dataset, batch_size = 50, shuffle = False)
        self.model = SAE(self.train_dataset, layer_dim = layer_dim, dropout = dropout)
        self.learning_rate = learning_rate
        
        # they use MSE for reconstruction loss
        self.loss_func = nn.MSELoss()
        
        
    def train_dataloader(self):
        return(self.train_loader)
    
    def val_dataloader(self):
        return(self.valid_loader)
    
    def forward(self, x):
        x = self.model(x)
        return x

    def split_batch(self, batch):
        return batch[0], batch[1]

    def training_step(self, batch, batch_idx):
        x, y = self.split_batch(batch)
        y_hat = self(x)
        loss = self.loss_func(y_hat, x)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.split_batch(batch)
        y_hat = self(x)
        loss = self.loss_func(y_hat, x)
        self.log('val_loss', loss)
        return {'val_loss':loss}
 
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


