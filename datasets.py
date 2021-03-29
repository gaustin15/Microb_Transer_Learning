### This file contains functions used to create the pytorch-format datasets + dataloaders
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os


class MicroDataset(Dataset):
    """Dataset class for column dataset.
    Args:
       cats (list of str): List of the name of columns contain
                           categorical variables.
       conts (list of str): List of the name of columns which 
                           contain continuous variables.
       y (Tensor, optional): Target variables.
       is_reg (bool): If the task is regression, set ``True``, 
                      otherwise (classification) ``False``.
       is_multi (bool): If the task is multi-label classification, 
                        set ``True``.
    """
    def __init__(self, df, is_marker = False):
        df = df.sample(frac=1)
        if not is_marker:
            self.taxa_cols = df.columns[df.columns.str.contains('k__')] 
        else:
            self.taxa_cols = df.columns[df.columns.str.contains('gi[|]')] 
        self.matrix = torch.Tensor( df[self.taxa_cols].astype(float).values ).float()
        
        #scale the dataset to be in relative abundance space
        self.matrix=F.softmax(self.matrix)
            
        if sum(df.disease == 'n')==0:
            self.y = torch.Tensor(df.disease != 'leaness' ).long()
        else:
            self.y = torch.Tensor(df.disease != 'n' ).long()
        
        self.n_samples, self.n_taxa= self.matrix.shape
        
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return [self.matrix[idx], self.y[idx]]
    
    

def load_abundance_data():
    #function to load a list of dfs, all of which are from the abundance folder
    # atm, it requires the data paths to be set up in a specific way
    dfs = []
    for file in os.listdir('data/abundance/'):
        df = pd.read_csv('data/abundance/' + file, sep = '\t') 
        df.index = df.dataset_name
        dfs.append(df.T.drop('dataset_name') )
    return(dfs)

def load_markers_data():
    #function to load a list of dfs, all of which are from the markers folder
    # atm, it requires the data paths to be set up in a specific way
    markers = []
    for file in os.listdir('data/marker/'):
        df = pd.read_csv('data/marker/' + file, sep = '\t') 
        df.index = df.dataset_name
        markers.append(df.T.drop('dataset_name') )
    return(markers)
