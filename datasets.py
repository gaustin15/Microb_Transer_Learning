### This file contains functions used to create the pytorch-format datasets + dataloaders
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os


class MicroDataset(Dataset):
    """Dataset class.
    Args:
       df (dataframe): Pandas dataframe, must be from the deepmicro repo
 
       is_marker (bool): If it is a marker dataset, set to True.
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
            self.y = torch.Tensor( (df.disease != 'leaness').values ).long()
        else:
            self.y = torch.Tensor( (df.disease != 'n').values).long()
            
        #self.matrix.requires_grad=True
        
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

def get_shared_taxa_dfs(dfs):
    """
    This function is to combine multiple dataframes so that their dimensionality is the same
    """
    
    disease_map=pd.DataFrame( ['Zeller_fecal_colorectal_cancer',
                             'Quin_gut_liver_cirrhosis',
                             'metahit',
                             't2dmeta_long',
                             'WT2D',
                             'Chatelier_gut_obesity', 
                              't2dmeta_short'], 
                 ['Colorectal', 'Cirrhosis', 'IBD', 'C-T2D', 'EW-T2D', 'Obesity', 'C-T2D'] 
                ).reset_index()

    disease_map.columns = ['clean', 'raw']
    cols = []#['disease']
    [cols.append( df.columns[df.columns.str.contains('k__')] ) for df in dfs]
    all_dfs = pd.concat( [dfs[i][cols[i]].astype(float)# if i > 0
                          #else dfs[i][cols[i]] 
                          for i in range(len(dfs))], axis = 1).fillna(0)
    all_dfs = all_dfs.T.groupby(['dataset_name']).sum().T
    all_dfs['disease'] = pd.concat( [df['disease'] for df in dfs] )

    all_dfs['dataset'] = all_dfs.index.str.split('.').str[0]
    all_dfs = all_dfs[['dataset', 'disease']+list(all_dfs.columns[:-2])]
    all_dfs = all_dfs.reset_index(drop=True)
    all_dfs = all_dfs.fillna(0)
    all_dfs=disease_map.merge(all_dfs, left_on='raw', right_on='dataset').drop(['raw', 'dataset'], axis=1)
    all_dfs.columns=['dataset'] + list(all_dfs.columns[1:])
    
    names = ['Obesity', 'Colorectal', 'IBD', 'EW-T2D','C-T2D', 'Cirrhosis']
    
    datasets = [all_dfs.loc[all_dfs.dataset==a]\
            for a in names]
    
    return({names[i]:datasets[i] for i in range(len(names))} )



def get_shared_taxa_markers(dfs):
    """
    same as last function, but for markers
    """
    disease_map=pd.DataFrame( ['Zeller_fecal_colorectal_cancer',
                             'Quin_gut_liver_cirrhosis',
                             'metahit',
                             't2dmeta_long',
                             'WT2D',
                             'Chatelier_gut_obesity', 
                              't2dmeta_short'], 
                 ['Colorectal', 'Cirrhosis', 'IBD', 'C-T2D', 'EW-T2D', 'Obesity', 'C-T2D'] 
                ).reset_index()

    disease_map.columns = ['clean', 'raw']
    marker_cols = []
    [marker_cols.append( df.columns[df.columns.str.contains('gi[|]')] ) for df in markers]

    all_markers = pd.concat( [markers[i][marker_cols[i]].astype(float) for i in range(len(markers))], axis = 1).fillna(0)
    all_markers = all_markers.T.groupby('dataset_name').sum().T
    all_markers['disease'] = pd.concat( [df['disease'] for df in markers] )
    all_markers['dataset'] = all_markers.index.str.split('.').str[0]
    all_markers = all_markers[['dataset', 'disease']+list(all_markers.columns[:-2])]
    all_markers = all_markers.reset_index(drop=True)
    all_markers = all_markers.fillna(0)
    all_markers=disease_map.merge(all_markers, left_on='raw', right_on='dataset').drop(['raw', 'dataset'], axis=1)
    all_markers.columns=['dataset'] + list(all_markers.columns[1:])
    names = ['Obesity', 'Colorectal', 'IBD', 'EW-T2D','C-T2D', 'Cirrhosis']
    
    datasets = [all_markers.loc[all_markers.dataset==a]\
            for a in names]
    
    return({names[i]:datasets[i] for i in range(len(names))} )


