import torch
import pandas as pd
import numpy as np

import pyximport
pyximport.install()

import learn2learn as l2l
from learn2learn.data import TaskDataset
from learn2learn.algorithms.lightning import LightningMAML
from learn2learn.utils.lightning import EpisodicBatcher

from baseline.pytorch_models import LitFFNN
#from baseline.training_functions import *
from baseline.training_functions import make_split

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger

from datasets import load_abundance_data, get_shared_taxa_dfs
from datasets import MicroDataset, Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ax import optimize

import math
import os
import warnings
warnings.filterwarnings('ignore')

MAML_FFNN_parameters = [
    {
     'name':'layer_1_size', 
     'type':'choice', 
     'values':[128, 256, 512, 1024]
    },
    {
     'name':'layer_2_size', 
     'type':'choice', 
     'values':[256, 128, 64]
    },
    {
     'name':'layer_3_size', 
     'type':'choice', 
     'values':[128, 64, 32]
    },
    {
    'name':'learning_rate', 
     "type": "range",
     "bounds": [1e-4, 1e-1],
    },
    {
    'name':'dropout', 
    'type':'choice', 
    'values':[.1, .3, .5]
    },
    {
    'name':'adaptation_lr', #learning rate during metalearning adaptation
     "type": "range",
     "bounds": [1e-4, 1e-1],
    },
    {
    'name':'k',  #k, as in 'k'-shot learning (minimize loss after this many steps during metatraining
     "type": "choice",
     "values": [1,2,4,8,16],
    },
#     {'name':'adapt_steps',  #k, as in 'k'-shot learning (minimize loss after this many steps during metatraining
#      "type": "choice",
#      "values": [1,2,5,10],
#     }
    ]


def collate(a):
            """
            collate function to simplify the tasks -- only want to ever distinguish between class 0 and 1
            Each class can represent any positive/negative group from any dataset
            6 datasets ==> 12 classes ==> 132 distinct metalearning tasks
            """
            #idx = max([b[1] for b in a])
            q = torch.Tensor([b[1] for b in a])
            return(torch.cat( [b[0].unsqueeze(0) for b in a] ),
                   ( q == q.max() ).long() )    
    
def build_taskset(datasets, k=4):
    """
    Function to build a learn2learn TaskDataset
    datasest : list -- a list of the different datasets to create training tasks from
    """
    MetaDS = l2l.data.UnionMetaDataset( [l2l.data.MetaDataset( MicroDataset(t) ) for t in datasets] )
    dataset = l2l.data.MetaDataset(MetaDS)
    transforms = [
        l2l.data.transforms.NWays(dataset, n=2),
        l2l.data.transforms.KShots(dataset, k=k, replacement=True),
        l2l.data.transforms.LoadData(dataset)
    ]
    return( TaskDataset(dataset, 
                        transforms, 
                        num_tasks=( len(datasets) * 2 * (len(datasets) * 2 - 1) ), 
                        task_collate=collate) )
    
    
    


def build_MAML_FFNN_eval_func(train_df,
                              valid_df,
                              test_df,
                              all_trains, 
                              all_valids,
                              is_marker=False,
                              return_model=False
                              ):
    """
    wrapper function for ax optimization
    """
    def eval_func(hyperparams):
        """this function runs a complete train/valid/testing loop
        for an encoder model, given the specified hyperparameters, 
        it returns the AUC on the test set

        Defining it inside the tuning function to avoid having to manually pass the train_df and valid_df,
                Ax tuning doesn't allow that
        """
        
        #first, run the metalearning
        lightning = LitFFNN(train_df, 
                            valid_df, 
                            layer_1_dim = hyperparams['layer_1_size'],
                            layer_2_dim = hyperparams['layer_2_size'],
                            layer_3_dim = hyperparams['layer_3_size'],
                            learning_rate = hyperparams['learning_rate'],
                            batch_size=50, 
                            is_marker=is_marker,
                            dropout=hyperparams['dropout']
                            )
        
        #build LightningMAML object
        maml = LightningMAML(lightning.model, 
                             lr = hyperparams['learning_rate'],
                             adaptation_lr=hyperparams['adaptation_lr'], 
                             train_ways=2, 
                             test_ways=2,
                             train_shots = hyperparams['k'],
                             test_shots =hyperparams['k'],
                             train_queries = hyperparams['k'],
                             test_queries =hyperparams['k'],
                             adaptation_steps=hyperparams['k']) #doesn't necessarily need to be 'k' across the board
                                                                # this does simplify parameter search though

        
        # build metalearning dataloaders
        train_set = build_taskset(all_trains, k=hyperparams['k']*2)
        val_set = build_taskset(all_valids, k=hyperparams['k']*2)
        
        for task in train_set:
            X, y = task
        
        episodic_data = EpisodicBatcher(train_set, val_set)
        
        #set up the trainer/logger/callbacks
        checkpoint_callback=ModelCheckpoint(
                        filepath = 'checkpoint_dir',
                        save_top_k=1,
                        verbose=False,
                        monitor='valid_loss',
                        mode='min'
                        )

        #build trainer for MAML
        trainer = pl.Trainer(max_epochs = 1000,
                             min_epochs = 50,
                             progress_bar_refresh_rate=0,
                             weights_summary=None,
                             check_val_every_n_epoch=40,
                             checkpoint_callback=checkpoint_callback,
                             callbacks=[EarlyStopping(monitor='valid_loss', 
                                            patience=20)]) 
        
        #run the metalearning
        trainer.fit(maml, episodic_data)
        
        #load metalearned parameters into standard lightning
        ## follow this by runnng the task-specific training
        maml.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])
        lightning.model.load_state_dict(maml.model.module.state_dict())

        
        #remove the log/model files ==> otherwise memory usage would really stack up
        os.system('rm '+ checkpoint_callback.best_model_path)
        os.system('rm -R lightning_logs/*')

        
        #setr up the trainer/logger/callbacks
        checkpoint_callback=ModelCheckpoint(
                        filepath = 'checkpoint_dir',
                        save_top_k=1,
                        verbose=False,
                        monitor='val_loss',
                        mode='min'
                        )

        tube_logger = TestTubeLogger('checkpoint_dir', 
                                    name='test_tube_logger')

        trainer = pl.Trainer(max_epochs = 500,
                             min_epochs = 1,
                             logger=tube_logger,
                             progress_bar_refresh_rate=0,
                             weights_summary=None,
                             check_val_every_n_epoch=1,
                             checkpoint_callback=checkpoint_callback,
                            callbacks=[EarlyStopping(monitor='val_loss', 
                                                    patience=10)]) #the patience of 20 is mentioned in the DeepMicro paper

        #run training
        trainer.fit(lightning)

        #load model from best-performing epoch
        lightning.load_state_dict(torch.load(checkpoint_callback.best_model_path )['state_dict'])
        
        #delete the files once we're done with them -- no need to store things unnecessarily
        os.system('rm '+ checkpoint_callback.best_model_path)
        os.system('rm -R checkpoint_dir/test_tube_logger/*')
        
        #set model to eval()
        lightning=lightning.eval()

        # Get AUC for the full setup on the test set
        test_dataset=MicroDataset(test_df, is_marker=is_marker)
        test_loader=DataLoader(test_dataset, shuffle=False, batch_size=50)
        
        test_preds=[]
        test_trues=[]
        
        for batch in test_loader:
            test_preds.append(lightning(batch[0]))
            test_trues.append(batch[1])
        
        
        test_preds=torch.cat(test_preds).detach().numpy()
        test_roc = roc_curve( torch.cat(test_trues).detach().numpy(), test_preds[:,1] )
        auc_val=auc(test_roc[0], test_roc[1])
        if math.isnan(auc_val):
            auc_val=0
            
        if return_model:
            return(auc_val, lightning)
            
        return(auc_val) 
    
    return(eval_func)





def tune_MAML_FFNN(datasets, 
                   is_marker=False, 
                   seed=0, 
                   total_trials=15):
    
    #set a seed so that splits are the same across different model tests
    np.random.seed(seed)
    
    #make data splits
    splits = [make_split(df) for df in datasets]
    
    #do a train/val/test split for the first dataset
    train, test = make_split(datasets[0])
    train, valid = make_split(train)
    
    splits[0] = (train, valid)
    
    trains =  [s[0] for s in splits]
    vals =  [s[1] for s in splits]
    
    #build evaluation function
    eval_func = build_MAML_FFNN_eval_func(train, 
                                          valid, 
                                          test,
                                          all_trains=trains, 
                                          all_valids=vals, 
                                          is_marker=is_marker
                                          )
    
    #selectthe parameters based on the model_name
    parameters=MAML_FFNN_parameters

    #optimize hyperparapeter with Facebook's ax tuning 
    best_parameters, best_values, experiment, model = optimize(
        parameters = parameters,
        evaluation_function=eval_func, 
        total_trials = total_trials, 
        minimize=False)
    
    
    data_name=train.dataset.values[0] 
    if is_marker==False:
        data_type='Abundance'
    else:
        data_type='Marker'
    model_name = 'MAML_FFNN'
    return([model_name, data_type, data_name, best_parameters, best_values])



    
    
    
    
    
    
    
    
    
    
