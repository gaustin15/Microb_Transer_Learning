import numpy as np
import pandas as pd
import math
import os

from datasets import MicroDataset
from torch.utils.data import DataLoader
from baseline.pytorch_models import LitEncoderDecoder, LitDAE, LitVAE, LitFFNN
from baseline.training_functions import *# make_split, VAE_parameters, DAE_parameters, SAE_parameters


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger

from ax import optimize




def build_sharing_encoder_eval_func(train,
                                    valid,
                                    test_df,
                                    all_trains, 
                                    all_valids,
                                    model_name = 'DAE', 
                                    is_marker=False,
                                    return_model=False
                                    ):
    def eval_func(hyperparams):
        """this function runs a complete train/valid/testing loop
        for an encoder model, given the specified hyperparameters, 
        it returns the AUC on the test set

        Defining it inside the tuning function to avoid having to manually pass the train_df and valid_df,
                Ax tuning doesn't allow that
        """
        #Initialize the model
        if model_name=='DAE':
            lightning = LitDAE(all_trains, 
                               all_valids, 
                               layer_1_dim = hyperparams['layer_1_size'],
                               layer_2_dim = hyperparams['layer_2_size'],
                               learning_rate = .001,
                               batch_size=50, 
                               is_marker=is_marker
                              )
        if model_name=='VAE':
            lightning = LitVAE(all_trains, 
                               all_valids, 
                               layer_1_dim = hyperparams['layer_1_size'],
                               layer_2_dim = hyperparams['layer_2_size'],
                               learning_rate = .001,
                               batch_size=50, 
                               is_marker=is_marker
                              )
            
        if model_name=='SAE':
            lightning = LitEncoderDecoder(all_trains, 
                                          all_valids, 
                                          layer_dim = hyperparams['layer_size'], 
                                          learning_rate = .001,
                                          batch_size=50, 
                                          is_marker=is_marker
                                          )



        #setr up the trainer/logger/callbacks
        
        #issue re lightning versions
        try: 
            checkpoint_callback=ModelCheckpoint(
                            dirpath = 'checkpoint_dir',
                            save_top_k=1,
                            verbose=False,
                            monitor='val_loss',
                            mode='min'
                            )
        except:
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
                             min_epochs = 50,
                             logger=tube_logger,
                             progress_bar_refresh_rate=0,
                             weights_summary=None,
                             check_val_every_n_epoch=1,
                             checkpoint_callback=checkpoint_callback,
                            callbacks=[EarlyStopping(monitor='val_loss', 
                                                    patience=20)]) #the patience of 20 is mentioned in the DeepMicro paper

        #run training
        trainer.fit(lightning)

        #load model from best-performing epoch
        lightning.load_state_dict(torch.load(checkpoint_callback.best_model_path )['state_dict'])
        
        #delete the files once we're done with them -- no need to store things unnecessarily
        os.system('rm '+ checkpoint_callback.best_model_path)
        os.system('rm -R checkpoint_dir/test_tube_logger/*')
        
        #set model to eval()
        lightning=lightning.eval()

        train_ = MicroDataset(train, is_marker=is_marker)
        val_ = MicroDataset(valid, is_marker=is_marker)        
        
        # Fit the final predictions on just the dataset of interest
        X = np.vstack([lightning.model.encode(train_.matrix).detach().numpy(), 
                       lightning.model.encode(val_.matrix).detach().numpy()])

        y = np.hstack([train_.y.detach().numpy(),
                       val_.y.detach().numpy()] )

        #set up classifier model tuning
        if hyperparams['classifier_model']=='rf':
            eval_model=rf_eval
            eval_params = rf_parameters
        elif hyperparams['classifier_model']=='svm':
            eval_model=svm_eval
            eval_params=svm_parameters

        #tune the best classifier model, using 5-fold CV ==> this also employs AX under the hood
        best_parameters=eval_model(X, y, eval_params)

        # make a classifier with teh best hyperparams
        if hyperparams['classifier_model']=='rf':
            best_classifier=RandomForestClassifier()
            [setattr(best_classifier, a, q) for a,q in best_parameters.items()]

        elif hyperparams['classifier_model']=='svm':
            best_classifier=SVC(probability=True)
            [setattr(best_classifier, a, np.power(2., q)) for a,q in best_parameters.items()]

        # fit it on the full train, valid data
        best_classifier.fit(X, y)

        # Get AUC for the full setup on the test set
        test_dataset=MicroDataset(test_df, is_marker=is_marker)
        test_preds=best_classifier.predict_proba( lightning.model.encode(test_dataset.matrix).detach().numpy() )
        test_roc = roc_curve( test_dataset.y.detach().numpy(), test_preds[:,1] )
        auc_val=auc(test_roc[0], test_roc[1])
        if math.isnan(auc_val):
            auc_val=0
            
        if return_model:
            return(auc_val, (lightning, best_classifier))
                   
        return(auc_val) 
    
    return(eval_func)





def tune_shared_encoder(datasets, 
                        model_name = 'DAE',
                        is_marker=False, 
                        seed=0, 
                        total_trials=15):
    
    if model_name=='SAE':
        # we only have 10 possible hyperparam combos for SAE -- this is just gridsearch
        total_trials = 10
    
    #set a seed so that splits are the same across different model tests
    np.random.seed(seed)
    #make data splits
    #doing train/valid/test splits across all datasets
    # ==> note - we don't really need test split for t
    splits = [make_split(df) for df in datasets]
    #splits = [(make_split(a[0]), a[1]) for a in splits ]
    
    #redo the split for the dataset of interest (the one at idx 0)
    # so we include a test set
    train, test = make_split(datasets[0])
    train, valid = make_split(train)
    
    splits[0] = (train, valid)
    
    trains = pd.concat( [s[0] for s in splits], axis=0 ).reset_index(drop=True)
    vals = pd.concat( [s[1] for s in splits], axis=0 ).reset_index(drop=True)
    #tests = pd.concat( [s[1] for s in splits], axis=0 ).reset_index(drop=True)
    
#     for i in range(1, len(trains)):
#         # we don't need the test set for the dataset's we aren't testing on
#         trains[i] = pd.concat( [trains[i],tests[i]], axis = 0 )
        
    
    eval_func = build_sharing_encoder_eval_func(train, 
                                                valid, 
                                                test,
                                                all_trains=trains, 
                                                all_valids=vals,
                                                model_name = model_name, 
                                                is_marker=is_marker
                                                )
    
    #selectthe parameters based on the model_name
    if model_name=='DAE':
        parameters=DAE_parameters
    if model_name=='SAE':
        parameters=SAE_parameters
    if model_name=='VAE':
        parameters=VAE_parameters

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
    
    return([model_name, data_type, data_name, best_parameters, best_values])


