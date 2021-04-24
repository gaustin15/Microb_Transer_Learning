import numpy as np
import pandas as pd
import math
import os

from datasets import MicroDataset
from torch.utils.data import DataLoader
from .pytorch_models import LitEncoderDecoder, LitDAE, LitVAE, LitFFNN

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger

from .hyperparams import *
# note - might want to adjust the ax eval structure
# ===> could instead make a build_eval_function function that creates an eval applicable to our setup

from ax import optimize

# function used for train/terst/vallid split - this mimicks their train - valid - test ratio of 64-16-20
def make_split(df, perc = .8):
    df = df.sample(frac = 1)
    neg_samps = df.loc[df.disease == 'n']
    pos_samps = df.loc[df.disease != 'n']
    
    neg_idx = round( neg_samps.shape[0] * perc )
    pos_idx = round( pos_samps.shape[0] * perc )
    
    train = pd.concat( [neg_samps.iloc[:neg_idx], 
                       pos_samps.iloc[:pos_idx] ], axis = 0 )
    
    valid = pd.concat( [neg_samps.iloc[neg_idx:], 
                       pos_samps.iloc[pos_idx:] ], axis = 0 )
    
    return(train, valid)

## k-fold training function

def run_training(train_df,
                y,
                n_folds = 5, 
                model = RandomForestClassifier(), 
                shuffle = True):
        #runs K-fold evalutaion for a given sklearn-format model, 
        # returns the aggregated roc for the datapoints from the validation setrs

        #split the positive/negative samples so we can have evenly distributed folds
        t1, t2 = train_df[y==0], train_df[y==1]

        if shuffle:
            np.random.shuffle(t1), np.random.shuffle(t2)

        t1_folds = np.linspace(0, t1.shape[0], n_folds+1, dtype = np.int64)
        t1_folds = [ (t1_folds[i], t1_folds[i+1]) for i in range(n_folds) ]

        t2_folds = np.linspace(0, t2.shape[0], n_folds+1, dtype = np.int64)
        t2_folds = [ (t2_folds[i], t2_folds[i+1]) for i in range(n_folds) ]


        all_trues = list()
        all_preds = list()

        t1_split = [t1[a[0]:a[1]] for a in t1_folds]
        t2_split = [t2[a[0]:a[1]] for a in t2_folds]

        for i in range(n_folds):

            t1_train = np.vstack( [x for j,x in enumerate(t1_split) if j!=i] )
            t2_train = np.vstack( [x for j,x in enumerate(t2_split) if j!=i] )

            t1_test = t1_split[i]
            t2_test = t2_split[i]

            train_labels = np.hstack((np.zeros(t1_train.shape[0]),
                       np.ones(t2_train.shape[0]) ) )

            test_labels = np.hstack((np.zeros(t1_test.shape[0]),
                       np.ones(t2_test.shape[0]) ) )

            full_train = np.vstack((t1_train, t2_train))
            full_test = np.vstack((t1_test, t2_test))

            rf = model#RandomForestClassifier()
            rf.fit(full_train, train_labels)

            all_trues.append( test_labels )
            all_preds.append( rf.predict_proba(full_test) )

        y = np.hstack(all_trues)
        y_hat = np.vstack(all_preds)[:, 1]

        return(roc_curve(y, y_hat), rf)
   

def rf_eval(X, y, rf_hyperparams, n_folds=5, total_trials=10):
    # find the best hyperamas vis k-fold cross-validation
    
    def evaluation_func(p):
        #the function used for the hyperparameter tuning
        model=RandomForestClassifier()
        [setattr(model, a, q) for a,q in p.items()]
        roc, model = run_training(X, y, n_folds=n_folds, model=model)
        auc_val=auc(roc[0],roc[1])
        if math.isnan(auc_val):
            auc_val=0
        return(auc_val) 
    
    
    best_parameters, best_values, experiment, model = optimize(
        parameters = rf_hyperparams,
        evaluation_function=evaluation_func,
        total_trials=total_trials, 
        minimize=False)
    
    return(best_parameters)
    
    
def svm_eval(X, y, svm_hyperparams, n_folds=5, total_trials=10):
    # find the best hyperamas vis k-fold cross-validation
    
    def evaluation_func(p):
        #the function used for the hyperparameter tuning
        model=SVC(probability=True)
        [setattr(model, a, np.power(2., q)) for a,q in p.items()]
        roc, model = run_training(X, y, n_folds=n_folds, model=model)
        auc_val=auc(roc[0],roc[1])
        if math.isnan(auc_val):
            auc_val=0
        return(auc_val) 
    
    
    best_parameters, best_values, experiment, model = optimize(
        parameters = svm_hyperparams,
        evaluation_function=evaluation_func, 
        total_trials = total_trials, 
        minimize=False)
    
    return(best_parameters)
    

### BELOW ARE THE MAIN TUNING FUNCTIONS
# TAKE A DATASET AS AN INPUT, AND RETURNS A SUMMARY OF THE BEST-PERFORMING MODEL
    
def tune_RF(dataset, 
             rf_parameters=rf_parameters, 
             is_marker=False, 
             seed=0, 
             total_trials=15):
    """
    function to get baseline for standalone random forest
    """
    
    #set a seed so that splits are the same across different model tests
    np.random.seed(seed)
    
    #make data splits
    train_df, test_df = make_split(dataset)
    
    #for rf, go straight to k-fold
    train_dataset=MicroDataset(train_df, is_marker=is_marker)
    X = train_dataset.matrix.detach().numpy()
    y = train_dataset.y.detach().numpy()
    
    def evaluation_func(p):
        #the function used for the hyperparameter tuning
        model=RandomForestClassifier(**p)
        #[setattr(model, a, q) for a,q in p.items()]
        roc, model = run_training(X, y, model=model)
        auc_val=auc(roc[0],roc[1])
        if math.isnan(auc_val):
            auc_val=0
        return(auc_val) 
    
    
    best_parameters, best_values, experiment, model = optimize(
        parameters = rf_parameters,
        evaluation_function=evaluation_func,
        total_trials=total_trials, 
        minimize=False)
    
    #make a model with the best parameters
    best_classifier=RandomForestClassifier()
    [setattr(best_classifier, a, q) for a,q in best_parameters.items()]
    
    # fit it on the full train, valid data
    best_classifier.fit(X, y)

    # Get AUC for the full setup on the test set
    test_dataset=MicroDataset(test_df, is_marker=is_marker)
    test_preds=best_classifier.predict_proba( test_dataset.matrix.detach().numpy() )
    test_roc = roc_curve( test_dataset.y.detach().numpy(), test_preds[:,1] )
    best_val=auc(test_roc[0], test_roc[1])
    
    
    data_name=dataset.index[0] 
    if is_marker==False:
        data_type='Abundance'
    else:
        data_type='Marker'
    
    return(['RF', data_type, data_name, best_parameters, best_val])



def tune_SVM(dataset, 
             svm_parameters=svm_parameters, 
             is_marker=False, 
             seed=0, 
             total_trials=15):
    """
    function to get baseline for standalone SVM
    """
    
    #set a seed so that splits are the same across different model tests
    np.random.seed(seed)
    
    #make data splits
    train_df, test_df = make_split(dataset)
    
    #for rf, go straight to k-fold
    train_dataset=MicroDataset(train_df, is_marker=is_marker)
    X = train_dataset.matrix.detach().numpy()
    y = train_dataset.y.detach().numpy()
    
    def evaluation_func(p):
        #the function used for the hyperparameter tuning
        model=SVC(probability=True)
        [setattr(model, a, np.power(2.,q)) for a,q in p.items()]
        roc, model = run_training(X, y, model=model)
        auc_val=auc(roc[0],roc[1])
        if math.isnan(auc_val):
            auc_val=0
        return(auc_val) 
    
    best_parameters, best_values, experiment, model = optimize(
        parameters = svm_parameters,
        evaluation_function=evaluation_func,
        total_trials=total_trials, 
        minimize=False)
    
    #make a model with the best parameters
    best_classifier=SVC(probability=True)
    [setattr(best_classifier, a, np.power(2.,q)) for a,q in best_parameters.items()]
    
    # fit it on the full train, valid data
    best_classifier.fit(X, y)

    # Get AUC for the full setup on the test set
    test_dataset=MicroDataset(test_df, is_marker=is_marker)
    test_preds=best_classifier.predict_proba( test_dataset.matrix.detach().numpy() )
    test_roc = roc_curve( test_dataset.y.detach().numpy(), test_preds[:,1] )
    best_val=auc(test_roc[0], test_roc[1])
    
    
    data_name=dataset.index[0] 
    if is_marker==False:
        data_type='Abundance'
    else:
        data_type='Marker'
    
    return(['SVM', data_type, data_name, best_parameters, best_val])
    
    
    
def tune_FFNN(dataset, 
             FFNN_parameters=FFNN_parameters, 
             is_marker=False, 
             seed=0, 
             total_trials=8):
    #set a seed so that splits are the same across different model tests
    np.random.seed(seed)
    #make data splits
    train_df, test_df = make_split(dataset)
    train_df, valid_df = make_split(train_df)
    
    def FFNN_eval(hyperparams):
        """this function runs a complete train/valid/testing loop
            for an FFNN model, given the specified hyperparameters, 
            it returns the AUC on the test set
            
            Defining it inside the tuning function to avoid having to manually pass the train_df and valid_df,
                    Ax tuning doesn't allow that
            """
        #Initialize the model
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

        #setr up the trainer/logger/callbacks
        checkpoint_callback=ModelCheckpoint(
                        dirpath = 'checkpoint_dir',
                        save_top_k=1,
                        verbose=False,
                        monitor='val_loss',
                        mode='min'
                        )

        tube_logger = TestTubeLogger('checkpoint_dir', 
                                    name='test_tube_logger')

        trainer = pl.Trainer(max_epochs = 500,
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
        #=====> would eat up LOTS of memory if we didn't do this
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
        return(auc_val) 

    #optimize hyperparapeter with Facebook's ax tuning 
    best_parameters, best_values, experiment, model = optimize(
        parameters = FFNN_parameters,
        evaluation_function=FFNN_eval, 
        total_trials = total_trials, 
        minimize=False)
    
    data_name=dataset.index[0] 
    if is_marker==False:
        data_type='Abundance'
    else:
        data_type='Marker'
    
    return(['FFNN', data_type, data_name, best_parameters, best_values])



    
    
def tune_SAE(dataset, 
             SAE_parameters=SAE_parameters, 
             is_marker=False, 
             seed=0, 
             total_trials=8):
    #set a seed so that splits are the same across different model tests
    np.random.seed(seed)
    #make data splits
    train_df, test_df = make_split(dataset)
    train_df, valid_df = make_split(train_df)
    
    def SAE_eval(hyperparams):
        """this function runs a complete train/valid/testing loop
            for an SAE model, given the specified hyperparameters, 
            it returns the AUC on the test set
            
            Defining it inside the tuning function to avoid having to manually pass the train_df and valid_df,
                    Ax tuning doesn't allow that
            """
        #Initialize the model
        lightning = LitEncoderDecoder(train_df, 
                                      valid_df, 
                                      layer_dim = hyperparams['layer_size'], 
                                      learning_rate = .001,
                                      batch_size=50, 
                                      is_marker=is_marker
                                      )

        #setr up the trainer/logger/callbacks
        checkpoint_callback=ModelCheckpoint(
                        dirpath = 'checkpoint_dir',
                        save_top_k=1,
                        verbose=False,
                        monitor='val_loss',
                        mode='min'
                        )

        tube_logger = TestTubeLogger('checkpoint_dir', 
                                    name='test_tube_logger')

        trainer = pl.Trainer(max_epochs = 500,
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

        #recombine the train/valid datasets for k-fold validation, as descrbed in the paper
        X = np.vstack([lightning.model.encode(lightning.train_dataset.matrix).detach().numpy(), 
                       lightning.model.encode(lightning.valid_dataset.matrix).detach().numpy()])

        y = np.hstack([lightning.train_dataset.y.detach().numpy(),
                       lightning.valid_dataset.y.detach().numpy()] )

        #set up classifier model tuning
        if hyperparams['classifier_model']=='rf':
            eval_model=rf_eval
            eval_params = rf_parameters
        elif hyperparams['classifier_model']=='svm':
            eval_model=svm_eval
            eval_params=svm_parameters

        #tune the best classifier model, using 5-fold CV ==> this also employs AX under the hood
        best_parameters=eval_model(X,y, eval_params)

        # make a classifier with teh best hyperparams
        if hyperparams['classifier_model']=='rf':
            best_classifier=RandomForestClassifier(**best_parameters)
            #[setattr(best_classifier, a, q) for a,q in best_parameters.items()]

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
        return(auc_val) 

    #optimize hyperparapeter with Facebook's ax tuning 
    best_parameters, best_values, experiment, model = optimize(
        parameters = SAE_parameters,
        evaluation_function=SAE_eval, 
        total_trials = total_trials, 
        minimize=False)
    
    data_name=dataset.index[0] 
    if is_marker==False:
        data_type='Abundance'
    else:
        data_type='Marker'
    
    return(['SAE', data_type, data_name, best_parameters, best_values])



def tune_DAE(dataset, 
             DAE_parameters=DAE_parameters, 
             is_marker=False, 
             seed=0, 
             total_trials=8):
    #set a seed so that splits are the same across different model tests
    np.random.seed(seed)
    #make data splits
    train_df, test_df = make_split(dataset)
    train_df, valid_df = make_split(train_df)
    
    def DAE_eval(hyperparams):
        """this function runs a complete train/valid/testing loop
            for an SAE model, given the specified hyperparameters, 
            it returns the AUC on the test set
            
            Defining it inside the tuning function to avoid having to manually pass the train_df and valid_df,
                    Ax tuning doesn't allow that
            """
        #Initialize the model
        lightning = LitDAE(train_df, 
                           valid_df, 
                           layer_1_dim = hyperparams['layer_1_size'],
                           layer_2_dim = hyperparams['layer_2_size'],
                           learning_rate = .001,
                           batch_size=50, 
                            is_marker=is_marker
                          )

        #setr up the trainer/logger/callbacks
        checkpoint_callback=ModelCheckpoint(
                        dirpath = 'checkpoint_dir',
                        save_top_k=1,
                        verbose=False,
                        monitor='val_loss',
                        mode='min'
                        )

        tube_logger = TestTubeLogger('checkpoint_dir', 
                                    name='test_tube_logger')

        trainer = pl.Trainer(max_epochs = 500,
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

        #recombine the train/valid datasets for k-fold validation, as descrbed in the paper
        X = np.vstack([lightning.model.encode(lightning.train_dataset.matrix).detach().numpy(), 
                       lightning.model.encode(lightning.valid_dataset.matrix).detach().numpy()])

        y = np.hstack([lightning.train_dataset.y.detach().numpy(),
                       lightning.valid_dataset.y.detach().numpy()] )

        #set up classifier model tuning
        if hyperparams['classifier_model']=='rf':
            eval_model=rf_eval
            eval_params = rf_parameters
        elif hyperparams['classifier_model']=='svm':
            eval_model=svm_eval
            eval_params=svm_parameters

        #tune the best classifier model, using 5-fold CV ==> this also employs AX under the hood
        best_parameters=eval_model(X,y, eval_params)

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
        return(auc_val) 

    #optimize hyperparapeter with Facebook's ax tuning 
    best_parameters, best_values, experiment, model = optimize(
        parameters = DAE_parameters,
        evaluation_function=DAE_eval, 
        total_trials = total_trials, 
        minimize=False)
    
    data_name=dataset.index[0] 
    if is_marker==False:
        data_type='Abundance'
    else:
        data_type='Marker'
    
    return(['DAE', data_type, data_name, best_parameters, best_values])




def tune_VAE(dataset, 
             VAE_parameters=VAE_parameters, 
             is_marker=False, 
             seed=0, 
             total_trials=8):
    #set a seed so that splits are the same across different model tests
    np.random.seed(seed)
    #make data splits
    train_df, test_df = make_split(dataset)
    train_df, valid_df = make_split(train_df)
    
    def VAE_eval(hyperparams):
        """this function runs a complete train/valid/testing loop
            for an VAE model, given the specified hyperparameters, 
            it returns the AUC on the test set
            
            Defining it inside the tuning function to avoid having to manually pass the train_df and valid_df,
                    Ax tuning doesn't allow that
            """
        #Initialize the model
        lightning = LitVAE(train_df, 
                           valid_df, 
                           layer_1_dim = hyperparams['layer_1_size'],
                           layer_2_dim = hyperparams['layer_2_size'],
                           learning_rate = .001,
                           batch_size=50, 
                            is_marker=is_marker
                          )

        #setr up the trainer/logger/callbacks
        checkpoint_callback=ModelCheckpoint(
                        dirpath = 'checkpoint_dir',
                        save_top_k=1,
                        verbose=False,
                        monitor='val_loss',
                        mode='min'
                        )

        tube_logger = TestTubeLogger('checkpoint_dir', 
                                    name='test_tube_logger')

        trainer = pl.Trainer(max_epochs = 500,
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

        #recombine the train/valid datasets for k-fold validation, as descrbed in the paper
        X = np.vstack([lightning.model.encode(lightning.train_dataset.matrix).detach().numpy(), 
                       lightning.model.encode(lightning.valid_dataset.matrix).detach().numpy()])

        y = np.hstack([lightning.train_dataset.y.detach().numpy(),
                       lightning.valid_dataset.y.detach().numpy()] )

        #set up classifier model tuning
        if hyperparams['classifier_model']=='rf':
            eval_model=rf_eval
            eval_params = rf_parameters
        elif hyperparams['classifier_model']=='svm':
            eval_model=svm_eval
            eval_params=svm_parameters

        #tune the best classifier model, using 5-fold CV ==> this also employs AX under the hood
        best_parameters=eval_model(X,y, eval_params)

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
        return(auc_val) 

    #optimize hyperparapeter with Facebook's ax tuning 
    best_parameters, best_values, experiment, model = optimize(
        parameters = VAE_parameters,
        evaluation_function=VAE_eval, 
        total_trials = total_trials, 
        minimize=False)
    
    data_name=dataset.index[0] 
    if is_marker==False:
        data_type='Abundance'
    else:
        data_type='Marker'
    
    return(['VAE', data_type, data_name, best_parameters, best_values])




def build_DAE_eval(train_df, valid_df, test_df, model_name='DAE', is_marker=False, return_model=False):
    def eval_func(hyperparams):
        """this function runs a complete train/valid/testing loop
            for an SAE model, given the specified hyperparameters, 
            it returns the AUC on the test set

            Defining it inside the tuning function to avoid having to manually pass the train_df and valid_df,
                    Ax tuning doesn't allow that
        """
        if model_name=='DAE':
            #Initialize the model
            lightning = LitDAE(train_df, 
                                   valid_df, 
                                   layer_1_dim = hyperparams['layer_1_size'],
                                   layer_2_dim = hyperparams['layer_2_size'],
                                   learning_rate = .001,
                                   batch_size=50, 
                                    is_marker=is_marker
                                  )
        if model_name=='SAE':
            lightning = LitEncoderDecoder(train_df, 
                                          valid_df, 
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

        #recombine the train/valid datasets for k-fold validation, as descrbed in the paper
        X = np.vstack([lightning.model.encode(lightning.train_dataset.matrix).detach().numpy(), 
                       lightning.model.encode(lightning.valid_dataset.matrix).detach().numpy()])

        y = np.hstack([lightning.train_dataset.y.detach().numpy(),
                       lightning.valid_dataset.y.detach().numpy()] )

        #set up classifier model tuning
        if hyperparams['classifier_model']=='rf':
            eval_model=rf_eval
            eval_params = rf_parameters
        elif hyperparams['classifier_model']=='svm':
            eval_model=svm_eval
            eval_params=svm_parameters

        #tune the best classifier model, using 5-fold CV ==> this also employs AX under the hood
        best_parameters=eval_model(X,y, eval_params)

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
