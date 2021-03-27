import numpy as np
import pandas as pd
from datasets import MicroDataset
from pytorch_models import LitEncoderDecoder

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

    
### hyperparameters to tune for each model type

SAE_parameters = [
    {
     'name':'layer_size', 
     'type':'choice', 
     'values':[32, 64, 128, 256, 512]
    },
    {
    'name':'classifier_model',
    'type':'choice',
    'values':['svm', 'rf']
    }
    ]

rf_parameters = [
      {
     'name':'n_estimators', 
     'type':'choice', 
     'values':[100, 300, 500, 700, 900]
    },
         {
     'name':'min_samples_leaf', 
     'type':'choice', 
     'values':[1, 2, 3, 4, 5]
    },
         {
     'name':'criterion', 
     'type':'choice', 
     'values':['gini', 'entropy']
    }
]

# ax doesn't like the numpy items as parameters
svm_parameters = [
          {
     'name':'C',
     'type':'choice', 
     'values': [-5, -3, -1, 1, 3, 5]#[np.power(2., a).astype(float) for a in [-5, -3, -1, 1, 3, 5]]
    },
         {
     'name':'gamma', 
     'type':'choice', 
     'values':[-15, -13, -11, -9, -7, -5, -3, -1, 2, 23]#[np.power(2., a).astype(float) for a in [-15, -13, -11, -9, -7, -5, -3, -1, 2, 23]]
    }
]
    
    
    
    

def rf_eval(X, y, rf_hyperparams, n_folds=5, total_trials=10):
    # find the best hyperamas vis k-fold cross-validation
    
    def evaluation_func(p):
        #the function used for the hyperparameter tuning
        model=RandomForestClassifier()
        [setattr(model, a, q) for a,q in p.items()]
        roc, model = run_training(X, y, n_folds=n_folds, model=model)
        return(auc(roc[0], roc[1]))
    
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
        return(auc(roc[0], roc[1]))
    
    best_parameters, best_values, experiment, model = optimize(
        parameters = svm_hyperparams,
        evaluation_function=evaluation_func, 
        total_trials = total_trials, 
        minimize=False)
    
    return(best_parameters)
    

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
        return(auc(test_roc[0], test_roc[1]) ) 

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
    
    return([data_type, data_name, best_parameters, best_values])

