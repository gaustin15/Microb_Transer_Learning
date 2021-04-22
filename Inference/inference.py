####################
# This file contains functions to help understand why our functions make the predictions they do
# We do this by running saliency maps for our trained models, and plotting the top features side by side
####################
import numpy as np
import pandas as pd
from datasets import MicroDataset
from captum.attr import Saliency
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from baseline.training_functions import DAE_parameters, VAE_parameters, SAE_parameters, build_DAE_eval, make_split
from Parameter_Sharing.train_param_sharing import build_sharing_encoder_eval_func
from Metalearning.MAML import MAML_FFNN_parameters, build_MAML_FFNN_eval_func



def show_importances(impzz, feature_names, disease = 'IBD'):
    n_feats = 4
    feats = np.unique( np.hstack( [np.argsort(np.abs(imp))[::-1][:n_feats] for imp in impzz] ) )
    n_used = len(feats)
    names = ['Standard', 'Parameter Sharing', 'Metalearning']

    plot_df = pd.concat( [pd.DataFrame( {'Importance':a[feats] , 'Training Type':n_used*[names[i]], 
                                        'Taxon':feature_names[feats]} )
                  for i,a in enumerate([imp/np.std(imp) for imp in impzz])], axis = 0 )
                    # normalizing the importances across models, so it's easier to compare relative importance
        

    sns.set_theme()
    sns.set(font_scale = 1.9)
    sns.set_style("darkgrid")

    fig = plt.figure(figsize=(15, 8))
    g=sns.barplot(y = 'Taxon', x = 'Importance', hue='Training Type', data= plot_df,
                   orient = 'h')
    plt.xticks(None)

    for i in range( n_used ):
        g.text(-2.4, i + .1  ,
               ' '.join( [a.capitalize() for a in feature_names[feats[i]].split('_')]),
               color='black', ha="center", rotation = 0, size = 15)

    plt.yticks(ticks=np.arange(0, 7, 1),
               labels =  [' '] * 7,
              rotation=45)
    plt.ylabel(None)

    
    plt.legend(loc = 'upper right')
    plt.title("Most Important Features")
    plt.xlabel('Relevance for ' + disease)
    plt.xticks([-3])
    return(g)


def get_model_attribution(forward_func, test_dataset):
    dim_out = forward_func(torch.zeros(size=(1,771))).shape[1]
    ig = Saliency(forward_func)

    test_input_tensor = test_dataset.matrix
    test_input_tensor.requires_grad_()
    
    
    attrs =  [ ig.attribute(test_input_tensor, target=i) for i in range(dim_out) ]
    return(attrs)



def summarize_models(stand_out, share_out, meta_out, test_df, disease = 'IBD'):
    """
    make plot summarizing most important features for given trained models, on test dataset
    """
    models = stand_out[1][0].eval(), share_out[1][0].eval(), meta_out[1].eval()
    # assuming we're working with enocders in standard/sharing
    forward_funcs = [lambda x: models[0].model.encode(x),
                     lambda x: models[1].model.encode(x),
                     lambda x: models[2](x),
                    ]
    
    test_dataset = MicroDataset(test_df)
    
    #get attribution for each test input
    attribz = [get_model_attribution(f, test_dataset) for f in forward_funcs]
    
    # get the average across all samples + output elements to get a level of feature importanct
    impzz = [torch.stack(a).abs().mean(dim=0).mean(dim=0).detach().numpy() for a in attribz]
    
    # get the names of the bacterial species
    feature_names = test_df.columns[test_df.columns.str.contains('k__')].str.split('__').str[-1]
    
    g = show_importances(impzz, feature_names=feature_names, disease = disease)
    return(g)
    
    
def build_eval_funcs(meta_ds, 
                     stand_model = 'DAE', # the model type used for the training + inference
                     share_model = 'DAE',
                     seed=0, 
                     is_marker=False, 
                     return_model=False):
    
    np.random.seed(seed) 
    #make data splits
    splits = [make_split(df) for df in meta_ds]
    #do a train/val/test split for the first dataset
    np.random.seed(seed)
    train, test = make_split(meta_ds[0])

#     train = train.sample(frac=frac) # sampling our training set
    train, valid = make_split(train)

    splits[0] = (train, valid)

    trains =  [s[0] for s in splits]
    vals =  [s[1] for s in splits]

    MAML_eval_func = build_MAML_FFNN_eval_func(train, 
                                          valid, 
                                          test,
                                          all_trains=trains, 
                                          all_valids=vals, 
                                          is_marker=is_marker,
                                          return_model=return_model
                                          )

    share_trains = pd.concat( [s[0] for s in splits], axis=0 ).reset_index(drop=True)
    share_vals = pd.concat( [s[1] for s in splits], axis=0 ).reset_index(drop=True)

    share_eval_func = build_sharing_encoder_eval_func(train, 
                                                valid, 
                                                test,
                                                all_trains=share_trains, 
                                                all_valids=share_vals, 
                                                is_marker=is_marker,
                                                model_name = share_model,
                                                return_model=return_model
                                                )

    #build evaluation functions
    stand_eval_func = build_DAE_eval(train, valid, test, return_model=return_model, model_name = stand_model)
    return(stand_eval_func, share_eval_func, MAML_eval_func, test )
    
    
    
def get_inference(best_stand_params,  #parameters
                  best_share_params,
                  best_meta_params, 
                  meta_ds,  # datasets --> index [0] has the main dataset of interest. Using same group for both sharing + MAML
                  stand_model = 'DAE', # the model type used for the training + inference
                  share_model = 'DAE', 
                  dataset_name = 'IBD',  #this is only used for the title of the plot
                  seed = 0
                 ):
        np.random.seed(seed)
        
        #build evaluation functions
        stand, share, maml, test_df = build_eval_funcs(meta_ds, 
                                                       stand_model,
                                                       share_model,
                                                       return_model=True)
        # theirs the models, get their trained versions to make saliency maps
        stand_out = stand(best_stand_params)
        share_out = share( best_share_params )
        meta_out = maml(best_meta_params)

        # make the saliency map, plot them in grouped barplot
        g = summarize_models(stand_out, share_out, meta_out, test_df, disease = dataset_name)
        return(g)
    
    
    
    
    
    
    
    
    