### hyperparameters to tune for each model type
## following the hyperparameter sets outlined in DeepMicro's Supplementary materials

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

DAE_parameters = [
    {
     'name':'layer_1_size', 
     'type':'choice', 
     'values':[64, 128, 256, 512, 1024]
    },
    {
     'name':'layer_2_size', 
     'type':'choice', 
     'values':[32, 64, 128, 256, 512]
    },
    {
    'name':'classifier_model',
    'type':'choice',
    'values':['svm', 'rf']
    }
    ]

VAE_parameters = [
    {
     'name':'layer_1_size', 
     'type':'choice', 
     'values':[32, 64, 128, 256, 512]
    },
    {
     'name':'layer_2_size', 
     'type':'choice', 
     'values':[4, 8, 16]
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
     'values': [-5, -3, -1, 1, 3, 5]
    },
         {
     'name':'gamma', 
     'type':'choice', 
     'values':[-15, -13, -11, -9, -7, -5, -3, -1, 2, 23]
    }
]
    
FFNN_parameters = [
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
    'type':'choice', 
    'values':[1e-3, 1e-2, 1e-1]
    },
    {
    'name':'dropout', 
    'type':'choice', 
    'values':[.1, .3, .5]
    }
    ]
    