# Transfer Learning for Gut Microbiome Analyss

This repo contains my project for the Machine Learning for Functional Genomics class. The goal is to explore how transfer learning techniques such as hard parameter sharing and metalearning can improve our ability to detect signals in gut microbiome datasets. This is done by extending the results of DeepMicro (https://www.nature.com/articles/s41598-020-63159-5), a study applying deep learning methods to 6 different collections of fecal gut samples, demontrating that signals can be improved by using autoencoders to reduce the datasets' dimension. We apply transfer learning on feedforward neural networks, in addition to their autoencoder structures, to determine if transfer learning techniques can lead to measurable improvements in predictive performance. 

Setting up the Environment
-------------------------
Most of the packages used do not have any specific requirements. The main exceptions are the learn2learn package, whose version on pypi currently does not offer the LightningMAML functions we use in this analysis. As a workaround, we are using a submodule to call their functions. We do require specific pytorch and pytorch-lightning versions, as learn2learn has some dependencies.

```
pip install pytorch==1.7.0
pip install pytorch-lightning==1.0.2
```

Downloading the data
-------------------
The data used in our analysis is available from DeepMicro's repo. Our full repo can be cloned and populated with their data using the following commands:
```
git clone https://github.com/gaustin15/Microb_Transer_Learning.git
git clone https://github.com/minoh0201/DeepMicro.git
mkdir Microb_Transer_Learning/data
mkdir Microb_Transer_Learning/data/abundance
mkdir Microb_Transer_Learning/data/marker
unzip DeepMicro/data/abundance.zip -d Microb_Transer_Learning/data/abundance
unzip DeepMicro/data/marker.zip -d Microb_Transer_Learning/data/marker/
rm -rf DeepMicro
cd Microb_Transfer_Learning
git submodule update --init --recursive
```

Outline of the Reposity
------------------
| Folder/file | Description |
|--|--|
| `baseline` | This has the code use to create the baseline models, in addition to their training and tuning. It also contains a file outlining each model's hyperparameter space that we consider in our tuning.|
| `Parameter_Sharing` | Has the code used to train and tune our parameter sharing encoders.|
| `Metalearning` | Has the code used to train and tune our Model Agnostic Metalearning models|
| `Inference` | Contains the work used to create plots visualizing what bacterial species are important to which model. This is accomplished using saliency maps|
| `datasets.py` |This file is used to process the datasets into a pytorch Dataloader format. All analyses are run using this file's MicroDataset class|
| `Notebooks` | These are the notebooks which run all the training, store all the results, and create all the figures. They should be run in their numbered order|
| `results` | This is the folder where the numerical results from each experiment is stored, in csv format. The notebooks are designed to avoid overwriting the files wtihin this folder, so the contents of results will need to be cleared if a user wants to run the notebooks|
| `figures` | This is where all the plots and visualizations of the data and results are stored. These are the figures used to create the reports|
| `Reports` | These are the written up summaries of the project, which are the proposal, the midpoint update, and the final report.|
