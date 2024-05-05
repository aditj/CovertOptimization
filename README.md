### Code for Covert Optimization using Structured Reinforcement Learning

#### Directory Structure
The repository is organized as follows:
- ``` main_experiment/```: Directory to store files for the main experiment in the paper. Contains a README file with instructions to run the code and preprocess the dataset
- ```bandit_benchmark_experiment/```: Directory containing files to run the bandit benchmark experiment. Contains a README explaining the different files
- ```archive/```: Directory to store old and misc (non-related) files; Does not have a README file.

The code is written in Python 3.7. The required libraries are:
- numpy
- pandas
- torch
- sklearn
- tqdm
- matplotlib

Initialize the following folders: 
- ```data/```: Directory to store the raw, cleaned, and client datasets
- - ```client_datasets/```
- ```parameters/```: Directory to store the parameters for the different experiments 
- ```plots/```


Link to Hatespeech Dataset: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge