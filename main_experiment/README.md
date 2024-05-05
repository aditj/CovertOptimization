#### Code to run the main experiment for the paper "Structured Reinforcement Learning for  Covert Optimization from Incentivized Stochastic Oracle"

##### Directory Structure
- ```main_exp.py``` : Main file to run the experiment. 
- ```spsa.py```: File containing the SPSA optimizer
- ```utils``` :
  - ```clean_datasets.py```: File to clean the dataset stored in the data folder
  - ```funcs.py```: File containing the functions to update the eavesdropper estimate and compute sigmoid policy.
- ```nn.py```: File containing the neural network architecture
- ```oracle.py```: File containing the oracle class to simulate the incentivized stochastic oracle (collection of clients) (distributed set of devices with data)
- ```client.py```: File containing the client class 
- ```create_dataset.py```: File to create the distributed dataset for the experiment

The dataset used in the experiment is the Hatespeech dataset from Kaggle. The dataset can be downloaded from the following link: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

##### Instructions to run the code
1. Download the dataset from the link provided above and store it in the data folder
2. Run the ```clean_datasets.py``` file to preprocess the dataset
3. Set the parameters in the ```main_exp.py``` file and run the file to start the experiment. The parameters are described in the file.
4. The results will be printed on the terminal. 



