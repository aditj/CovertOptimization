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