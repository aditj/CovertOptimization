import torch
from client import Client
from models import BERTClass,CNNBERTClass
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
## add date and time to log file
import datetime
now = datetime.datetime.now()


class Learner():
    def __init__(self,n_clients,n_communications,parameters,n_classes,client_parameters,generate_policy = False,greedy_policy = False):
        ### FL Server Related ###
        self.n_clients = n_clients
        self.global_parameters = parameters.copy()
        self.aggregated_parameters = parameters.copy()
        self.n_communications = n_communications
        self.n_classes = n_classes
        
      
        ## Create file to write log

        ### FL Client Related ###
        self.clients = []
        self.client_parameters = client_parameters
        self.model = BERTClass
        self.initialize_clients()
        self.n_batch_per_client = self.clients[0].n_batch_per_client
        self.train_batch_size = self.clients[0].train_batch_size
        self.count_learning_queries = 0

    def train(self,round_idx):
       
                    ### Zero the aggregated parameters and loss
        self.zero_aggregated_parameters() # zero the aggregated parameters
        self.aggregated_loss = 0 # zero the aggregated loss
        self.aggregated_f1 = 0
        self.aggregated_balanced_accuracy = 0
        self.gradient_norm = 0
        ### Train the clients and aggregate the parameters
        for j in range(self.n_clients): # for each client
            self.gradient_norm += self.clients[j].train(self.global_parameters,self.train_batch_size,round_idx) # train the client
            self.add_parameters(self.clients[j].get_parameters()) # add the parameters to the aggregated parameters
            evaluations = self.clients[j].evaluate(self.clients[j].get_parameters(),self.train_batch_size) 
            self.aggregated_loss += evaluations[0]
            self.aggregated_f1 += evaluations[1]
            self.aggregated_balanced_accuracy += evaluations[2]
        self.gradient_norm = self.gradient_norm/self.n_clients
        self.divide_parameters(len(self.clients)) # divide the aggregated parameters by the number of clients
        self.aggregated_loss/=len(self.clients) 
        self.aggregated_f1/=len(self.clients)
        self.aggregated_balanced_accuracy/=len(self.clients)
        self.assign_global_parameters(self.aggregated_parameters) # assign the global parameters to the aggregated parameters
        ### write in self.file
    
   
    def get_gradient_state(self):
        gradient_thresholds = [0,0.1,0.2,0.3,0.4]
        return np.digitize(self.gradient_norm,gradient_thresholds)
    def get_parameters(self):
        return self.global_parameters
    def add_parameters(self,parameters):
        for layer in self.aggregated_parameters:
            self.aggregated_parameters[layer] += parameters[layer]  # add the parameters to the aggregated parameters
    def divide_parameters(self,divisor):
        for layer in self.aggregated_parameters:
            self.aggregated_parameters[layer] = self.aggregated_parameters[layer]/divisor # divide the aggregated parameters by the divisor
    def assign_global_parameters(self,parameters):
        for layer in self.global_parameters:
            self.global_parameters[layer] = parameters[layer] # assign the global parameters to the aggregated parameters
    def zero_aggregated_parameters(self):
        for layer in self.aggregated_parameters:
            self.aggregated_parameters[layer] = torch.zeros_like(self.aggregated_parameters[layer])
    def is_equal_parameters(self,parameters):
        for layer in self.obfuscating_parameters:
            if not torch.equal(self.obfuscating_parameters[layer],parameters[layer]):
                return False
        return True