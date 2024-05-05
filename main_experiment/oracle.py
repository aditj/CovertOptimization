### Importing Libraries
import numpy as np
from nn import BERTClass
from client import Client
from create_datasets import create_datasets_clients
N_CLASSES = 1 ## Number of classes (binary classification)
import torch

class Oracle():
    '''
    Oracle class
    '''
    def __init__(self,dont_initialize_clients = False):
        self.O = 3 ## Number of oracle states
        self.I = 3 ## Number of incentives
        self.oracle_states_thresholds = [0,16,21] ## Oracle state thresholds
        self.n_clients = 35 ## Number of clients
        self.participation_prob = [[0.8,0.2],[0.2,0.8]] ## Participation probabilities
        self.client_dataset_size = 600 ## Client dataset size
        self.data_thresholds = 4500 ## Data thresholds
        self.initial_probs = [[0.3,0.7],[0.5,0.5],[0.7,0.3]] ## Initial probabilities
        self.initial_prob = self.initial_probs[np.random.choice([0,1,2])] ## Initial probability
        self.participation_client  = np.random.choice([0,1],self.n_clients, p=self.initial_prob) ## Participation of clients
        self.n_participating_clients = sum(self.participation_client) ## Number of participating clients
        self.agg_parameters = BERTClass(N_CLASSES).to("mps").state_dict()
        ### Initialize clients
        if not dont_initialize_clients:
            self.initialize_clients()
            create_datasets_clients(N_device=self.n_clients,fraction_of_data=1,batch_size = self.client_dataset_size)
       
        self.update_oracle_state() ## Update oracle state
    def initialize_clients(self):
        '''
        Function to initialize clients
        '''
        self.clients = []
        for client in range(self.n_clients):
            self.clients.append(Client(client,BERTClass(N_CLASSES)))

       
    def train(self,incentive,parameters,return_only_succ = False):
        '''
        Function to train the oracle
        incentive: Incentive
        parameters: Parameters
        return_only_succ: Boolean to return only successful round
        '''
        incentive_map = [0.2,0.4,0.9] ## Incentive map (data available for sampling)
        evaluation = 0 ## Evaluation
        data_available_for_sampling = incentive_map[int(incentive)]*self.client_dataset_size ## Data available for sampling
        self.total_data = np.random.uniform(data_available_for_sampling,data_available_for_sampling,self.n_participating_clients).sum() ## Total data from clients
        round_succ = self.total_data > self.data_thresholds ## Round successful if total data is greater than data threshold
        if return_only_succ:
            return round_succ,0 ## Return only successful round
        if round_succ:
            ### Train loop 
            self.zero_aggregated_parameters()
            # self.participation_client = np.zeros(self.n_clients,dtype = int)
            # self.participation_client[np.random.choice(self.n_clients,2)] = 1

            # self.n_participating_clients = 2
            for client in np.nonzero(self.participation_client)[0]: ## Loop over clients
                self.clients[client].train(data_available_for_sampling) ## Train client
                self.add_parameters(self.clients[client].get_parameters()) ## Add parameters
            self.divide_parameters(self.n_participating_clients) ## Divide parameters
            for client in np.nonzero(self.participation_client)[0]: ## Loop over clients
                self.clients[client].set_parameters(self.agg_parameters) ## Set parameters
                evaluation += self.clients[client].evaluate(data_available_for_sampling)    ## Evaluate client
            evaluation = evaluation/self.n_participating_clients   ## Average evaluation
            
        return round_succ,evaluation ## Return round successful and evaluation
    def evaluate(self,parameters):
        return 0
    def divide_parameters(self,n):
        for layer in self.agg_parameters:
            self.agg_parameters[layer] = self.agg_parameters[layer]/n
        
    def zero_aggregated_parameters(self):
        for layer in self.agg_parameters:
            self.agg_parameters[layer] = torch.zeros_like(self.agg_parameters[layer])
    def add_parameters(self,parameters):
        for layer in self.agg_parameters:
            self.agg_parameters[layer] += parameters[layer]

    def reset_oracle(self,initial_prob=None):
        self.initial_probs = [[0.3,0.7],[0.5,0.5],[0.7,0.3]]

        if initial_prob is not None:
            self.initial_prob = initial_prob
        else:
            self.initial_prob = self.initial_probs[np.random.choice([0,1,2])]
        self.participation_client  = np.random.choice([0,1],self.n_clients, p=self.initial_prob)
        self.update_oracle_state()
    
   
    def update_client_participation(self):
        for client in range(self.n_clients):
            self.participation_client[client] = np.random.choice([0,1],p=self.participation_prob[self.participation_client[client]])
        self.n_participating_clients = sum(self.participation_client)
    def update_oracle_state(self):
        self.oracle_state =  np.digitize(self.n_participating_clients,self.oracle_states_thresholds)-1
