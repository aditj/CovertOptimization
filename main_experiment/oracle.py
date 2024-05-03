
import numpy as np
from nn import BERTClass
from client import Client
from create_datasets import create_datasets_clients
N_CLASSES = 1
import torch
class Oracle():
    def __init__(self):
        self.O = 3
        self.I = 3
        self.oracle_states_thresholds = [0,16,21]
        self.n_clients = 35
        self.participation_prob = [[0.8,0.2],[0.2,0.8]]
        self.client_dataset_size = 600
        self.data_thresholds = 4500
        self.initial_probs = [[0.3,0.7],[0.5,0.5],[0.7,0.3]]
        self.initial_prob = self.initial_probs[np.random.choice([0,1,2])]
        self.participation_client  = np.random.choice([0,1],self.n_clients, p=self.initial_prob)
        self.n_participating_clients = sum(self.participation_client)
        create_datasets_clients(N_device=self.n_clients,fraction_of_data=1,batch_size = self.client_dataset_size)
        self.agg_parameters = BERTClass(N_CLASSES).to("mps").state_dict()
        self.initialize_clients()
        self.update_oracle_state()
    def initialize_clients(self):
        self.clients = []
        for client in range(self.n_clients):
            self.clients.append(Client(client,BERTClass(N_CLASSES)))

       
    def train(self,incentive,parameters):
        incentive_map = [0.2,0.4,0.9]
        evaluation = 0
        data_available_for_sampling = incentive_map[int(incentive)]*self.client_dataset_size
        self.total_data = np.random.uniform(data_available_for_sampling,data_available_for_sampling,self.n_participating_clients).sum()
        round_succ = self.total_data > self.data_thresholds
        
        if round_succ:
            ### Train loop 
            self.zero_aggregated_parameters()
            # self.participation_client = np.zeros(self.n_clients,dtype = int)
            # self.participation_client[np.random.choice(self.n_clients,2)] = 1

            # self.n_participating_clients = 2
            for client in np.nonzero(self.participation_client)[0]:
                self.clients[client].train(data_available_for_sampling)
                self.add_parameters(self.clients[client].get_parameters())
            self.divide_parameters(self.n_participating_clients)
            for client in np.nonzero(self.participation_client)[0]:
                self.clients[client].set_parameters(self.agg_parameters)
                evaluation += self.clients[client].evaluate(data_available_for_sampling)
            evaluation = evaluation/self.n_participating_clients    
            
        return round_succ,evaluation
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
