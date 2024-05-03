import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from utils.funcs import sigmoid_policy,update_estimate,seed_everything

from nn import BERTClass

I = 3
A = 2
U = I*A
M = 50
N = 100
O = 3

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
        #create_datasets_clients(N_device=self.n_clients,fraction_of_data=1,batch_size = self.client_dataset_size)
        #self.agg_parameters = BERTClass(N_CLASSES).to("mps").state_dict()
        #self.initialize_clients()
        self.update_oracle_state()
    def initialize_clients(self):
        self.clients = []
        for client in range(self.n_clients):
            self.clients.append(Client(client,BERTClass(N_CLASSES)))

       
    def train(self,incentive):
        incentive_map = [0.2,0.4,0.9]
        evaluation = 0
        data_available_for_sampling = incentive_map[int(incentive)]*self.client_dataset_size
        self.total_data = np.random.uniform(data_available_for_sampling,data_available_for_sampling,self.n_participating_clients).sum()
        round_succ = self.total_data > self.data_thresholds
        return round_succ
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

def terminal_cost(learner_state,eavesdropper_estimate):
    return learner_state**2

def running_cost(action,oracle_state,learner_state,eavesdropper_estimate_prev,eavesdropper_estimate,I = 3):
    learn_action = action//I
    incentive = action%I if learn_action == 1 else (I-1-action)
    cost = 0
    if learn_action == 0:
        cost =  ((oracle_state+1)**2/((learner_state+1)**2)) * np.log(eavesdropper_estimate/eavesdropper_estimate_prev)
    else:
        cost = ((learner_state+1)**2)/(oracle_state+1)**(2)* np.log(eavesdropper_estimate/eavesdropper_estimate_prev)/(1+np.exp((-eavesdropper_estimate+0.5)/0.01))
    return cost

def approximate_finite_horizon_cost(N,N_MC,O,M,U,policy,update_estimate,get_cost,get_terminal_cost):
    costs_ = np.zeros(N_MC)
    oracle = Oracle()
    for mc in range(N_MC):
        learner_state = M-1
        eavesdropper_estimate = 0.5
        oracle.reset_oracle([0.5,0.5])
        oracle_state = oracle.oracle_state
        for t in range(N):
            state = M*oracle_state + learner_state
            action = policy(state) 
            learn_action = action//I
            incentive = action%I if learn_action == 1 else (I-1-action)
            if learner_state == 0:
                action = 0
                In = I - 1
            
            
            eavesdropper_prev = eavesdropper_estimate
            eavesdropper_estimate = update_estimate(eavesdropper_estimate,action,t+1)
            costs_[mc] += get_cost(action,oracle_state,learner_state,eavesdropper_prev,eavesdropper_estimate)
            eavesdropper_estimate = min(max(eavesdropper_estimate,0.1),0.9)
            if action == 1:
                succ_round = oracle.train(incentive)
                if succ_round:
                    learner_state -= 1
            oracle.update_client_participation()
            oracle.update_oracle_state()
            oracle_state = oracle.oracle_state
        costs_[mc] += get_terminal_cost(learner_state,eavesdropper_estimate)
    return costs_.mean()

TRAIN_SPSA = True
N_iter = 1000
delta = 2
N_MC = 100
tau = 0.2
parameters = np.ones((O,U))*25
step_size = 0.05
if TRAIN_SPSA:
    for iter_ in range(N_iter):
        parameters_change = np.random.choice([0,1],size = (O,U),p = [0.5,0.5])
        parameters_plus = parameters + delta*parameters_change
        parameters_minus = parameters - delta*parameters_change
        costs_plus = approximate_finite_horizon_cost(N,N_MC,O,M,U,lambda x: sigmoid_policy(x,parameters_plus,tau),update_estimate,running_cost,terminal_cost)
        costs_minus = approximate_finite_horizon_cost(N,N_MC,O,M,U,lambda x: sigmoid_policy(x,parameters_minus,tau),update_estimate,running_cost,terminal_cost)
        gradient = np.zeros((O,U))
        gradient[parameters_change == 1] = (costs_plus - costs_minus)/(2*delta)
        parameters = parameters - step_size*gradient

        tau = tau*0.99
        delta = delta*0.99
        step_size = step_size*0.99

        print(parameters,costs_plus)

