import numpy as np
import matplotlib.pyplot as plt
import torch
from learner import Learner
import tqdm

N_MC = 1
N_rounds  = 100 


I = 3
A = 2
U = I*A
M = 50
O = 3
client_dataset_size = 800
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_everything(0)
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

        self.update_oracle_state()
    def initialize_clients(self):
        self.clients = []
        for client in range(self.n_clients):
            self.clients.append(Client(client))

       
    def train(self,incentive,parameters):
        incentive_map = [0.2,0.4,0.9]
        evaluation = 0
        data_available_for_sampling = incentive_map[int(incentive)]*self.client_dataset_size
        self.total_data = np.random.uniform(data_available_for_sampling,data_available_for_sampling,self.n_participating_clients).sum()
        round_succ = self.total_data > self.data_thresholds
        if round_succ:
            ### Train loop 
            
            for client in np.nonzero(self.participation_client)[0]:
                self.clients[client].train(parameters,data_available_for_sampling)
                evaluation += self.clients[client].evaluate(parameters,data_available_for_sampling)
            evaluation = evaluation/self.n_participating_clients    
            print(evaluation)
        return round_succ,evaluation

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

def update_estimate(currentest,u,In):
    querytype = u//I
    incentive = u%I
    if querytype == 0:
        currentest = currentest*In/(In + incentive+1)
    else:
        currentest = (currentest*In + incentive+1)/(In+incentive+1)
    return currentest

def sigmoid_policy(x,thresholds,tau = 0.01):
    m = x%M
    o = x//M
    action = 0
  
    for u in range(U):
        action += 1/(1+ np.exp(-(m-thresholds[o,u])/tau))
    action = min(max(action,0),U-1)
    return action

def update_estimate(currentest,u,In):
    querytype = u//I
    incentive = u%I if querytype == 1 else (I-1-u)
    if querytype == 0:
        currentest_eav = currentest*In/(In + incentive+1)
    else:
        currentest_eav = (currentest*In + incentive+1)/(In+incentive+1)
    return currentest_eav
random_policy = lambda x: np.random.randint(0,U)
greedy_policy = lambda x: U-1
sa_parameters = np.array([
    [50,30,10,50,50,50],
    [15,15,15,15,15,15],
    [0,0,0,0,50,50]
])
sa_parameters_constant_incentivation = np.array([
    [50,50,50,50,50,50],
    [15,15,15,15,15,15],
    [0,0,0,0,0,0]
])

sa_policy = lambda x: sigmoid_policy(x,sa_parameters,0.01)
sa_policy_constant_incentivation = lambda x: sigmoid_policy(x,sa_parameters_constant_incentivation,0.01)
def plot_sigmoid_policy(sa_parameters = sa_parameters):
    policy = np.zeros((M*O))
    for m in range(M):
        for o in range(O):
            policy[o*M+m] = sigmoid_policy(m+M*o,sa_parameters)
    plt.figure()
    plt.plot(policy)
    plt.xlabel('State')
    plt.ylabel('Action')
    plt.savefig('plots/sigmoid_policy.png')
plot_sigmoid_policy(sa_parameters)
policies = [sa_policy,random_policy,greedy_policy,sa_policy_constant_incentivation]

n_comm = np.zeros((N_MC, len(policies), N_rounds))
successful_queries = np.zeros((N_MC, len(policies), N_rounds))
eavesdropper_estimates = np.zeros((N_MC, len(policies), N_rounds))
eavesdropper_estimates_actual = np.zeros((N_MC, len(policies), N_rounds))
learner_states = np.ones((N_MC, len(policies), N_rounds))*(M-1)
oracle_states = np.zeros((N_MC, len(policies), N_rounds))
succ_rates = np.zeros((O,U))
oracle_state_occurences = np.zeros((O,U))
evaluation = np.zeros((N_MC, len(policies), N_rounds))
RUN_EXP = 1
if RUN_EXP:
    for mc in tqdm.tqdm(range(N_MC)):
        for policy_idx,policy in enumerate(policies):
            learner_state = M-1
            eavesdropper_estimate = 0.5
            oracle = Oracle()
            In = 0

            for round_ in range(N_rounds):
                oracle_state = oracle.oracle_state
                # get state
                state = M*oracle_state + learner_state 
                action = policy(state) ## get action
                incentive = action%I ## get incentive and type of query
                type_query = action//I
                if type_query == 0:
                    incentive = I - incentive - 1
                    
      
                if learner_state == 0:
                    type_query = 0
                    incentive = I-1
                    action = type_query*I + incentive
                
                
                if type_query == 0:
                    learner_states[mc,policy_idx,round_] = learner_state
                else:
                    succ_round,evaluation = oracle.train(incentive)
                    succ_rates[int(oracle_state),int(action)] += succ_round
                    oracle_state_occurences[int(oracle_state),int(action)] += 1

                    if succ_round:
                        learner_state = learner_state - 1
                        successful_queries[mc,policy_idx, round_] = 1
                        learner_states[mc,policy_idx,round_] = learner_state
                        
<<<<<<< HEAD
                evaluation[mc,policy_idx,round_] = evaluation
=======

>>>>>>> 50c8562ff1c939fafc1effd90bf9ed3159a60512
                eavesdropper_estimate = update_estimate(eavesdropper_estimate,action,In)
                eavesdropper_estimates[mc,policy_idx, round_] = eavesdropper_estimate
                In = In + incentive + 1 ### Total Incentive update

                oracle.update_client_participation()
                oracle.update_oracle_state()

                oracle_states[mc,policy_idx,round_] = oracle_state ### oracle state store
                n_comm[mc,policy_idx, round_] = In ### incentive
                if round_%20 == 0:
                    oracle.reset_oracle([0.3,0.7])

                if round_%40 == 0:
                    oracle.reset_oracle([0.7,0.3])
    np.save('parameters/n_comm.npy', n_comm)
    np.save('parameters/successful_queries.npy', successful_queries)
    np.save('parameters/eavesdropper_estimates.npy', eavesdropper_estimates)
    np.save('parameters/learner_state_final.npy', learner_states)
    np.save('parameters/oracle_states.npy', oracle_states)
    np.save('parameters/evaluation.npy', evaluation)

n_comm = np.load('parameters/n_comm.npy')
successful_queries = np.load('parameters/successful_queries.npy')
eavesdropper_estimates = np.load('parameters/eavesdropper_estimates.npy')
learner_state_final = np.load('parameters/learner_state_final.npy')
oracle_states = np.load('parameters/oracle_states.npy')


print(oracle_state_occurences.sum(axis=1))
print(succ_rates/oracle_state_occurences)

print(successful_queries.sum(axis=2).mean(axis=0))
print(n_comm[:,:,-1].mean(axis=0))

n_comm = n_comm.mean(axis = 0)
successful_queries = successful_queries.mean(axis = 0)
eavesdropper_estimates = eavesdropper_estimates.mean(axis = 0)
learner_state_final = learner_state_final.mean(axis = 0)


plt.figure()
plt.plot(n_comm[0],label = 'SA')
plt.plot(n_comm[1],label = 'Random')
plt.plot(n_comm[2],label = 'Greedy')
plt.xlabel('Round')
plt.ylabel('Number of communications')
plt.legend()
plt.savefig('plots/n_comm.png')
plt.close()

plt.figure()
plt.plot(successful_queries[0],label = 'SA')
plt.plot(successful_queries[1],label = 'Random')
plt.plot(successful_queries[2],label = 'Greedy')
plt.xlabel('Round')
plt.ylabel('Success rate')
plt.legend()
plt.savefig('plots/successful_queries.png')
plt.close()

plt.figure()
plt.plot(eavesdropper_estimates[0],label = 'SA')
plt.plot(eavesdropper_estimates[1],label = 'Random')
plt.plot(eavesdropper_estimates[2],label = 'Greedy')
plt.xlabel('Round')
plt.ylabel('Eavesdropper estimate')
plt.legend()
plt.savefig('plots/eavesdropper_estimates.png')
plt.close()
 
plt.figure()
plt.plot(learner_state_final[0],label = 'SA')
plt.plot(learner_state_final[1],label = 'Random')
plt.plot(learner_state_final[2],label = 'Greedy')
plt.xlabel('Round')
plt.ylabel('Learner state')
plt.legend()
plt.savefig('plots/learner_state_final.png')

plt.figure()
plt.hist(oracle_states.flatten())
plt.xlabel('Round')
plt.ylabel('Gradient state')
plt.legend()
plt.savefig('plots/oracle_states.png')