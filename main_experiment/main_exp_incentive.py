import numpy as np
import matplotlib.pyplot as plt
import torch
from client import Client
import tqdm
from create_datasets import create_datasets_clients
from nn import BERTClass
from oracle import Oracle
from utils.funcs import sigmoid_policy,update_estimate
N_MC = 5
N_rounds  = 100 
N_CLASSES = 1

I = 3
A = 2
U = I*A
M = 50
O = 3

random_policy = lambda x: np.random.randint(0,U)
greedy_policy = lambda x: U-1
sa_parameters = np.array([
    [50,50,30,20,50,50],
    [15,15,15,15,15,15],
    [0,0,0,0,40,50]
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
policies = [sa_policy_constant_incentivation]
base_nn = BERTClass(N_CLASSES).to('mps')

n_comm = np.zeros((N_MC, len(policies), N_rounds))
successful_queries = np.zeros((N_MC, len(policies), N_rounds))
eavesdropper_estimates = np.zeros((N_MC, len(policies), N_rounds))
eavesdropper_estimates_actual = np.zeros((N_MC, len(policies), N_rounds))
learner_states = np.ones((N_MC, len(policies), N_rounds))*(M-1)
oracle_states = np.zeros((N_MC, len(policies), N_rounds))
succ_rates = np.zeros((O,U))
oracle_state_occurences = np.zeros((O,U))
evaluations = np.zeros((N_MC, len(policies), N_rounds))
eavesdropper_evaluations = np.zeros((N_MC, len(policies), N_rounds))
eavesdropper_client = Client(0,BERTClass(N_CLASSES))
RUN_EXP = 1
if RUN_EXP:
    for mc in tqdm.tqdm(range(N_MC)):
        
        for policy_idx,policy in enumerate(policies):
            seed_everything(mc)
            learner_parameters = base_nn.state_dict()
            eavesdropper_parameters = base_nn.state_dict()
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
                    succ_round,evaluation = oracle.train(incentive,learner_parameters)
                    
                    
                    succ_rates[int(oracle_state),int(action)] += succ_round
                    oracle_state_occurences[int(oracle_state),int(action)] += 1

                    if succ_round:
                        learner_parameters = oracle.agg_parameters
                        learner_state = learner_state - 1
                        successful_queries[mc,policy_idx, round_] = 1
                        learner_states[mc,policy_idx,round_] = learner_state
                        
                evaluations[mc,policy_idx,round_] = evaluation
                eavesdropper_estimate = update_estimate(eavesdropper_estimate,action,In)
                if eavesdropper_estimate >0.5:
                    eavesdropper_evaluation = 0
                else:
                    eavesdropper_evaluation = eavesdropper_client.evaluate(700)

                eavesdropper_estimates[mc,policy_idx, round_] = eavesdropper_estimate
                print(evaluation,eavesdropper_evaluation)
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
    np.save('parameters/evaluation_extra.npy', evaluations)
    np.save('parameters/eavesdropper_evaluations.npy', eavesdropper_evaluations)

eavesdropper_evaluation = np.load('parameters/eavesdropper_evaluations.npy')
n_comm = np.load('./parameters/n_comm.npy')
successful_queries = np.load('./parameters/successful_queries.npy')
eavesdropper_estimates = np.load('parameters/eavesdropper_estimates.npy',allow_pickle=True)
learner_state_final = np.load('./parameters/learner_state_final.npy')
oracle_states = np.load('./parameters/oracle_states.npy')
evaluations = np.load('./parameters/evaluation.npy')
print(evaluations)
print(eavesdropper_estimates)

print(oracle_state_occurences.sum(axis=1))
print(succ_rates/oracle_state_occurences)
print(successful_queries.sum(axis=2).mean(axis=0))
print(n_comm[:,:,-1].mean(axis=0))

n_comm = n_comm.mean(axis = 0)
successful_queries = successful_queries.mean(axis = 0)
eavesdropper_estimates = eavesdropper_estimates.mean(axis = 0)
learner_state_final = learner_state_final.mean(axis = 0)

print()
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