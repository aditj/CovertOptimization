#### File: main_exp_incentive.py
#### Main Experiment to compare the performance of different policies in the presence of an eavesdropper

#### Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
from client import Client
import tqdm
from create_datasets import create_datasets_clients
from nn import BERTClass
from oracle import Oracle
from utils.funcs import sigmoid_policy,update_estimate,seed_everything

#### Setting up the experiment
N_MC = 100 ## Number of Monte Carlo simulations
N_rounds  = 100 ## Number of rounds
N_CLASSES = 1 ## Number of classes

I = 3 ## Number of incentives
A = 2 ## Learn Or Obfuscate
U = I*A ## Total number of actions
M = 50 ## Number of learner states
O = 3 ## Number of oracle states

random_policy = lambda x: np.random.randint(0,U) ## Random policy
greedy_policy = lambda x: U-1 ## Greedy policy
sa_parameters = np.load("parameters/spsa_parameters.npy")
# np.array([ ### toy parameters
#     [50,50,30,20,50,50],
#     [15,15,15,15,15,15],
#     [0,0,0,0,40,50]
# ])
sa_parameters_constant_incentivation = np.load("parameters/spsa_parameters_constant_incentivization.npy")
# np.array([ ### toy parameters
#     [50,50,50,50,50,50],
#     [15,15,15,15,15,15],
#     [0,0,0,0,0,0]
# ])

sa_policy = lambda x: sigmoid_policy(x,sa_parameters,0.01) ## Sigmoid policy
sa_policy_constant_incentivation = lambda x: sigmoid_policy(x,sa_parameters_constant_incentivation,0.01) ### Sigmoid policy with constant incentivization

policies = [sa_policy,random_policy,greedy_policy,sa_policy_constant_incentivation] ## List of policies
base_nn = BERTClass(N_CLASSES).to('mps') ## Base Neural Network

n_comm = np.zeros((N_MC, len(policies), N_rounds)) ## Number of communications 
successful_queries = np.zeros((N_MC, len(policies), N_rounds)) ## Number of Successful queries
eavesdropper_estimates = np.zeros((N_MC, len(policies), N_rounds)) ## Eavesdropper estimates (posterior probability)
learner_states = np.ones((N_MC, len(policies), N_rounds))*(M-1) ## Learner states
oracle_states = np.zeros((N_MC, len(policies), N_rounds)) ## Oracle states
succ_rates = np.zeros((O,U)) ## Success rates
oracle_state_occurences = np.zeros((O,U)) ## Oracle state occurences
evaluations = np.zeros((N_MC, len(policies), N_rounds)) ## Evaluations (Learner)
eavesdropper_evaluations = np.zeros((N_MC, len(policies), N_rounds)) ## Evaluations (Eavesdropper)

create_datasets_clients(N_device=35,fraction_of_data=1,batch_size = 600, create_eavesdropper = True) ## Create datasets (for eavesdropper)
eavesdropper_client = Client(1000,BERTClass(N_CLASSES)) ## Eavesdropper client to train (parallel SGD)
eavesdropper_client_valid = Client(2000,BERTClass(N_CLASSES)) ## Eavesdropper client to evaluate on validation set
RUN_EXP = 1 ## Boolean to run the experiment
if RUN_EXP:
    for mc in tqdm.tqdm(range(N_MC)):
        for policy_idx,policy in enumerate(policies):
            seed_everything(mc) ## Seed everything for reproducibility
            learner_parameters = base_nn.state_dict() ## Initial learner parameters
            eavesdropper_parameters = base_nn.state_dict() ## Initial eavesdropper parameters
            learner_state = M-1 ## Initial learner state
            eavesdropper_estimate = 0.5 ## Initial eavesdropper estimate
            oracle = Oracle() ## Oracle
            In = 0 ## Total Incentive
            for round_ in range(N_rounds):
                oracle_state = oracle.oracle_state ## get oracle state
                state = M*oracle_state + learner_state  ## get state
                action = policy(state) ## get action
                incentive = action%I ## get incentive and type of query
                type_query = action//I ## get type of query

                ## if type of query is 0, then the incentive is subtracted from the total incentive 
                ### (to maintain the order of actions)
                if type_query == 0:
                    incentive = I - incentive - 1
            
                ## If learner state is 0, then only type of query is 0 and incentive is maximum
                if learner_state == 0:
                    type_query = 0 ## If learner state is 0, then only type of query is 0
                    incentive = I-1 ## Incentive is maximum
                    action = type_query*I + incentive ## Action is set accordingly
                
                ### If type of query is 0, then the learner uses the parallel SGD to train the model
                if type_query == 0: ## Obfuscation query
                    learner_states[mc,policy_idx,round_] = learner_state
                    eavesdropper_client.train(400)
                    
                else:
                    ### If type of query is 1, then the learner queries the oracle with learner parameters
                    succ_round,evaluation = oracle.train(incentive,learner_parameters,return_only_succ = True)
                    
                    succ_rates[int(oracle_state),int(action)] += succ_round ## Update success rates
                    oracle_state_occurences[int(oracle_state),int(action)] += 1 ## Update oracle state occurences
                    if succ_round:
                        learner_parameters = oracle.agg_parameters ## Update learner parameters
                        learner_state = learner_state - 1 ## Update learner state
                        successful_queries[mc,policy_idx, round_] = 1 ## Update successful queries
                        learner_states[mc,policy_idx,round_] = learner_state ## Update learner state
                        
                evaluations[mc,policy_idx,round_] = evaluation ## Store evaluation
                eavesdropper_estimate = update_estimate(eavesdropper_estimate,action,In) ## Update eavesdropper estimate
                
                
                if eavesdropper_estimate >0.5: ## If eavesdropper estimate is greater than 0.5, then the eavesdropper is successful
                    eavesdropper_evaluation = evaluation
                else:
                    eavesdropper_client_valid.set_parameters(eavesdropper_client.get_parameters())  ## Set eavesdropper parameters
                    eavesdropper_evaluation = eavesdropper_client_valid.evaluate(400) ## Evaluate on validation set
                eavesdropper_evaluations[mc,policy_idx,round_] = eavesdropper_evaluation ## Store eavesdropper evaluation
                eavesdropper_estimates[mc,policy_idx, round_] = eavesdropper_estimate ## Store eavesdropper estimate
                In = In + incentive + 1 ### Total Incentive update
                oracle.update_client_participation() ## Update client participation
                oracle.update_oracle_state() ## Update oracle state 

                oracle_states[mc,policy_idx,round_] = oracle_state ### oracle state store
                n_comm[mc,policy_idx, round_] = In ### incentive

                ### Code to reset oracle every x rounds
                # if round_%20 == 0:
                #     oracle.reset_oracle([0.3,0.7]) 

                # if round_%40 == 0:
                #     oracle.reset_oracle([0.7,0.3])
    #### Save the parameters
    np.save('parameters/n_comm.npy', n_comm)
    np.save('parameters/successful_queries.npy', successful_queries)
    np.save('parameters/eavesdropper_estimates.npy', eavesdropper_estimates)
    np.save('parameters/learner_state_final.npy', learner_states)
    np.save('parameters/oracle_states.npy', oracle_states)
    np.save('parameters/evaluation.npy', evaluations)
    np.save('parameters/eavesdropper_evaluations.npy', eavesdropper_evaluations)

### Load the parameters
eavesdropper_evaluation = np.load('parameters/eavesdropper_evaluations.npy')
n_comm = np.load('./parameters/n_comm.npy')
successful_queries = np.load('./parameters/successful_queries.npy')
eavesdropper_estimates = np.load('parameters/eavesdropper_estimates.npy',allow_pickle=True)
learner_state_final = np.load('./parameters/learner_state_final.npy')
oracle_states = np.load('./parameters/oracle_states.npy')
evaluations = np.load('./parameters/evaluation.npy')

### Print the results
print("Last Learner Evaluation Averaged over Monte Carlo simulations: ",evaluations.mean(axis = 0)[-1])
print("Last Eavesdropper Evaluation Averaged over Monte Carlo simulations: ",eavesdropper_evaluation.mean(axis = 0)[-1])
print("Oracle State Occurences: ",oracle_state_occurences.sum(axis=1))
print("Oracle Success Probabilities: ", succ_rates/oracle_state_occurences)

n_comm = n_comm.mean(axis = 0)
successful_queries = successful_queries.mean(axis = 0)
eavesdropper_estimates = eavesdropper_estimates.mean(axis = 0)

#### Plotting

##### Number of communications
plt.figure()
plt.plot(n_comm[0],label = 'SA')
plt.plot(n_comm[1],label = 'Random')
plt.plot(n_comm[2],label = 'Greedy')
plt.xlabel('Round')
plt.ylabel('Number of communications')
plt.legend()
plt.savefig('plots/n_comm.png')
plt.close()

##### Eavesdropper estimates
plt.figure()
plt.plot(eavesdropper_estimates[0],label = 'SA')
plt.plot(eavesdropper_estimates[1],label = 'Random')
plt.plot(eavesdropper_estimates[2],label = 'Greedy')
plt.xlabel('Round')
plt.ylabel('Eavesdropper estimate')
plt.legend()
plt.savefig('plots/eavesdropper_estimates.png')
plt.close()
 
