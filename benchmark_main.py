
### Federated Learning Setup where the learner controls the learning based on different policies  

import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm
N_MC = 100
N_rounds  = 200
 
I = 3
A = 2
U = I*A
M = 75
estimates = np.linspace(1,0,M)
O=3
succ_prob = [[0.1,0.2,0.3],
             [0.2,0.4,0.6],
             [0.3,0.6,0.9]             
             ]
P_O = [[
    0.7,0.2,0.1
],
[
    0.1,0.7,0.2
],
[
    0.1,0.2,0.7
]
]
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
sa_parameters = np.load("./mdp_parameters.npy")[-1]

sa_parameters = np.array([[10,20,60,65,80,80],
                          [0,30,50,55,60,70],
                          [0,0,0,20,35,50]])
sa_parameters = np.array([
    [65,65,65,65,65,65],
    [30,40,45,48,49,50],
    [0,0,0,0,0,0]
])
def plot_sigmoid_policy():
    policy = np.zeros((M*O))
    for m in range(M):
        for o in range(O):
            policy[o*M+m] = sigmoid_policy(m+M*o,sa_parameters)
    plt.figure()
    plt.plot(policy)
    plt.xlabel('State')
    plt.ylabel('Action')
    plt.savefig('plots/sigmoid_policy.png')
plot_sigmoid_policy()

sa_policy = lambda x: sigmoid_policy(x,sa_parameters,0.01)
policies = [sa_policy,random_policy,greedy_policy]

n_comm = np.zeros((N_MC, len(policies), N_rounds))
successful_queries = np.zeros((N_MC, len(policies), N_rounds))
eavesdropper_estimates = np.zeros((N_MC, len(policies), N_rounds))
eavesdropper_estimates_actual = np.zeros((N_MC, len(policies), N_rounds))
learner_states = np.ones((N_MC, len(policies), N_rounds))*(M-1)
oracle_states = np.zeros((N_MC, len(policies), N_rounds))
RUN_EXP = True
epsilon = 0.2
if RUN_EXP: 
    for mc in tqdm.tqdm(range(N_MC)):
        for policy_idx,policy in enumerate(policies):
            learner_state = M-1
            oracle_state = np.random.choice(range(O))
            eavesdropper_estimate = 0.5
            In = 0
            for round_ in range(N_rounds):
                state = M*oracle_state + learner_state 
                action = policy(state)
                
                incentive = action%I
                type_query = action//I
                if learner_state == 0:
                    type_query = 0
                    incentive = I-1
                    action = type_query*I + incentive
                
                
                if type_query == 0:
                    learner_states[mc,policy_idx,round_] = learner_state
                else:
                    if np.random.rand() < succ_prob[int(oracle_state)][int(incentive)]:
                        learner_state = learner_state - 1
                        successful_queries[mc,policy_idx, round_] = 1
                        learner_states[mc,policy_idx,round_] = learner_state

                eavesdropper_estimate = update_estimate(eavesdropper_estimate,action,In)
                eavesdropper_estimates[mc,policy_idx, round_] = eavesdropper_estimate
                In = In + incentive + 1
                oracle_state = np.random.choice(range(O),p = P_O[int(oracle_state)])
                oracle_states[mc,policy_idx,round_] = oracle_state

                n_comm[mc,policy_idx, round_] = In
            
        np.save('n_comm.npy', n_comm)
        np.save('successful_queries.npy', successful_queries)
        np.save('eavesdropper_estimates.npy', eavesdropper_estimates)
        np.save('learner_state_final.npy', learner_states)
        np.save('oracle_states.npy', oracle_states)

n_comm = np.load('n_comm.npy')
successful_queries = np.load('successful_queries.npy')
eavesdropper_estimates = np.load('eavesdropper_estimates.npy')
learner_state_final = np.load('learner_state_final.npy')
oracle_states = np.load('oracle_states.npy')
print(successful_queries.sum(axis=2).mean(axis=0))
print(n_comm.sum(axis=2).mean(axis=0))
# print(gradient_states.mean(axis=0))

n_comm = n_comm.mean(axis = 0)
successful_queries = successful_queries.mean(axis = 0)
eavesdropper_estimates = eavesdropper_estimates.mean(axis = 0)
learner_state_final = learner_state_final.mean(axis = 0)
oracle_states = oracle_states.mean(axis = 0)

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
plt.plot(oracle_states[0],label = 'SA')
plt.plot(oracle_states[1],label = 'Random')
plt.plot(oracle_states[2],label = 'Greedy')
plt.xlabel('Round')
plt.ylabel('Gradient state')
plt.legend()
plt.savefig('plots/oracle_states.png')