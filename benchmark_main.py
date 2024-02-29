
### Federated Learning Setup where the learner controls the learning based on different policies  

import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm
N_MC = 100
N_rounds  = 120

I = 5
A = 2
U = I*A
M = 40
G=3
succ_prob = [0.2,0.4,0.6,0.65,0.7]
P_G = [[
    0.5,0.3,0.2
],
[
    0.2,0.5,0.3
],
[
    0.2,0.3,0.5
]
]
def sigmoid_policy(x,thresholds,tau = 0.01):
    m = x%M
    g = x//M
    action = 0
  
    for u in range(U):
        action += 1/(1+ np.exp(-(m-thresholds[g,u])/tau))
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
# sa_parameters = np.array([[0,0,0,0,0,0,0,0,0,0],
#                  [5,5,5,5,5,5,5,5,5,5],
#                  [10,10,10,10,10,10,10,10,10,10],
#                  [15,15,15,15,15,15,15,15,15,15],
#                  [20,20,20,20,20,20,20,20,20,20],
#                  ])
def plot_sigmoid_policy():
    policy = np.zeros((M*G))
    for m in range(M):
        for g in range(G):
            policy[g*M+m] = sigmoid_policy(m+M*g,sa_parameters)
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
learner_states = np.ones((N_MC, len(policies), N_rounds))*(M-1)
gradient_states = np.zeros((N_MC, len(policies), N_rounds))
RUN_EXP = True
if RUN_EXP: 
    for mc in tqdm.tqdm(range(N_MC)):
        for policy_idx,policy in enumerate(policies):
            learner_state = M-1
            gradient_state = G-1
            eavesdropper_estimate = 0.5
            In = 0
            for round_ in range(N_rounds):
                state = M*gradient_state + learner_state 
                action = policy(state)
                
                incentive = action%I
                type_query = action//I
                if learner_state == 0:
                    type_query = 0
                    incentive = I-1
                else: 
                    if policy_idx == 0:
                        if type_query == 0:
                            type_query = np.random.choice([0,1],1,p=[0.4,0.6])
                            if type_query==1:
                                incentive = I-1
                        else:
                            type_query = np.random.choice([0,1],1,p=[0,1])
                        
                
                eavesdropper_estimate = update_estimate(eavesdropper_estimate,action,In)
                eavesdropper_estimates[mc,policy_idx, round_] = eavesdropper_estimate
                In = In + incentive + 1
     
                if type_query == 0:
                    gradient_states[mc,policy_idx,round_] = gradient_state
                    learner_states[mc,policy_idx,round_] = learner_state

                    continue
                else:
                    if np.random.rand() < succ_prob[int(incentive)]:
                        learner_state = learner_state - 1
                        gradient_state = np.random.choice(range(G),1,p = P_G[int(gradient_state)])
                        successful_queries[mc,policy_idx, round_] = 1
                        learner_states[mc,policy_idx,round_] = learner_state
                        gradient_states[mc,policy_idx,round_] = gradient_state
                n_comm[mc,policy_idx, round_] = In
            
        np.save('n_comm.npy', n_comm)
        np.save('successful_queries.npy', successful_queries)
        np.save('eavesdropper_estimates.npy', eavesdropper_estimates)
        np.save('learner_state_final.npy', learner_states)
        np.save('gradient_states.npy', gradient_states)

n_comm = np.load('n_comm.npy')
successful_queries = np.load('successful_queries.npy')
eavesdropper_estimates = np.load('eavesdropper_estimates.npy')
learner_state_final = np.load('learner_state_final.npy')
gradient_states = np.load('gradient_states.npy')
print(successful_queries.sum(axis=2).mean(axis=0))
print(n_comm.sum(axis=2).mean(axis=0))
# print(gradient_states.mean(axis=0))

n_comm = n_comm.mean(axis = 0)
successful_queries = successful_queries.mean(axis = 0)
eavesdropper_estimates = eavesdropper_estimates.mean(axis = 0)
learner_state_final = learner_state_final.mean(axis = 0)
gradient_states = gradient_states.mean(axis = 0)

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
plt.plot(gradient_states[0],label = 'SA')
plt.plot(gradient_states[1],label = 'Random')
plt.plot(gradient_states[2],label = 'Greedy')
plt.xlabel('Round')
plt.ylabel('Gradient state')
plt.legend()
plt.savefig('plots/gradient_states.png')