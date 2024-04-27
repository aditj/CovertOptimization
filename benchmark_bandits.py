
### Federated Learning Setup where the learner controls the learning based on different policies  

import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm
N_rounds  = 25
 
I = 1
A = 2
U = I*A
M = 10
estimates = np.linspace(1,0,M)
O=2
succ_prob = [0.2,
             0.8]    
P_O = [[
    0.8,0.2
],
[
    0.2,0.8
]]

def sigmoid_policy(x,thresholds,tau = 0.01):
    m = x%M
    o = int(x//M)
    action = 1/(1+ np.exp(-(m-thresholds[o])/tau))
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

def get_cost(oracle_state,learner_state,eavesdropper_estimate,action,num_queries):
    C = np.array([[
        -1,2
    ],
    [
        -0.1,0
    ]])
    return C[int(oracle_state),int(action)]/(learner_state+1)**2 
    if action == 0:
        return (oracle_state+1)**2/(learner_state+1)**2 * np.log(num_queries/(num_queries+1))
    else:
        return (learner_state+1)**2/(oracle_state+1)**2 * np.log((num_queries+(1/eavesdropper_estimate))/(num_queries+1))

def get_terminal_cost(learner_state):
    return 2*learner_state**4

def get_approx_cost(P_O,policy,O,M,U,N,N_mc):
    costs = np.zeros(N_mc)
    for mc in range(N_mc):
        oracle_state = np.random.choice(O)
        learner_state = M-1
        cost_ = 0
        eavesdropper_estimate = 0.5
        for t in range(N):
            state = M*oracle_state + learner_state
            action = policy(state) 
            if learner_state == 0:
                action = 0
            In = action
            if action == 1:
                if np.random.rand() < succ_prob[oracle_state]:  
                    learner_state -= action
            eavesdropper_estimate = update_estimate(eavesdropper_estimate,action,In)
            eavesdropper_estimate = min(max(eavesdropper_estimate,0.01),0.99)

            oracle_state = np.random.choice(O,p=P_O[oracle_state])
            cost_ += get_cost(oracle_state,learner_state,eavesdropper_estimate,action,t)
        cost_ += get_terminal_cost(learner_state)
        costs[mc] = cost_
    return np.mean(costs)

TRAIN_BANDIT = 0
N_MC = 1000
single_threshold = list(range(0,M))
## create mesh grid for thresholds (O*A)
thresholds = np.meshgrid(*[single_threshold]*O)
costs = np.zeros(M**(O))
if TRAIN_BANDIT:
    for arm in tqdm.tqdm(range(M**(O))):
        thresholds_policy = np.array([thresholds[i].flatten()[arm]  for i in range(O)])
        sig_pol = lambda x: sigmoid_policy(x,thresholds_policy,0.01)
        costs[arm] = get_approx_cost(P_O,sig_pol,O,M,U,N_rounds,N_MC)
    np.save('costs.npy',costs)

TRAIN_SA = 1
thresholds = np.ones((O,1)).flatten()*M//2
N_iter = 10000
delta = 0.1
step_size = 0.001
N_MC = 100
if TRAIN_SA:
    for i in range(N_iter):

        selected_thresholds = np.random.choice(O)
        threshold_plus = thresholds.copy()
        threshold_minus = thresholds.copy()
        threshold_plus[selected_thresholds] = thresholds[selected_thresholds] + delta
        threshold_minus[selected_thresholds] = thresholds[selected_thresholds] - delta        
        sig_pol_plus = lambda x: sigmoid_policy(x,threshold_plus,0.01)
        sig_pol_minus = lambda x: sigmoid_policy(x,threshold_minus,0.01)
        costs_plus = get_approx_cost(P_O,sig_pol_plus,O,M,U,N_rounds,N_MC)
        costs_minus = get_approx_cost(P_O,sig_pol_minus,O,M,U,N_rounds,N_MC)
        grad_approx = (costs_plus - costs_minus)/(2*delta)
        thresholds[selected_thresholds] -= step_size*grad_approx
        step_size = step_size*0.99999
       
        print(costs_plus)
    np.save('thresholds.npy',thresholds)

 
costs = np.load('costs.npy')
print(costs)
print(np.min(costs))
threshold_idx = np.argmin(costs)
print(threshold_idx)
thresholds_policy = np.array([thresholds[i].flatten()[threshold_idx] for i in range(O)])
print(thresholds_policy)
# for x in range(M*O):
#     print(sigmoid_policy(x,thresholds_policy,0.01))
