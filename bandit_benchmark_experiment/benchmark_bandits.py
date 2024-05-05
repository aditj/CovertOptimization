
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
O = 2
succ_prob = [0.6,
             0.8]    
P_O = [[
    0.8,0.2
],
[
    0.2,0.8
]]

def sigmoid_policy(x,thresholds,tau = 0.0001):
    m = x%M
    o = int(x//M)

    action = 1/(1+ np.exp(-(m-thresholds[o])/tau))
    if m == thresholds[o]:
        action = 0
    action = min(max(action,0),U-1)
    return action

def plot_sigmoid_policy(sa_parameters):
    policy = np.zeros((M*O))
    for m in range(M):
        for o in range(O):
            policy[o*M+m] = sigmoid_policy(m+M*o,sa_parameters)
    plt.figure()
    plt.plot(policy)
    plt.xlabel('State')
    plt.ylabel('Action')
    plt.savefig('plots/sigmoid_policy.png')

def update_estimate(currentest,u,num_queries):
    currentest_eav = ((num_queries-1)*currentest + u)/num_queries
    return currentest_eav

def get_cost(oracle_state,learner_state,eavesdropper_estimate,action,num_queries):
    # C = np.array([[
    #     -1,2
    # ],
    # [
    #     -0.1,0
    # ]])
    # return C[int(oracle_state),int(action)]/((learner_state+1)**(0.5))
    cost = 0
    if action == 0:
        cost =  ((oracle_state+1)**(2)/((learner_state+1)**(1))) * np.log((num_queries*eavesdropper_estimate)/((num_queries+1)*eavesdropper_estimate))
    else:
        cost = ((learner_state+1)**(1))/(oracle_state+1)**(2)* np.log((num_queries*eavesdropper_estimate+1)/((num_queries+1)*(eavesdropper_estimate)))*1/(1+np.exp((-eavesdropper_estimate+0.5)/0.01))
    if action == 1 and cost<0:
        print("Help!")
   # print(action,oracle_state,learner_state,eavesdropper_estimate,cost)
    return cost

def get_terminal_cost(learner_state):
    return learner_state**(2)

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

            oracle_state = np.random.choice(O,p=P_O[oracle_state])
            cost_ += get_cost(oracle_state,learner_state,eavesdropper_estimate,action,t+1)
            eavesdropper_estimate = update_estimate(eavesdropper_estimate,action,t+1)
            eavesdropper_estimate = min(max(eavesdropper_estimate,0.1),0.9)

        cost_ += get_terminal_cost(learner_state)
        costs[mc] = cost_
    return np.mean(costs)

lambda_ = 0.01
TRAIN_BANDIT = 0
N_MC = 500
single_threshold = list(range(0,M))
## create mesh grid for thresholds (O*A)
thresholds = np.meshgrid(*[single_threshold]*O)
costs = np.zeros(M**(O))
if TRAIN_BANDIT:
    for arm in tqdm.tqdm(range(M**(O))):
        thresholds_policy = np.array([thresholds[i].flatten()[arm]  for i in range(O)])
        sig_pol = lambda x: sigmoid_policy(x,thresholds_policy,0.0001)
        costs[arm] = get_approx_cost(P_O,sig_pol,O,M,U,N_rounds,N_MC)
    np.save(f'parameters/costs_{lambda_}.npy',costs)

TRAIN_SA = 0
thresholds = np.ones((O,1)).flatten()*M
N_iter = 1000
delta = 2
step_size = 0.8
N_MC = 200
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
        step_size = step_size*0.999
        delta *= 0.999
        print(costs_plus,thresholds)
        thresholds = np.maximum(np.minimum(thresholds,M-1),0)
    np.save(f'parameters/thresholds_{lambda_}.npy',thresholds)

thresholds = np.load(f'parameters/thresholds_{lambda_}.npy')

print(thresholds)
print(get_approx_cost(P_O,lambda x: sigmoid_policy(x,thresholds,0.0001),O,M,U,N_rounds,10000))
costs = np.load(f'parameters/costs_{lambda_}.npy')
print(costs)
print(np.min(costs))
threshold_idx = np.argmin(costs)
print(threshold_idx)
thresholds = np.meshgrid(*[single_threshold]*O)

thresholds_policy = np.array([thresholds[i].flatten()[threshold_idx] for i in range(O)])
print(thresholds_policy)
print(get_approx_cost(P_O,lambda x: sigmoid_policy(x,thresholds_policy,0.0001),O,M,U,N_rounds,10000))
# for x in range(M*O):
#     print(sigmoid_policy(x,thresholds_policy,0.01))
