
### Benchmarking different algorithm for estimating the optimal threshold policy
import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm
N_rounds  = 25 ### Number of rounds (queries)
I = 1 
A = 2 ## Actions
U = I*A
M = 10 ## Learner states
O = 2 ## Oracle states
succ_prob = [0.2,
             0.8]     ### Success probability of different oracle states
### Transition probabilities of the oracle
P_O = [[
    0.8,0.2
],
[
    0.2,0.8
]]

def sigmoid_policy(x,thresholds,tau = 0.0001):
    '''
    Sigmoid policy 
    x: state
    thresholds: thresholds for each oracle state
    '''
    m = x%M
    o = int(x//M)

    action = 1/(1+ np.exp(-(m-thresholds[o])/tau))
    if m == thresholds[o]:
        action = 0
    action = min(max(action,0),U-1)
    return action

def plot_sigmoid_policy(sa_parameters):
    '''
    Plot the sigmoid policy
    '''

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
    ''' Update the estimate of the eavesdropper
    currentest: current estimate
    u: action
    num_queries: number of queries
    '''
    currentest_eav = ((num_queries-1)*currentest + u)/num_queries
    return currentest_eav

def get_cost(oracle_state,learner_state,eavesdropper_estimate,action,num_queries):
    ''' Get the cost of the action
    oracle_state: oracle state
    learner_state: learner state
    eavesdropper_estimate: estimate of the eavesdropper
    action: action
    num_queries: number of queries
    '''
    # C = np.array([[
    #     -1,2
    # ],
    # [
    #     -0.1,0
    # ]])
    # return C[int(oracle_state),int(action)]/((learner_state+1)**(0.5)) ### C_3
    
    cost = 0
    if action == 0:
        cost =  ((oracle_state+1)**(2)/((learner_state+1)**(1))) * np.log((num_queries*eavesdropper_estimate)/((num_queries+1)*eavesdropper_estimate)) ### Cost if action is 0
    else:
        cost = ((learner_state+1)**(1))/(oracle_state+1)**(2)* np.log((num_queries*eavesdropper_estimate+1)/((num_queries+1)*(eavesdropper_estimate))) ### Cost if action is 1
    
    return cost

def get_terminal_cost(learner_state):
    return learner_state**(2)

def get_approx_cost(P_O,policy,O,M,U,N,N_mc):
    ''' Get the approximate cost of the policy
    P_O: Transition probabilities of the oracle
    policy: policy
    O: Number of oracle states
    M: Number of learner states
    U: Number of actions
    N: Number of rounds
    N_mc: Number of Monte Carlo simulations
    '''
    costs = np.zeros(N_mc)
    for mc in range(N_mc):
        oracle_state = np.random.choice(O)
        learner_state = M-1
        cost_ = 0
        eavesdropper_estimate = 0.5
        for t in range(N):
            state = M*oracle_state + learner_state ### State
            action = policy(state) ### Action
            
            if learner_state == 0:
                action = 0
            In = action
            if action == 1:
                if np.random.rand() < succ_prob[oracle_state]:   ### Success probability
                    learner_state -= action

            oracle_state = np.random.choice(O,p=P_O[oracle_state]) ### Transition of the oracle
            cost_ += get_cost(oracle_state,learner_state,eavesdropper_estimate,action,t+1) ### Cost
            eavesdropper_estimate = update_estimate(eavesdropper_estimate,action,t+1)
            eavesdropper_estimate = min(max(eavesdropper_estimate,0.1),0.9)

        cost_ += get_terminal_cost(learner_state) ### Terminal cost
        costs[mc] = cost_
    return np.mean(costs)

#### Multi-armed bandit for estimating the optimal threshold policy
lambda_ = 1 ### parameter to distinguish different costs
TRAIN_BANDIT = 0
N_MC = 500 ### Number of Monte Carlo simulations for estimating the cost
single_threshold = list(range(0,M))
## create mesh grid for thresholds (O*A)
thresholds = np.meshgrid(*[single_threshold]*O) ### Thresholds grid
costs = np.zeros(M**(O)) ### Costs for each threshold
if TRAIN_BANDIT:
    for arm in tqdm.tqdm(range(M**(O))):
        thresholds_policy = np.array([thresholds[i].flatten()[arm]  for i in range(O)]) ### Thresholds for the policy
        sig_pol = lambda x: sigmoid_policy(x,thresholds_policy,0.0001) ### Sigmoid policy
        costs[arm] = get_approx_cost(P_O,sig_pol,O,M,U,N_rounds,N_MC)   ### Cost of the policy
    np.save(f'parameters/costs_{lambda_}.npy',costs) ### Save the costs

## SPSA for estimating the optimal threshold policy
TRAIN_SA = 0
thresholds = np.ones((O,1)).flatten()*M ## Initial thresholds
N_iter = 1000  ## Number of iterations
delta = 2 ## Delta for estimating the gradient
step_size = 0.8 ## Step size
N_MC = 200 ## Number of Monte Carlo simulations
if TRAIN_SA: 
    for i in range(N_iter):

        selected_thresholds = np.random.choice(O) ## Select a random threshold
        threshold_plus = thresholds.copy() ## Thresholds for the policy
        threshold_minus = thresholds.copy() ## Thresholds for the policy
        threshold_plus[selected_thresholds] = thresholds[selected_thresholds] + delta  ## Thresholds for the policy
        threshold_minus[selected_thresholds] = thresholds[selected_thresholds] - delta ## Thresholds for the policy
        sig_pol_plus = lambda x: sigmoid_policy(x,threshold_plus,0.01) ## Sigmoid policy
        sig_pol_minus = lambda x: sigmoid_policy(x,threshold_minus,0.01) ## Sigmoid policy
        costs_plus = get_approx_cost(P_O,sig_pol_plus,O,M,U,N_rounds,N_MC) ## Cost of the policy
        costs_minus = get_approx_cost(P_O,sig_pol_minus,O,M,U,N_rounds,N_MC) ## Cost of the policy
        grad_approx = (costs_plus - costs_minus)/(2*delta) ## Gradient approximation
        thresholds[selected_thresholds] -= step_size*grad_approx ## Update the thresholds
        step_size = step_size*0.999 ## Step size
        delta *= 0.999 ## Delta
        print(costs_plus,thresholds)
        thresholds = np.maximum(np.minimum(thresholds,M-1),0) ## Thresholds
    np.save(f'parameters/thresholds_{lambda_}.npy',thresholds) ## Save the thresholds

thresholds = np.load(f'parameters/thresholds_{lambda_}.npy') ## Load the thresholds

print(" Costs and Thresholds from SPSA: ")
print("Thresholds: ",thresholds)
print("Approximate Cost: ")
print(get_approx_cost(P_O,lambda x: sigmoid_policy(x,thresholds,0.0001),O,M,U,N_rounds,10000))


costs = np.load(f'parameters/costs_{lambda_}.npy')
print(" Costs and Thresholds from Multi-armed bandit: ")
print("Thresholds: ",thresholds)
threshold_idx = np.argmin(costs)
print(threshold_idx)
thresholds = np.meshgrid(*[single_threshold]*O)
thresholds_policy = np.array([thresholds[i].flatten()[threshold_idx] for i in range(O)])
print(thresholds_policy)
print("Approximate Cost: ")
print(get_approx_cost(P_O,lambda x: sigmoid_policy(x,thresholds_policy,0.0001),O,M,U,N_rounds,10000))
