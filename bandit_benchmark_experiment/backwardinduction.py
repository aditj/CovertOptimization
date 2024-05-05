### This file is used to run the backward induction algorithm on the MDP model 
### Import the necessary libraries
import mdptoolbox
import numpy as np 
M = 10 ## Number of learner states
O = 2 ## Number of oracle states
X = O*M ## Number of states
P_O = [[0.8,0.2],[0.2,0.8]] ## Transition probabilities of the oracle
succ_prob = [0.3,0.8] ## Success probabilities
A = 2 ## Number of actions
P = np.zeros((A,X,X)) ## Transition probabilities
for o in range(O):
    for o_prime in range(O):
        for a in range(A):
            if a == 0: 
                succ_prob_a = 0 ### probability of success if action is 0
            else: 
                succ_prob_a = succ_prob[o] ### Transition probability
            for m in range(M):
                P[a,o*M+m,o_prime*M+m-1] = succ_prob_a*P_O[o][o_prime] ## Success probabilities
                P[a,o*M+m,o_prime*M+m] = (1-succ_prob_a)*P_O[o][o_prime] ## Failure probabilities
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
        oracle_state = np.random.choice(O) ## Oracle state
        learner_state = M-1
        cost_ = 0
        for t in range(N):
            state = M*oracle_state + learner_state
            
            action = policy[state,t] ### Action
            if learner_state == 0:
                action = 0
            In = action
            if action == 1:
                if np.random.rand() < succ_prob[oracle_state]:   ### Success probability
                    learner_state -= action
   
            oracle_state = np.random.choice(O,p=P_O[oracle_state])
            cost_ += get_cost(oracle_state,learner_state,action)
        cost_ += get_terminal_cost(learner_state) ## Terminal cost
        costs[mc] = cost_
    return np.mean(costs)
R = np.zeros((X,A)) ## Reward function

def get_terminal_cost(learner_state):
    return 0.5*learner_state

def get_cost(oracle_state,learner_state,action):
    ### C3 cost function
    C = np.array([[
        -1,2
    ],
    [
        -0.1,0
    ]])
    return C[int(oracle_state),int(action)]/((learner_state+1)**(0.5))
for o in range(O):
    for m in range(M):
        for a in range(A):
            R[o*M+m,a] = get_cost(o,m,a) ## Reward function

gamma = 0.9 ## Discount factor
fh = mdptoolbox.mdp.FiniteHorizon(P, -R, 1, 25) ## Finite horizon MDP
fh.run() ## Run the MDP
print(fh.V) ## Value function
print(fh.policy) ## Optimal policy
print(get_approx_cost(P_O,fh.policy,O,M,A,25,10)) ## Approximate cost of the policy