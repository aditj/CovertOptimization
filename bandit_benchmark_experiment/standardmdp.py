import mdptoolbox
import numpy as np 
M = 10
O = 2
X = O*M
P_O = [[0.8,0.2],[0.2,0.8]]
succ_prob = [0.3,0.8]
A = 2
P = np.zeros((A,X,X))
for o in range(O):
    for o_prime in range(O):
        for a in range(A):
            if a == 0: 
                succ_prob_a = 0
            else:
                succ_prob_a = succ_prob[o]
            for m in range(M):
                P[a,o*M+m,o_prime*M+m-1] = succ_prob_a*P_O[o][o_prime]
                P[a,o*M+m,o_prime*M+m] = (1-succ_prob_a)*P_O[o][o_prime]

R = np.zeros((X,A))

def get_terminal_cost(learner_state):
    return 0.5*learner_state
def get_approx_cost(P_O,policy,O,M,U,N,N_mc):
    costs = np.zeros(N_mc)
    for mc in range(N_mc):
        oracle_state = np.random.choice(O)
        learner_state = M-1
        cost_ = 0
        for t in range(N):
            state = M*oracle_state + learner_state
            print(state)
            action = policy[state,t]
            if learner_state == 0:
                action = 0
            In = action
            if action == 1:
                if np.random.rand() < succ_prob[oracle_state]:  
                    learner_state -= action
   
            oracle_state = np.random.choice(O,p=P_O[oracle_state])
            cost_ += get_cost(oracle_state,learner_state,action)
        cost_ += get_terminal_cost(learner_state)
        costs[mc] = cost_
    return np.mean(costs)

def get_cost(oracle_state,learner_state,action):
    C = np.array([[
        -1,2
    ],
    [
        -0.1,0
    ]])
    return C[int(oracle_state),int(action)]/((learner_state+1)**(1.4))
for o in range(O):
    for m in range(M):
        for a in range(A):
            R[o*M+m,a] = get_cost(o,m,a)

gamma = 0.9
fh = mdptoolbox.mdp.FiniteHorizon(P, -R, 1, 25)
fh.run()
print(fh.V)
print(fh.policy)
print(get_approx_cost(P_O,fh.policy,O,M,A,25,10))