import numpy as np
### np print options
np.set_printoptions(precision=1)
G = 5 ### number of gradients states
M = 600 ### queue size
X = G*M
N = 1000 ### number of queries 
I = 5 ### Incentive
A = 2 ### Learn or Obfuscate
U = I*A ### total number of actions


P_G = [[0.4,0.3,0.1,0.1,0.1],
       [0.1,0.4,0.3,0.1,0.1],
         [0.1,0.1,0.4,0.3,0.1],
         [0.1,0.1,0.1,0.4,0.3],
         [0.1,0.1,0.1,0.2,0.5]
       ]
# P_G = np.eye(G)
succ_prob = [0.2,0.4,0.6,0.8,1]

P = np.zeros((U,X,X))
for u in range(U):
    for i in range(X):
        for j in range(X):
        
            if u < I:
                if i == j:
                    P[u][i][j]= 1
            else:
                m = i%M
                g = i//M
                m_prime = j%M
                g_prime = j//M
                incentive = u - I
                if m == 0 and m_prime == 0:
                    P[u][i][j] = P_G[g][g_prime]
                elif m_prime == m-1 and m>0:
                    P[u][i][j] = P_G[g][g_prime]*succ_prob[incentive]
                elif m_prime == m and g_prime == g and m>0:
                    P[u][i][j] = (1-succ_prob[incentive])
def cost_gradientstate(g):
    return (1+g)**2
def cost_queuestate(m):
    return (1+m)**2       
def return_cost(x,u,In,currentest):
    querytype = u//I
    incentive = u%I if querytype == 1 else (I-u)-1
    m = x%M
    g = x//M
    constant = cost_gradientstate(g)/cost_queuestate(m)
    if querytype == 0:
        
        cost = incentive*np.log(In/(In+incentive+1))/constant   
        if cost>0:
            print("Cost is positive!") 
    else:
        
        cost = incentive*np.log((In+(incentive+1)/currentest)/(In+incentive+1))/constant
        if cost<0:
            print("Cost is negative!")
    return cost*2
def return_terminalcost(x):
    m = x%M
    g = x//M
    return cost_queuestate(m)*5
def update_estimate(currentest,u,In):
    querytype = u//I
    incentive = u%I if querytype == 1 else (I-u)-1
    if querytype == 0:
        currentest = currentest*In/(In + incentive+1)
    else:
        currentest = (currentest*In + incentive+1)/(In+incentive+1)
    return currentest
#### simulate the mdp 
def simulatemdp(x_0 = np.array([M-1]),P = P,N = N,policy = None):
    
    
    n_mc = 100
    costs = np.zeros(n_mc)
    for mc in range(n_mc):
        x = x_0.copy()[0]
        cost = 0
        In = 0
        current_estimate = 0.5
        for i in range(N):
            u = int(policy(x))
            incentive = u%I
    
            x = np.random.choice(X,1,p = P[u,x].flatten())

            cost += return_cost(x,u,I,current_estimate)
            current_estimate = update_estimate(current_estimate,u,In)
            In += incentive+1
        cost += return_terminalcost(x)
        costs[mc] = cost
    return np.mean(costs) 



def sigmoid_policy(x,thresholds,tau = 1):
    m = x%M
    g = x//M
    action = 0
  
    for u in range(U):
        action += 1/(1+ np.exp(-(m-thresholds[g,u])/tau))
    action = min(max(action,0),U-1)
    return action
def clip_parameters(parameters):
    return np.clip(parameters,0,M-1)
n_iter = 40
parameters = np.ones((n_iter,G,U))*10
costs = np.zeros(n_iter)
delta = 1
step_size = 0.001
tau = 0.5
import tqdm
RUN_EXP = 1
if RUN_EXP:
    for i in tqdm.tqdm(range(n_iter-1)):
        parameters_to_optimize = np.random.binomial(1,0.5,size = (G,U))
        parameters_plus = parameters[i] + delta*parameters_to_optimize 
        parameters_minus = parameters[i] - delta*parameters_to_optimize
        cost_plus = simulatemdp(policy = lambda x: sigmoid_policy(x,parameters_plus,tau))
        cost_minus = simulatemdp(policy = lambda x: sigmoid_policy(x,parameters_minus,tau))
        gradient = (cost_plus - cost_minus)/(2*delta)
        print('gradient: ',gradient)
        parameters[i+1] = parameters[i] - step_size*gradient*parameters_to_optimize
        costs[i] = simulatemdp(policy= lambda x: sigmoid_policy(x,parameters[i],tau))
        tau = tau*0.9999
        parameters = clip_parameters(parameters)
        print('parameters: ',parameters[i])

    np.save('./mdp_parameters.npy',parameters)
    np.save('./costs.npy',costs)
parameters = np.load('./mdp_parameters.npy')
costs = np.load('./costs.npy')
import matplotlib.pyplot as plt
plt.plot(costs)
plt.savefig('plots/costs.png')
plt.close()
plt.plot(parameters[:,0,0])
plt.plot(parameters[:,-1,0])
print(parameters[-1])
plt.savefig('plots/parameters.png')
### Stochastic approximation for finding the optimal policy with threshold structure


