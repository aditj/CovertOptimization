''' 
This file contains the implementation of the SPSA algorithm for the main experiment.
'''

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from utils.funcs import sigmoid_policy,update_estimate,seed_everything
from oracle import Oracle
from nn import BERTClass
def sigmoid_policy_from_constant(state,parameters,tau,U = 6):
    '''
    Function to compute the sigmoid policy from constant incentivization
    state: State
    parameters: Parameters
    tau: Parameter for sigmoid policy
    '''
    ## copy the parameters to all the actions

    parameters_augmented = np.repeat(parameters,U,axis = 1)
    return sigmoid_policy(state,parameters_augmented,tau)


def terminal_cost(learner_state,eavesdropper_estimate):
    '''
    Function to compute the terminal cost
    learner_state: Learner state
    eavesdropper_estimate: Eavesdropper estimate
    '''

    return learner_state**2

def running_cost(action,oracle_state,learner_state,eavesdropper_estimate_prev,eavesdropper_estimate,I = 3):
    '''
    Function to compute the running cost
    action: Action taken
    oracle_state: Oracle state
    learner_state: Learner state
    eavesdropper_estimate_prev: Previous eavesdropper estimate
    eavesdropper_estimate: Current eavesdropper estimate
    I: Number of incentives
    '''

    learn_action = action//I ## Learn or Obfuscate
    incentive = action%I if learn_action == 1 else (I-1-action) ## Incentive 
    cost = 0
    if learn_action == 0:
        cost =  ((oracle_state+1)**2/((learner_state+1)**2)) * np.log(eavesdropper_estimate/eavesdropper_estimate_prev) ### Cost for obfuscation
    else:
        cost = ((learner_state+1)**2)/(oracle_state+1)**(2)* np.log(eavesdropper_estimate/eavesdropper_estimate_prev)/(1+np.exp((-eavesdropper_estimate+0.5)/0.01)) ### Cost for learning
    return cost

def approximate_finite_horizon_cost(N,N_MC,O,M,U,policy,update_estimate,get_cost,get_terminal_cost):
    '''
    Function to approximate the finite horizon cost
    N: Number of rounds
    N_MC: Number of Monte Carlo simulations
    O: Number of oracle states
    M: Number of learner states
    U: Number of actions
    policy: Policy function
    update_estimate: Function to update the eavesdropper estimate
    get_cost: Function to get the cost
    get_terminal_cost: Function to get the terminal cost
    '''

    costs_ = np.zeros(N_MC) ## Costs
    oracle = Oracle(dont_initialize_clients = True) ## Oracle
    for mc in range(N_MC):
        learner_state = M-1 ## Initial learner state
        eavesdropper_estimate = 0.5 ## Initial eavesdropper estimate
        oracle.reset_oracle([0.5,0.5]) ## Reset oracle
        oracle_state = oracle.oracle_state ## Oracle state
        for t in range(N):
            state = M*oracle_state + learner_state ## State
            action = policy(state)  ## Action
            learn_action = action//I ## Learn or Obfuscate
            incentive = action%I if learn_action == 1 else (I-1-action) ## Incentive
            if learner_state == 0:
                action = 0 ## If learner state is 0, then only obfuscate
                In = I - 1 ## Incentive
            eavesdropper_prev = eavesdropper_estimate ## Previous eavesdropper estimate
            eavesdropper_estimate = update_estimate(eavesdropper_estimate,action,t+1) ## Update eavesdropper estimate
            costs_[mc] += get_cost(action,oracle_state,learner_state,eavesdropper_prev,eavesdropper_estimate) ## Compute cost
            eavesdropper_estimate = min(max(eavesdropper_estimate,0.1),0.9) ## Clip eavesdropper estimate
            if action == 1:
                succ_round = oracle.train(incentive,parameters=None,return_only_succ = True) ## Train oracle
                if succ_round:
                    learner_state -= 1
            oracle.update_client_participation() ## Update client participation
            oracle.update_oracle_state() ## Update oracle state
            oracle_state = oracle.oracle_state ## Oracle state
        costs_[mc] += get_terminal_cost(learner_state,eavesdropper_estimate) ## Compute terminal cost
    return costs_.mean() ## Return mean cost

I = 3  ## Number of incentives
A = 2 ## Learn Or Obfuscate
U = I*A ## Total number of actions
M = 50 ## Number of learner states
N = 100 ## Number of rounds
O = 3 ## Number of oracle states

TRAIN_SPSA = 0
N_iter = 1000 ## Number of iterations for SPSA
delta = 1 ## Delta (perturbation) for SPSA
N_MC = 40 ## Number of Monte Carlo simulations
tau = 0.2 ## Parameter for sigmoid policy
parameters = np.ones((O,U))*25 ## Initial parameters
step_size = 0.1 ## Step size for SPSA
if TRAIN_SPSA:
    for iter_ in range(N_iter):
        parameters_change = np.random.choice([0,1],size = (O,U),p = [0.5,0.5]) ## Parameters to change
        parameters_plus = parameters + delta*parameters_change ## Parameters plus delta
        parameters_minus = parameters - delta*parameters_change ## Parameters minus delta
        costs_plus = approximate_finite_horizon_cost(N,N_MC,O,M,U,lambda x: sigmoid_policy(x,parameters_plus,tau),update_estimate,running_cost,terminal_cost) ## Cost for parameters plus delta
        costs_minus = approximate_finite_horizon_cost(N,N_MC,O,M,U,lambda x: sigmoid_policy(x,parameters_minus,tau),update_estimate,running_cost,terminal_cost) ## Cost for parameters minus delta
        gradient = np.zeros((O,U)) ## Gradient
        gradient[parameters_change == 1] = (costs_plus - costs_minus)/(2*delta) ## Compute gradient
        parameters = parameters - step_size*gradient ## Update parameters

        tau = tau*0.995 ## Update tau
        delta = delta*0.999 ## Update delta
        step_size = step_size*0.999 ## Update step size

        print(parameters,costs_plus)

    np.save("parameters/spsa_parameters")

### SPSA for constant incentivization ###

DO_SPSA_CONSTANT = 0
N_iter = 1000 ## Number of iterations for SPSA
delta = 1 ## Delta (perturbation) for SPSA
N_MC = 40 ## Number of Monte Carlo simulations
tau = 0.2 ## Parameter for sigmoid policy
parameters = np.ones((O,1))*25 ## Initial parameters
step_size = 0.1 ## Step size for SPSA
if DO_SPSA_CONSTANT:
    for iter_ in range(N_iter):
        parameters_change = np.random.choice([0,1],size = (O,1),p = [0.5,0.5]) ## Parameters to change
        parameters_plus = parameters + delta*parameters_change ## Parameters plus delta
        parameters_minus = parameters - delta*parameters_change ## Parameters minus delta
        costs_plus = approximate_finite_horizon_cost(N,N_MC,O,M,U,lambda x: sigmoid_policy_from_constant(x,parameters_plus,tau,U=6),update_estimate,running_cost,terminal_cost) ## Cost for parameters plus delta
        costs_minus = approximate_finite_horizon_cost(N,N_MC,O,M,U,lambda x: sigmoid_policy_from_constant(x,parameters_minus,tau,U=6),update_estimate,running_cost,terminal_cost) ## Cost for parameters minus delta
        gradient = np.zeros((O,1)) ## Gradient
        gradient[parameters_change == 1] = (costs_plus - costs_minus)/(2*delta) ## Compute gradient
        parameters = parameters - step_size*gradient ## Update parameters

        tau = tau*0.995 ## Update tau
        delta = delta*0.999 ## Update delta
        step_size = step_size*0.999 ## Update step size

        print(parameters,costs_plus)

    np.save("parameters/spsa_parameters_constant_incentivization")

