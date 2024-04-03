
### Federated Learning Setup where the learner controls the learning based on different policies  

import numpy as np
import matplotlib.pyplot as plt
import torch
from learner import Learner

N_MC = 100
N_rounds  = 100 

I = 5
A = 2
U = I*A
M = 20
succ_prob = [0.2,0.4,0.6,0.8,1]

learner = Learner()
eavesdropper = Learner()
weights = learner.get_weights()

def update_estimate(currentest,u,In):
    querytype = u//I
    incentive = u%I
    if querytype == 0:
        currentest = currentest*In/(In + incentive+1)
    else:
        currentest = (currentest*In + incentive+1)/(In+incentive+1)
    return currentest

policies = []

learner_evaluations = np.zeros((N_MC, len(policies), N_rounds))
eavesdropper_evaluations = np.zeros((N_MC, len(policies), N_rounds))

RUN_EXP = True
if RUN_EXP: 
    for mc in range(N_MC):
        for policy_idx,policy in enumerate(policies):
            learner_state = 40
            gradient_state = learner.get_gradient_state()
            eavesdropper_estimate = 0.5
            In = 0
            for round_ in range(N_rounds):
                state = M*gradient_state + learner_state 
                action = policy(state)
                incentive = action%I
                type_query = action//I
                eavesdropper_estimate = update_estimate(eavesdropper_estimate,action,In)
                In = In + incentive + 1
                if type_query == 0:
                    continue
                else:
                    if np.random.rand() < succ_prob[incentive]:
                        learner_state = learner_state - 1
                        learner.train()
                        learner_evaluations[mc,policy_idx, round_] = learner.evaluate()
                        eavesdropper.set_weights(learner.get_weights())
                        eavesdropper_evaluations[mc,policy_idx, round_] = eavesdropper.evaluate()
                    
        np.save('learner_evaluations.npy', learner_evaluations)
        np.save('eavesdropper_evaluations.npy', eavesdropper_evaluations)