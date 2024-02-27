from learner import Learner
import numpy as np
N_MC = 100
N_rounds  = 100


gradient_norms = np.zeros((N_MC, N_rounds))
learner = Learner()

RUN_EXP = True
if RUN_EXP: 
    for mc in range(N_MC):
        for round_ in range(N_rounds):
            learner.train(round_)
            gradient_norms[mc, round_] = learner.gradient_norm
    np.save('gradient_norms.npy', gradient_norms)
    