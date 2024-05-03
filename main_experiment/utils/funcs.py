import numpy as np
import torch
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def sigmoid_policy(x,thresholds,tau = 0.01,M=50,U=6):
    m = x%M
    o = x//M
    action = 0
  
    for u in range(U):
        action += 1/(1+ np.exp(-(m-thresholds[o,u])/tau))
    action = min(max(action,0),U-1)
    return action

def update_estimate(currentest,u,In,I=3):
    querytype = u//I
    incentive = u%I if querytype == 1 else (I-1-u)
    if querytype == 0:
        currentest_eav = currentest*In/(In + incentive+1)
    else:
        currentest_eav = (currentest*In + incentive+1)/(In+incentive+1)
    return currentest_eav