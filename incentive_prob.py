import numpy as np
import tqdm
n_clients = 30
I = 3
oracle_states = [0,15,25]
O = len(oracle_states)
def compute_oracle_state(n_clients_participating,oracle_states):
    return np.digitize(n_clients_participating,oracle_states)-1

def round_succ(n_participating_clients,thresholds,client_dataset_size,i):
    data_available_for_sampling = np.exp(-1/(i+1))*client_dataset_size
    total_data = np.random.uniform(0,data_available_for_sampling,n_participating_clients).sum()
    return total_data > thresholds, total_data

client_dataset_size = 400

N_MC = int(3e4)
participation_client = np.random.choice([0,1],n_clients, p=[0.3,0.7])
participation_prob = [[0.8,0.2],[0.1,0.9]]
state_count = np.zeros((I,O))
succ_count = np.zeros((I,O))
oracle_states_store = np.zeros(N_MC)
client_dataset_contribution = np.zeros((N_MC,I,O))
for mc in tqdm.tqdm(range(N_MC)):
    n_clients_participating = sum(participation_client)

    oracle_state = compute_oracle_state(n_clients_participating,oracle_states)
    oracle_states_store[mc] = n_clients_participating
    for i in range(I):
        state_count[i,oracle_state] += 1
        succ = round_succ(n_clients_participating,2000,client_dataset_size,i)
        succ_count[i,oracle_state] += succ[0]
        client_dataset_contribution[mc,i,oracle_state] = succ[1]
    for client in range(n_clients):
        participation_client[client] = np.random.choice([0,1],p=participation_prob[participation_client[client]])


succ_prob = succ_count/state_count
print(succ_prob)
print(state_count)
print(client_dataset_contribution.std(0))

import matplotlib.pyplot as plt
plt.figure()
plt.hist(oracle_states_store)
plt.savefig('plots/oracle_states.png')

plt.figure()
plt.hist(client_dataset_contribution[:,0,0])
plt.savefig('plots/client_dataset_contribution_0_0.png')

plt.figure()
plt.hist(client_dataset_contribution[:,0,1])
plt.savefig('plots/client_dataset_contribution_0_1.png')

plt.figure()
plt.hist(client_dataset_contribution[:,0,2])
plt.savefig('plots/client_dataset_contribution_0_2.png')

