
import numpy as np
### Rosenbrock function
def rosenbrock(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

### gradient of the Rosenbrock function
def rosenbrock_grad(x):
    grad = np.zeros_like(x)
    grad[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    grad[-1] = 200*(x[-1]-x[-2]**2)
    grad[1:-1] = -400*x[1:-1]*(x[2:]-x[1:-1]**2) + 200*(x[1:-1]-x[:-2]**2) - 2*(1-x[1:-1])
    return grad

def rastrigin(x):
    A = 10
    return A*x.size + sum(x**2 - A*np.cos(2*np.pi*x))

def rastrigin_grad(x):
    A = 10
    grad = 2*x + 2*np.pi*A*np.sin(2*np.pi*x)
    return grad

### noisy gradient of the Rosenbrock function
def noisy_grad(x,grad_fn, noise_level=1e-2):
    noise = noise_level * np.random.normal(size=x.size)
    return grad_fn(x) + noise



### sgd with constant step size for minimizing the Rosenbrock function
def sgd(x0,grad_fn,fn, n_iter=100000, step_size=1e-3, noise_level=5e-2):
    gradient_norms = np.zeros(n_iter)
    x = x0.copy()
    for i in range(n_iter):
        grad = noisy_grad(x,grad_fn, noise_level=noise_level)
        x -= step_size * grad
        x = np.clip(x,-5,5)
        gradient_norms[i] = np.linalg.norm(grad)
        if i%1000==0:
            print(gradient_norms[i],fn(x),np.linalg.norm(x))
    return gradient_norms

def assign_gradient_state(gradient):
### 
    thresholds = [0,0.5,1]
    return np.digitize(gradient,thresholds)-1
def compute_transition_probability_matrix(gradient_states):
    n_states = np.max(gradient_states)+1
    P = np.zeros((n_states,n_states))
    for i in range(len(gradient_states)-1):
        P[gradient_states[i],gradient_states[i+1]] += 1
    return P
if __name__ == "__main__":
    x0 = np.ones(3)
    gradient_norms = sgd(x0,rastrigin_grad,fn=rastrigin)
    gradient_states = assign_gradient_state(gradient_norms)
    print(gradient_states)
    P = compute_transition_probability_matrix(gradient_states)
    P = P/P.sum(axis=1,keepdims = True)
    print(P)
