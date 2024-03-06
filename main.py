
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

def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    return -a * np.exp(-b * np.sqrt(np.sum(x**2) / x.shape[-1])) - np.exp(np.sum(np.cos(c * x)) /x.shape[-1] ) + a + np.exp(1)

def ackley_grad(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    grad = np.zeros_like(x)
    d = np.sqrt(np.sum(x**2) / x.shape[-1])
    grad = -a * (x / (x.shape[-1] * d)) * np.exp(-b * d) - c*(np.sin(c * x) / x.shape[-1]) * np.exp(np.sum(np.cos(c * x)) / x.shape[-1])
    return grad

### noisy gradient of the Rosenbrock function
def noisy_grad(x,grad_fn, noise_level=1e-2):
    noise = noise_level * np.random.normal(size=x.size)
    return grad_fn(x) + noise



### sgd with constant step size for minimizing the Rosenbrock function
def sgd(x0,grad_fn,fn, n_iter=1000, step_size=1e-1, noise_level=1):
    x = x0.copy()
    argmin_x = x0.copy()
    for i in range(n_iter):
        grad = noisy_grad(x,grad_fn, noise_level=noise_level)
        x -= step_size * grad
        x = np.clip(x,-32,32)
        if fn(x) < fn(argmin_x):
            argmin_x = x.copy()
        if i%100==0:
            print(fn(argmin_x),argmin_x,x,grad)
    


if __name__ == "__main__":
    x0 = np.ones((1,10))*0.8
    sgd(x0,ackley_grad,fn=ackley)
    print(ackley(x0+10))
    
