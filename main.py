
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

### noisy gradient of the Rosenbrock function
def noisy_rosenbrock_grad(x, noise_level=1e-2):
    noise = noise_level * np.random.normal(size=x.size)
    return rosenbrock_grad(x) + noise

### sgd with constant step size for minimizing the Rosenbrock function
def sgd_rosenbrock(x0, n_iter=100000, step_size=1e-3, noise_level=1e-2):
    x = x0.copy()
    for i in range(n_iter):
        grad = noisy_rosenbrock_grad(x, noise_level=noise_level)
        x -= step_size * grad
    return x

### 
if __name__ == "__main__":
    x0 = np.zeros(100)
    x = sgd_rosenbrock(x0)
    print("x: ", x)
    print("f(x): ", rosenbrock(x))

