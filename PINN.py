import autograd.numpy as np  # Autograd-aware NumPy
from autograd import grad
import numpy as onp          # Regular NumPy for non-differentiable ops
import matplotlib.pyplot as plt

# Import Adam from autograd
from autograd.misc.optimizers import adam

###############################################################################
# 1) Exact Analytical Solution
###############################################################################
def exact_solution(d, w0, t):
    """
    Analytical solution to the under-damped harmonic oscillator:
        d2u/dt2 + 2d du/dt + (w0^2) u = 0,
    under the condition d < w0.
    """
    assert d < w0, "Require underdamped condition d < w0."
    w = np.sqrt(w0**2 - d**2)  # damped frequency
    phi = np.arctan(-d / w)
    A = 1.0 / (2.0 * np.cos(phi))
    return np.exp(-d * t) * 2.0 * A * np.cos(phi + w * t)

###############################################################################
# 2) Small Fully-Connected Network (autograd.numpy)
###############################################################################
def init_network(n_input, n_output, n_hidden, n_layers, seed=123):
    """
    Initialize a multi-layer perceptron (MLP) with the given dimensions.
    We store parameters as a list of (W, b) pairs for each layer.
    """
    onp.random.seed(seed)  # seed the RNG for reproducibility
    params = []
    
    # Input -> first hidden
    W = onp.random.randn(n_input, n_hidden) * np.sqrt(2.0/(n_input+n_hidden))
    b = onp.zeros(n_hidden)
    params.append((np.array(W), np.array(b)))

    # Hidden -> hidden
    for _ in range(n_layers - 1):
        W = onp.random.randn(n_hidden, n_hidden) * np.sqrt(2.0/(n_hidden+n_hidden))
        b = onp.zeros(n_hidden)
        params.append((np.array(W), np.array(b)))
        
    # Last hidden -> output
    W = onp.random.randn(n_hidden, n_output) * np.sqrt(2.0/(n_hidden+n_output))
    b = onp.zeros(n_output)
    params.append((np.array(W), np.array(b)))

    return params


def neural_net(t, params):
    """
    Forward pass of the MLP. 
    t can be shape (n_samples,) or (n_samples, 1). Returns shape (n_samples, 1).
    """
    if t.ndim == 1:
        t = t[:, None]  # ensure shape (N,1)
    
    x = t
    # Hidden layers (tanh activation)
    for (W, b) in params[:-1]:
        x = np.dot(x, W) + b
        x = np.tanh(x)
    # Final layer (linear)
    W, b = params[-1]
    x = np.dot(x, W) + b
    return x

###############################################################################
# 3) Helper: scalar version + derivatives
###############################################################################
def neural_net_scalar(t, params):
    """
    A wrapper that treats 't' as a single scalar for autograd.grad.
    We call the full network on a batch of size 1, then return the scalar.
    """
    return neural_net(np.array([t]), params)[0, 0]

# 1st derivative wrt scalar t
dudt_nn_scalar = grad(neural_net_scalar, 0)
# 2nd derivative wrt scalar t
d2udt2_nn_scalar = grad(dudt_nn_scalar, 0)

def dudt_nn_array(t_array, params):
    """
    For multiple points, compute du/dt at each t by calling dudt_nn_scalar.
    Returns shape (N,1).
    """
    vals = [dudt_nn_scalar(ti, params) for ti in t_array]
    return np.array(vals)[:, None]

def d2udt2_nn_array(t_array, params):
    """
    For multiple points, compute d2u/dt2 at each t by calling d2udt2_nn_scalar.
    Returns shape (N,1).
    """
    vals = [d2udt2_nn_scalar(ti, params) for ti in t_array]
    return np.array(vals)[:, None]

###############################################################################
# 4) Define the Physics-Informed Loss
###############################################################################
def loss_fn(params, iteration):
    """
    total_loss = boundary_loss + lambda2 * physics_loss
    boundary_loss = (u(0)-1)^2 + lambda1*(du/dt(0)-0)^2
    physics_loss = MSE of (d2u/dt2 + mu*(du/dt) + k*u)
    """
    # Weight factors
    lambda1 = 1e-1
    # Increase PDE weight from 1e-4 to 1e-3 for stronger PDE enforcement:
    lambda2 = 1e-4

    # A) Boundary loss (enforce u(0)=1, du/dt(0)=0)
    u0 = neural_net_scalar(0.0, params)       
    dudt0 = dudt_nn_scalar(0.0, params)      

    loss_b1 = (u0 - 1.0)**2
    loss_b2 = (dudt0 - 0.0)**2
    boundary_loss = loss_b1 + lambda1 * loss_b2

    # B) Physics loss: d2u/dt2 + mu*dudt + k*u = 0
    u_vals = neural_net(t_physics, params)            
    dudt_vals = dudt_nn_array(t_physics, params)      
    d2udt2_vals = d2udt2_nn_array(t_physics, params)  

    residual = d2udt2_vals + mu*dudt_vals + k*u_vals
    physics_loss = np.mean(residual**2)

    return boundary_loss + lambda2*physics_loss

###############################################################################
# 5) Setup: PDE parameters, data, etc.
###############################################################################
d = 2.0
w0 = 20.0
mu = 2*d
k = w0**2

# Increase physics points from 30 to 50 for better coverage
N_physics = 50
t_physics = np.linspace(0, 1, N_physics)

# For evaluation
t_test = np.linspace(0, 1, 300)
u_exact = exact_solution(d, w0, t_test)

# Initialize network
n_input = 1
n_output = 1
n_hidden = 32
n_layers = 3
params = init_network(n_input, n_output, n_hidden, n_layers, seed=123)

###############################################################################
# 6) Adam Optimizer Setup
###############################################################################
def callback(p, i, g):
    """
    Callback function called by Adam after each iteration.
    We can use it to monitor loss, print or plot occasionally, etc.
    """
    
    # print metrics
    current_loss = loss_fn(p, i)
    print(f"Iteration {i}, Loss={current_loss:.4e}")

    # For efficiency, only evaluate and plot every 2000 steps
    if i % 1000 == 0:
        
        # Evaluate current solution for plotting
        u_pred = neural_net(t_test, p)[:, 0]
        
        # Plot the result
        plt.figure(figsize=(5,2.5))
        # Physics points (green)
        plt.scatter(t_physics, np.zeros_like(t_physics), s=20, lw=0, 
                    color="tab:green", alpha=0.6, label="Physics pts")
        # Boundary (t=0)
        plt.scatter([0.0], [0.0], s=20, lw=0, color="tab:red", alpha=0.6, label="Boundary pt")
        
        # Exact
        plt.plot(t_test, u_exact, label="Exact", color="tab:gray", alpha=0.8)
        # PINN
        plt.plot(t_test, u_pred, label="PINN", color="tab:green")
        
        plt.title(f"Iteration {i}, Loss={current_loss:.4e}")
        plt.legend()
        plt.show()

# We define the gradient function for Adam
grad_func = lambda p, i: grad(loss_fn)(p, i)

# Increase the number of iterations to 30k for a difficult, highly oscillatory problem
num_iters = 30000
step_size = 1e-3  # typical Adam step size

###############################################################################
# 7) Train with Adam
###############################################################################
from autograd.misc.optimizers import adam

best_params = adam(grad_func, 
                   params,
                   step_size=step_size, 
                   num_iters=num_iters, 
                   callback=callback)

###############################################################################
# 8) Final Comparison
###############################################################################
u_pred = neural_net(t_test, best_params)[:, 0]
final_loss = loss_fn(best_params, 0)
print(f"Training complete. Final loss: {final_loss:.4e}")

plt.figure(figsize=(6,3))
plt.plot(t_test, u_exact, label="Exact", color="tab:blue", alpha=0.8, linewidth=2)
plt.plot(t_test, u_pred, label="PINN", color="tab:orange", linestyle="--", linewidth=2)
plt.title("Final PINN vs. Exact (Damped Harmonic Oscillator)")
plt.xlabel("t")
plt.ylabel("u(t)")
plt.legend()
plt.show()
