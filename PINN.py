import numpy as np
import matplotlib.pyplot as plt

########################################################################
# Node class
########################################################################
class Node:
    def __init__(self, value, children=(), op=''):
        self.value = float(value)
        self.grad = 0.0
        self.children = children
        self.op = op
        self._backward = lambda: None

    def __repr__(self):
        return f"Node(value={self.value}, grad={self.grad}, op='{self.op}')"

    # --------------------- Overloaded Operators --------------------- #
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value + other.value, (self, other), op='+')
        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value - other.value, (self, other), op='-')
        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * -1.0
        out._backward = _backward
        return out

    def __rsub__(self, other):
        return Node(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value * other.value, (self, other), op='*')
        def _backward():
            self.grad += out.grad * other.value
            other.grad += out.grad * self.value
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value / other.value, (self, other), op='/')
        def _backward():
            self.grad += out.grad * (1.0 / other.value)
            other.grad += out.grad * (-self.value / (other.value ** 2))
        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return Node(other) / self

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "Exponent must be float/int"
        out = Node(self.value ** exponent, (self,), op=f'**{exponent}')
        def _backward():
            self.grad += out.grad * (exponent * (self.value ** (exponent - 1)))
        out._backward = _backward
        return out
    
    # --------------------- Elementary Functions --------------------- #
    def exp(self):
        out = Node(np.exp(self.value), (self,), op='exp')
        def _backward():
            self.grad += out.grad * out.value
        out._backward = _backward
        return out

    def sin(self):
        out = Node(np.sin(self.value), (self,), op='sin')
        def _backward():
            self.grad += out.grad * np.cos(self.value)
        out._backward = _backward
        return out

    def cos(self):
        out = Node(np.cos(self.value), (self,), op='cos')
        def _backward():
            self.grad += out.grad * -np.sin(self.value)
        out._backward = _backward
        return out

    def tanh(self):
        """Hyperbolic tangent activation."""
        val = np.tanh(self.value)
        out = Node(val, (self,), op='tanh')
        def _backward():
            # derivative of tanh(x) = 1 - tanh^2(x)
            self.grad += out.grad * (1 - val*val)
        out._backward = _backward
        return out

    def zero_grad(self):
        self.grad = 0.0


########################################################################
# Backprop & Utility Functions
########################################################################
def backward(root: Node):
    topo_order = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v.children:
                build_topo(child)
            topo_order.append(v)

    build_topo(root)
    root.grad = 1.0

    for node in reversed(topo_order):
        node._backward()

def zero_grad_all(node_list):
    for n in node_list:
        n.grad = 0.0


########################################################################
# Tanh Activation (for hidden layers)
########################################################################
def tanh_activation(x: Node) -> Node:
    return x.tanh()


########################################################################
# NeuralNetwork class
########################################################################
class NeuralNetwork:
    def __init__(self, n_input, n_hidden, n_output, n_hidden_layers=2, weight_init_scale=0.1):
        """
        We'll build a network:
            Input -> [Linear -> Tanh] x (n_hidden_layers) -> [Linear -> Output]
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_hidden_layers = n_hidden_layers
        
        self.weights = []
        self.biases = []
        
        # 1) Input -> first hidden layer
        in_dim = n_input
        out_dim = n_hidden
        W, b = self._init_layer(in_dim, out_dim, weight_init_scale)
        self.weights.append(W)
        self.biases.append(b)
        
        # 2) Hidden -> Hidden
        for _ in range(n_hidden_layers - 1):
            W, b = self._init_layer(n_hidden, n_hidden, weight_init_scale)
            self.weights.append(W)
            self.biases.append(b)
        
        # 3) Last hidden -> Output
        W, b = self._init_layer(n_hidden, n_output, weight_init_scale)
        self.weights.append(W)
        self.biases.append(b)

    def _init_layer(self, in_dim, out_dim, scale):
        W = []
        b = []
        for _out in range(out_dim):
            b_node = Node(np.random.randn() * scale)
            b.append(b_node)
            W_row = []
            for _in in range(in_dim):
                w_node = Node(np.random.randn() * scale)
                W_row.append(w_node)
            W.append(W_row)
        return W, b

    def forward(self, x):
        """Forward pass for a single input x (float or Node)."""
        if not isinstance(x, list):
            x = [x] if isinstance(x, Node) else [Node(x)]
        
        out = x
        total_layers = len(self.weights)  # hidden layers + final output layer
        for layer_idx in range(total_layers):
            W = self.weights[layer_idx]
            b = self.biases[layer_idx]
            new_out = []
            for neuron_idx in range(len(W)):
                z = b[neuron_idx]
                for in_idx, in_val in enumerate(out):
                    z = z + W[neuron_idx][in_idx] * in_val
                # Use Tanh for hidden layers, no activation on final layer
                if layer_idx < total_layers - 1:
                    z = tanh_activation(z)
                new_out.append(z)
            out = new_out
        
        # if single output, return Node rather than list[Node]
        if self.n_output == 1:
            return out[0]
        return out

    def forward_batch(self, X):
        """Forward pass for an array/list of inputs."""
        outputs = []
        for x_val in X:
            outputs.append(self.forward(x_val))
        return outputs

    def parameters(self):
        params = []
        for W, b in zip(self.weights, self.biases):
            for row in W:
                for w in row:
                    params.append(w)
            for b_node in b:
                params.append(b_node)
        return params


########################################################################
# Adam optimizer
########################################################################
class AdamOptimizer:
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0
        for p in self.params:
            self.m[p] = 0.0
            self.v[p] = 0.0

    def step(self):
        self.t += 1
        for p in self.params:
            g = p.grad
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * g
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (g*g)
            m_hat = self.m[p] / (1 - self.beta1**self.t)
            v_hat = self.v[p] / (1 - self.beta2**self.t)
            p.value -= self.lr * m_hat / (v_hat**0.5 + self.eps)


########################################################################
# Finite-Difference Approx for Derivatives (Vectorized)
########################################################################
def finite_diff_derivs_batch(net, t_points, delta=1e-4):
    """
    Vectorized approach to get [u(t+delta), u(t-delta), u(t)] in 3 forward batches.
    Returns three lists of Node: (u_plus, u_minus, u_zero).
    """
    t_plus = t_points + delta
    t_minus = t_points - delta
    
    u_plus = net.forward_batch(t_plus)    # list of Node
    u_minus = net.forward_batch(t_minus)  # list of Node
    u_zero = net.forward_batch(t_points)  # list of Node
    
    return u_plus, u_minus, u_zero


########################################################################
# Vectorized PINN Loss
########################################################################
def pinn_loss_vectorized(net, t_collocation, mu, k, 
                         boundary_conditions=None,
                         boundary_weight=1.0, physics_weight=1.0,
                         delta=1e-4):
    """
    1) Boundary conditions loss
    2) PDE residual (using vectorized forward_batch for the collocation points)
    """
    # Boundary Loss
    bc_loss = Node(0.0)
    if boundary_conditions:
        for bc in boundary_conditions:
            t_bc_node = Node(bc["t"])
            u_bc = net.forward(t_bc_node)
            
            if "target_u" in bc:
                diff_u = u_bc - bc["target_u"]
                bc_loss = bc_loss + diff_u*diff_u
            
            if "target_du" in bc:
                # finite diff for derivative at boundary
                t_plus = t_bc_node + delta
                t_minus = t_bc_node - delta
                u_plus = net.forward(t_plus)
                u_minus = net.forward(t_minus)
                du_bc = (u_plus - u_minus) / (2*delta)
                diff_du = du_bc - bc["target_du"]
                bc_loss = bc_loss + diff_du*diff_du

    # PDE Residual
    # Instead of looping for each t, we do 3 batch passes
    u_plus, u_minus, u_zero = finite_diff_derivs_batch(net, t_collocation, delta=delta)
    
    residual_sum = Node(0.0)
    N = len(t_collocation)
    for i in range(N):
        du_i = (u_plus[i] - u_minus[i]) / (2*delta)
        d2u_i = (u_plus[i] - (u_zero[i]*2.0) + u_minus[i]) / (delta*delta)
        
        r_i = d2u_i + mu*du_i + k*u_zero[i]
        residual_sum = residual_sum + r_i*r_i

    physics_loss = residual_sum / N
    return bc_loss*boundary_weight + physics_loss*physics_weight


########################################################################
# Driver Code
########################################################################
if __name__ == "__main__":
    # PDE constants
    d = 2
    w0 = 20
    mu = 2*d
    k = w0**2  # 400

    # 1) Create network
    #    3 hidden layers, 32 neurons each, Tanh activation
    net = NeuralNetwork(n_input=1, n_hidden=16, n_output=1, n_hidden_layers=3)

    # 2) Boundary conditions
    #    u(0) = 1, u'(0) = 0
    bc = [
        {"t": 0.0, "target_u": 1.0},
        {"t": 0.0, "target_du": 0.0},
    ]

    # 3) Collocation points
    #    80 points for better coverage
    t_physics = np.linspace(0, 1, 80)

    # -- (Optional) For random collocation each epoch, comment above lines and do:
    # def sample_collocation_points(n=80):
    #     return np.random.rand(n)  # in [0,1]

    # 4) Create Adam optimizer (smaller lr)
    opt = AdamOptimizer(net.parameters(), lr=1e-4)

    # 5) Training
    n_epochs = 2500
    for epoch in range(n_epochs):
        zero_grad_all(net.parameters())
        
        # If you want random collocation each epoch, do:
        # t_physics = sample_collocation_points(80)
        
        loss_node = pinn_loss_vectorized(
            net,
            t_collocation=t_physics,
            mu=mu,
            k=k,
            boundary_conditions=bc,
            boundary_weight=1.0,
            physics_weight=1.0,
            delta=1e-4
        )
        
        backward(loss_node)
        opt.step()

        # Print every 500 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss = {loss_node.value:.6f}")

    # 6) Evaluate
    t_test = np.linspace(0, 1, 200)
    preds = []
    for t_ in t_test:
        out_node = net.forward(t_)
        preds.append(out_node.value)

    # 7) Exact solution
    def exact_solution(d, w0, t_array):
        w = np.sqrt(w0**2 - d**2)
        phi = np.arctan(-d/w)
        A = 1.0 / (2.0 * np.cos(phi))
        return np.exp(-d*t_array) * 2.0 * A * np.cos(phi + w*t_array)

    u_exact = exact_solution(d, w0, t_test)

    print("\nFinal Comparison:")
    idxs = [0, 50, 100, 150, 199]
    for i in idxs:
        print(f"  t={t_test[i]:.3f} | PINN={preds[i]:.5f} vs Exact={u_exact[i]:.5f}")

    plt.plot(t_test, preds, label="PINN")
    plt.plot(t_test, u_exact, label="Exact", alpha=0.7)
    plt.legend()
    plt.show()
