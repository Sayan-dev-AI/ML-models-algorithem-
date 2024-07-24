import numpy as np

# Adam optimizer class
class AdamOptimizer:
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        
        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        params -= self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params

# Example usage
def example_function(x):
    return x**2

def example_gradient(x):
    return 2*x

# Initialize parameters
params = np.array([5.0])  # Example parameter
optimizer = AdamOptimizer()

# Training loop
for i in range(1100):
    grads = example_gradient(params)  # Compute gradient
    params = optimizer.update(params, grads)  # Update parameters
    print(f"Iteration {i}:  params = {params}   loss = {example_function(params)}")

print(f"Final parameters: {params}")
