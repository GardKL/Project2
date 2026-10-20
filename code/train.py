import numpy as np

from neural_network import backprop_batch, cost_batch

def optimiser_step(method, theta, g, state, t, gamma, beta=0.9, rho=0.99,
                   beta1=0.9, beta2=0.999, eps=1e-8):
    """One update of theta from the gradient g at step t = 1, 2, ...; state carries the running quantities."""
    if method == "plain":                                    # Eq. (4.10)
        return theta - gamma * g, state
    if method == "momentum":
        v = beta * state.get("v", 0.0) + gamma * g                                 # Eq. (4.28)
        state["v"] = v
        return theta - v, state
    if method == "adagrad":    
        r = state.get("r", 0.0) + g * g                              # Eqs. (4.42) and (4.45)
        state["r"] = r 
        return theta - gamma * g /(np.sqrt(r) + eps ), state
    if method == "rmsprop":  
        r = rho * state.get("r", 0.0) + (1.0 - rho)*g*g                                # Eqs. (4.47) and (4.48)
        state["r"] = r
        return theta - gamma * g / (np.sqrt(r) + eps), state
    if method == "adam":   
        m = beta1 * state.get("m", 0.0) + (1.0-beta1)*g
        r = beta2*state.get("r", 0.0) + (1.0-beta2)*g*g                                  # Eqs. (4.51), (4.52), (4.54), (4.55)
        state["m"] = m
        state["r"] = r 
        m_hat, r_hat = m / (1.0 - beta1**t), r / (1.0 - beta2**t)
        return theta - gamma*m_hat / (np.sqrt(r_hat) + eps), state
    raise ValueError(f"unknown method {method}")

def make_batches(n, batch_size, rng):
    """Shuffle the indices and split them into minibatches."""
    idx = rng.permutation(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


def train_gd(inputs, targets, layers, activation_funcs, activation_ders, gamma=0.1, n_steps=10000):
    history = []
    for step in range(n_steps):
        grads = backprop_batch(inputs, layers, activation_funcs, targets, activation_ders)
        layers = [(W - gamma*dW, b - gamma*db) for (W, b), (dW, db) in zip(layers, grads)]
        if step % 100 == 0 or step == n_steps - 1:
            history.append(float(cost_batch(layers, inputs, activation_funcs, targets)))
    return layers, history


def train_sgd(inputs, targets, layers, activation_funcs, activation_ders,
              method="adam", gamma=0.01, n_epochs=1000, batch_size=16, seed=2026):
    """Minibatch SGD with the optimisers of project 1; returns the layers and the training MSE per epoch."""
    rng = np.random.default_rng(seed)
    states = [[{},{}] for _ in layers]   # én state for W og én for b, per lag
    t = 0
    history = []

    for epoch in range(n_epochs):
        for idx in make_batches(len(inputs), batch_size, rng):
            grads = backprop_batch(inputs[idx], layers, activation_funcs, targets[idx], activation_ders)
            t += 1
            new_layers = []
            for i, ((W, b), (dW, db)) in enumerate(zip(layers, grads)):
                W, states[i][0] = optimiser_step(method, W, dW, states[i][0], t, gamma)
                b, states[i][1] = optimiser_step(method, b, db, states[i][1],t, gamma)
                new_layers.append((W,b))
            layers = new_layers
        history.append(float(cost_batch(layers, inputs, activation_funcs, targets)))

    return layers, history