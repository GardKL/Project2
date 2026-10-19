"""Unit tests for neural_network.py, cost.py, activations.py

Run with `pytest test_backprop.py`, or with `python test_backprop.py`
if pytest is not installed.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that results agree with numpy to ~1e-15

from neural_network import create_layers_batch, backprop_batch, cost_batch
from activations import sigmoid, sigmoid_der, linear, linear_der, ReLU, ReLU_der, leaky_ReLU, leaky_ReLU_der, softmax
from cost import cross_entropy, cross_entropy_softmax_der

def test_activation_values():
    assert np.isclose(sigmoid(0.0),0.5)
    assert np.isclose(ReLU(-2.0),0.0)
    assert np.isclose(ReLU(3.0),3.0)
    assert np.isclose(leaky_ReLU(-2.0), -0.02)    # alpha = 0.01
    assert np.isclose(leaky_ReLU(3.0), 3.0)
    assert np.isclose(linear(5.0), 5.0)

def test_activation_der_jax():
    z = np.linspace(-3,3,10) #10 points, without having z = 0
    mine = sigmoid_der(z)
    check = jax.vmap(jax.grad(sigmoid))(z)
    assert np.allclose(mine,check)
    mine = ReLU_der(z)
    check = jax.vmap(jax.grad(ReLU))(z)
    assert np.allclose(mine,check)
    mine = leaky_ReLU_der(z)
    check = jax.vmap(jax.grad(leaky_ReLU))(z)
    assert np.allclose(mine,check)
    mine = linear_der(z)
    check = jax.vmap(jax.grad(linear))(z)
    assert np.allclose(mine,check)

def test_softmax():
    rng = np.random.default_rng(2026)
    z = rng.normal(size=(4,10))
    z[0] = 1000.0 + z[0]

    a = softmax(z)

    assert np.allclose(np.sum(a,axis=1),1)
    assert np.all(a>= 0) and np.all(a<= 1)
    assert not np.any(np.isnan(a))

def test_cross_entropy():
    y = np.zeros((4,10))
    y[np.arange(4),[3,0,7,1]] = 1

    a = y
    assert np.isclose(cross_entropy(a,y),0.0)

    a = np.full((4,10),0.1)
    assert np.isclose(cross_entropy(a,y), -np.log(0.1))

def test_layer_shapes():
    rng = np.random.default_rng(2026)
    sizes = [784,100,50,10]

    layers = create_layers_batch(sizes[0],sizes[1:],rng)

    assert len(layers) == len(sizes) - 1

    for (W,b), n_in, n_out in zip(layers, sizes[:-1], sizes[1:]):
        assert W.shape == (n_in, n_out)
        assert b.shape == (n_out,)

def test_layers_reproduce():
    sizes = [1,50,1]
    layers1 = create_layers_batch(sizes[0], sizes[1:], np.random.default_rng(2026))
    layers2 = create_layers_batch(sizes[0], sizes[1:], np.random.default_rng(2026))
    layers3 = create_layers_batch(sizes[0], sizes[1:], np.random.default_rng(42))

    for (W1,b1), (W2,b2) in zip(layers1,layers2):
        assert np.array_equal(W1,W2)
        assert np.array_equal(b1,b2)
    assert not np.array_equal(layers1[0][0], layers3[0][0])


def test_backprop_sigmoid_jax():
    rng = np.random.default_rng(2026)

    layers = create_layers_batch(1,[3,1], rng)
    activation_funcs = [sigmoid, linear]
    activation_ders = [sigmoid_der, linear_der]

    inputs = rng.normal(size=(5,1))
    targets = rng.normal(size=(5,1))

    mine = backprop_batch(inputs,layers,activation_funcs,targets,activation_ders)
    check = jax.grad(cost_batch)(layers,inputs,activation_funcs,targets)

    for (dW,db), (dW_jax, db_jax) in zip(mine,check):
        assert np.allclose(dW,dW_jax)
        assert np.allclose(db,db_jax)


def test_backprop_relu_jax():
    rng = np.random.default_rng(2026)
    layers = create_layers_batch(1,[4,3,1], rng)
    a_funcs = [ReLU, leaky_ReLU, linear]
    d_funcs = [ReLU_der, leaky_ReLU_der, linear_der]

    inputs = rng.normal(size=(5,1))
    targets = rng.normal(size=(5,1))
    
    mine = backprop_batch(inputs,layers,a_funcs,targets,d_funcs)
    check = jax.grad(cost_batch)(layers,inputs,a_funcs,targets)
    
    for (dW,db), (dW_jax, db_jax) in zip(mine,check):
        assert np.allclose(dW,dW_jax)
        assert np.allclose(db,db_jax)

def test_backprop_soft_cross_jax():
    rng = np.random.default_rng(2026)
    layers = create_layers_batch(4,[5,3],rng)
    a_funcs = [sigmoid, softmax]
    d_funcs = [sigmoid_der, linear_der] # softmax: cost_der already has full delta^L
    cost_der = cross_entropy_softmax_der
    inputs = rng.normal(size = (6,4))
    targets = np.zeros((6,3))
    targets[np.arange(6), rng.integers(0,3,size = 6)] = 1

    mine = backprop_batch(inputs,layers, a_funcs,targets,d_funcs,cost_der = cost_der)
    check = jax.grad(cost_batch)(layers,inputs,a_funcs,targets,cost=cross_entropy)


    for (dW,db), (dW_jax, db_jax) in zip(mine,check):
        assert np.allclose(dW,dW_jax)
        assert np.allclose(db,db_jax)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("passed:", t.__name__)