import numpy as np

from cost import mse, mse_der

def create_layers_batch(network_input_size, layer_output_sizes, rng):
    layers = []
    i_size = network_input_size

    for layer_output_size in layer_output_sizes:
        W = rng.normal(0,1/np.sqrt(i_size),(i_size, layer_output_size))
        b = np.zeros(layer_output_size) + 0.01
        layers.append((W,b))
        i_size = layer_output_size
    return layers

def feed_forward_batches(inputs,layers,activation_funcs):
    a = inputs

    for (W,b), activation_func in zip(layers,activation_funcs):
        a = activation_func(a@W +b)
    return a

def cost_batch(layers, inputs, activation_funcs, targets, cost=mse):
    return cost(feed_forward_batches(inputs, layers, activation_funcs), targets)

def feed_forward_saver_batches(inputs, layers, activation_funcs):
    layer_inputs = []
    zs = []
    a = inputs

    for (W,b), activation_func in zip(layers,activation_funcs):
        layer_inputs.append(a)
        z = a @W +b
        a = activation_func(z)
        zs.append(z)
    return layer_inputs, zs, a

def backprop_batch(inputs, layers, activation_funcs, targets, activation_ders, cost_der=mse_der):

    layer_inputs,zs, predict = feed_forward_saver_batches(inputs, layers, activation_funcs)
    layer_grads = [() for layer in layers]

    for i in reversed(range(len(layers))):
        layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]
        if i == len(layers) - 1:
            dC_da = cost_der(predict,targets)
        else:
            (W,b) = layers[i+1]
            dC_da = dC_dz @ W.T
        dC_dz = dC_da * activation_der(z)
        dC_dW = layer_input.T @ dC_dz
        dC_db = np.sum(dC_dz, axis=0)
        layer_grads[i] = (dC_dW,dC_db)
    return layer_grads