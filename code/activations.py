import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp


def sigmoid(z):
    return 1.0/(1.0 + jnp.exp(-z))


def sigmoid_der(z):
    s = sigmoid(z)
    return s*(1.0 - s)

def ReLU(z):
    return jnp.where(z > 0, z, 0.0)

def ReLU_der(z):
    return jnp.where(z > 0, 1.0, 0.0)

def leaky_ReLU(z, alpha = 0.01):
    return jnp.where(z>0 , z, alpha*z)

def leaky_ReLU_der(z, alpha = 0.01):
    return jnp.where(z > 0, 1.0, alpha)

def softmax(z):                                   # rows are samples, Eq. (5.30)
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))
    return e_z/jnp.sum(e_z, axis=1, keepdims=True)

def linear(z):
    return z

def linear_der(z):
    return jnp.ones_like(z)

