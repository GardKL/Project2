import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp

def mse(predict, target):
    return jnp.mean((predict - target)**2)


def mse_der(predict, target):
    return 2.0*(predict - target)/predict.size  

def cross_entropy(predict, target):
    return -jnp.mean(jnp.sum(target*jnp.log(predict + 1e-12), axis=1))

def cross_entropy_softmax_der(predict, target):
    return (predict - target)/predict.shape[0]