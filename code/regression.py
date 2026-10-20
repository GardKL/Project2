"""Regression tools for Project 1 -- Runge's function.

FYS-STK3155/4155, autumn 2026.

"""

import numpy as np

# Paul Tol's colourblind-safe palette, as used in the weekly slides
BLUE, RED, YELLOW, GREY, GREEN = "#004488", "#BB5566", "#DDAA33", "#777777", "#228833"


# ---------------------------------------------------------------------------
# Step 1: the data
# ---------------------------------------------------------------------------

def runge(x):
    """Runge's function f(x) = 1 / (1 + 25 x^2)."""
    return 1 / (1 + 25 * x**2)


def make_data(n=100, sigma=0.1, seed=2026, uniform=True):
   
    rng= np.random.default_rng(seed)

    if uniform:
        x=rng.uniform(-1, 1, n)
    else:
        x=np.linspace(-1, 1, n)

    y= runge(x) + sigma * rng.standard_normal(n)
    return x, y
    


# ---------------------------------------------------------------------------
# Step 2: the design matrix
# ---------------------------------------------------------------------------

def polynomial_features(x, degree, intercept=False):
    
    X= np.vander(x, degree + 1, increasing=True)
    if not intercept:
        X= X[:, 1:]
    return X
   

# ---------------------------------------------------------------------------
# Step 3: the metrics
# ---------------------------------------------------------------------------

def MSE(y_data, y_model):
    """Mean squared error."""
    
    return np.mean((y_data - y_model)**2 )


def R2(y_data, y_model):
    """The R^2 score."""
    numerator = np.sum (( y_data - y_model)**2)
    denominator= np.sum((y_data - np.mean(y_data))**2)

    return 1- numerator/denominator 


# ---------------------------------------------------------------------------
# Step 4: ordinary least squares
# ---------------------------------------------------------------------------

def ols(X, y):
    return np.linalg.pinv(X) @ y


# Step 5: Ridge
def ridge(X, y, lam):
    n = X.shape[0]
    p = X.shape[1]

    I = np.eye(p)

    theta = np.linalg.solve(
        X.T @ X + n * lam * I,
        X.T @ y
    )

    return theta


# Step 6: build features, scale, fit, predict
def fit_predict(x_train, x_test, y_train, degree, lam=0.0):

    X_train = polynomial_features(x_train, degree)
    X_test = polynomial_features(x_test, degree)

    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)

    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std

    y_mean = np.mean(y_train)
    y_train_centered = y_train - y_mean

    if lam == 0.0:
        theta = ols(X_train_scaled, y_train_centered)
    else:
        theta = ridge(
            X_train_scaled,
            y_train_centered,
            lam
        )

    y_train_pred = X_train_scaled @ theta + y_mean
    y_test_pred = X_test_scaled @ theta + y_mean

    return y_train_pred, y_test_pred, theta



#LLM-assisted (Claude): short snippets and guidance on structure; tested by the authors.




  