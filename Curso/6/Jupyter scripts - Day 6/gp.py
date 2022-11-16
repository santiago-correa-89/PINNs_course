from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def kernel(a, b):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * sqdist)

n = 50 # number of test points.

Xtest = np.linspace(-5, 5, n).reshape(-1,1) # Test points.

K_ = kernel(Xtest, Xtest) # Kernel at test points.
# draw samples from the prior at our test points.

L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))

f_prior = np.dot(L, np.random.normal(size=(n,10)))

plt.plot(Xtest, f_prior)
plt.show()