# -*- coding: utf-8 -*-
import autograd.numpy as np
from autograd import grad

A = np.array([[1, 2], [2, 1]], dtype = np.float64)

x0 = np.array([-0.75, 0.5], dtype = np.float64).reshape(- 1, 1)

def f(x):
    
    return 1.0 * np.squeeze(x.T @ A @ x)


gradf = grad(f)
