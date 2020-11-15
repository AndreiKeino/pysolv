# -*- coding: utf-8 -*-

import autograd.numpy as np
from autograd import grad, elementwise_grad as egrad, jacobian

def newton_opt(func, x0 = None, max_iters = 100):
    
    # the Newton's method implementation
    while True:
        
        print('iteration number ', iter_num)
        grad = h.gradf(x, a) #  gradient of f
        hess = h.hessf(x, a) #  hessian of f
        ihess = np.linalg. inv(hess) #  inverse hessian of f
        dx = - ihess @ grad #  Newton step
        lam_sq = grad.T @ (ihess @ grad) # Newton decrement
    
        print('lam_sq = %e' % lam_sq)
        if np.sqrt(lam_sq / 2) <= nu_min:
            print("Newton's method: tolerance achieved, exiting...")
            print('iteration number ', iter_num)
            #  print('a = ', a)
            print('optimal value = %e' % f(x, a))
            print('optimal x = ', x)
            break
        #  Backtracking line search
    
        t = h.backtrack_2(x, a, grad, ihess, alpha, beta)
        step = t
        # step = 1
    
        print('step =', step)
        
        x = x + step * dx
        
        print('new x = ', x)    
        iter_num += 1
        if iter_num >= max_iters:
            print("Newton's method: max_iters number exceeded")
            break


if __name__ == "__main__":
    
    A = np.array([[1, 2], [0, 1]], dtype=np.float64)
        
    x0 = np.array([1, 2], dtype=np.float64).reshape(- 1, 1)
    #  x0 = np.array([1, -2], dtype=np.float64).reshape(- 1, 1)
    
    def f(x):
        
        return 1.0 * np.squeeze(x.T @ A @ x)
    
    
    gradf = grad(f)

    #  https://math.stackexchange.com/questions/3254520/computing-hessian-in-python-using-finite-differences    
    #  hesian 
    hessianf = jacobian(egrad(f))
    
    print('f(x0) = ', f(x0)) # 9 # 1
    print('gradf(x0) = ', gradf(x0)) #  [6, 6] # [-2, -2]
    
    print('hessianf(x0) = ', np.squeeze(hessianf(x0)))

    x0 = np.array([1, 2], dtype = np.float64).reshape(- 1, 1)
    
    assert f(x0) == 9, 'function calculation is incorrect'
    assert np.allclose(gradf(x0), np.array([[6], [6]], dtype=np.float64)), \
        'gradient calculation is incorrect'
    assert np.allclose(np.squeeze(hessianf(x0)), np.array([[2, 2], [2, 2]], 
                                                         dtype=np.float64)), \
        'hessian calculation is incorrect'

                

    x0 = np.array([1, -2], dtype = np.float64).reshape(- 1, 1)
    assert f(x0) == 1, 'function calculation is incorrect'
    assert np.allclose(gradf(x0), np.array([[-2], [-2]], dtype=np.float64)), \
        'gradient calculation is incorrect'
    assert np.allclose(np.squeeze(hessianf(x0)), np.array([[2, 2], [2, 2]], 
                                                         dtype=np.float64)), \
        'hessian calculation is incorrect'
        
    
    