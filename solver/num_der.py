# -*- coding: utf-8 -*-

def derivative_numerical(f, x0, i, delta = 1e-8):
    xi_plus = x0.copy()
    xi_plus[i] += delta

    xi_minus = x0.copy()
    xi_minus[i] -= delta
    return (f(xi_plus) - f(xi_minus)) / (2 * delta)



def gradient_numerical(f, x0, delta = 1e-8):
    """
    function calculates the numerical gradient for function f in 
    the point x0
    """
    N = len(x0)
    grad_num = np.zeros([N, 1])
    for i in range(N):
        grad_num[i] = derivative_numerical(f, x0, i, delta)
    return grad_num


def check_grad(f, gradf, x0, delta = 1e-8, verbose = True):
    grad = np.array(gradf(x0))
    grad_num = gradient_numerical(f, x0, delta)
    if (verbose):
        print('check_grad: precise gradient = ', grad)
        print('check_grad: approximate gradient = ', grad_num)
        print('check_grad: gradient error = ', grad - grad_num)        
        
    return np.sqrt(np.sum((grad - grad_num) ** 2))

def second_derivative_numerical(f, x0, i, k, delta = 1e-5):
    """
	function calculates second derivative
    returns d^2f/(dx_k dx_i)
    """
    xk_plus = x0.copy()
    xk_plus[k] += delta

    xk_minus = x0.copy()
    xk_minus[k] -= delta
    
    dfi_plus = derivative_numerical(f, xk_plus, i, delta)
    dfi_minus = derivative_numerical(f, xk_minus, i, delta)
    
    return (dfi_plus - dfi_minus) / (2 * delta)

def hessian_numerical(f, x0, delta = 1e-5):
    """
	#  function calculates the hessian matrix
    """
    assert x.shape[1] == 1, 'hessian_numerical: input array should have shape [N, 1]'
        
    N = len(x)
    hessian = np.zeros([N, N], dtype = np.float64)
    for i in range(N):
        for k in range(i, N):
            hessian[i, k] = second_derivative_numerical(f, x0, i, k, delta)
            if i != k:
                hessian[k, i] = hessian[i, k]
    return hessian

def check_hessian(f, hess_analytical, x0, delta = 1e-5, verbose = True):
    """
	function checks he hessian matrix
    """
    hessian_analytical = np.array(hess_analytical)
    hessian_num = hessian_numerical(f, x0, delta)
    if verbose:
        print('check_hessian: hessian_analytical = ', hessian_analytical)
        print('check_hessian: hessian_num = ', hessian_num)
        print('check_hessian: hessian difference = ', 
              hessian_analytical - hessian_num)
        
    return np.sqrt(np.sum((hessian_analytical - hessian_num) ** 2))
