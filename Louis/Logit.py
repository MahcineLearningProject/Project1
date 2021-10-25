import numpy as np
from gradient_descent import line_search
from implementation import build_k_indices
import time
''' Logistic regression '''

def sigmoid(t):
    return 1/(1+np.exp(-t))
    
def compute_log_likeli(y, tx, w):
    '''  Negative log-likelihood  '''
    sum = 0
    for i in range(tx.shape[0]):
        z = tx[i,:] @ w
        if z < -500: # to tackle overflow problems
            sum = sum - z + np.log1p(np.exp(z))
        else:
            sum = sum + np.log1p(np.exp(-z))
    loss = sum - 1/2*y.T @ tx @ w + np.ones((1,len(y))) @ tx @ w/2
    return loss
    
def compute_grad_log_likeli(y, tx, w):
    return np.dot(tx.T,sigmoid(np.dot(tx,w)) - y/2 +1/2)

def compute_hess_log_likeli(y, tx, w):
    hess = 0
    for i in range(len(tx @ w)):
        hess = hess + tx.T[:,i].reshape(len(w),1) @ tx[i,:].reshape(1,len(w)) * sigmoid(tx[i,:] @ w) * (1 - sigmoid(tx[i,:] @ w))
    return hess

def log_reg(y, tx, w):
    loss, grad, hess = compute_log_likeli(y, tx, w), compute_grad_log_likeli(y, tx, w), compute_hess_log_likeli(y, tx, w)
    return loss, grad, hess

'''def line_search_log_reg(y, tx, w, gamma_ini = 1/4, rho = 0.5):
    gamma = gamma_ini
    g = compute_grad_log_likeli(y, tx, w)
    loss = compute_log_likeli(y, tx, w)
    f_suiv = compute_log_likeli(y, tx, w-gamma*g)
    k = 1
    while (f_suiv > loss - 1e-3 * gamma * g.T @ g and k<20): # restriction on the number of iterations to save time
        gamma = rho*gamma
        f_suiv = compute_log_likeli(y, tx, w-gamma*g)
        k = k+1
    return gamma''' # method takes too long
    
def logistic_regression_gd(y, tx, initial_w, max_iters = 100, initial_gamma = 1e-2):
    gamma = initial_gamma
    w = initial_w
    loss, grad, _ = log_reg(y, tx, w)
    k = 1
    while k <= max_iters:
        t_start = time.time()
        w = w - gamma*grad
        loss, grad, _ = log_reg(y, tx, w)
        # gamma = line_search_log_reg(y, tx, w, initial_gamma)
        
        #print("Logistic gd regression({bi}/{ti}): time taken = {t} s".format(bi=k, ti=max_iters, t = t))
        k = k+1
    return w, loss
    
def cross_validation_lr(y, tx, k_fold, initial_w, max_iters, initial_gamma):
    k_indices = build_k_indices(y, k_fold, seed = 44)
    w = []
    for k in range(k_fold):
    # Split the data
        tx_test = tx[k_indices[k,:],:]
        y_test = y[k_indices[k,:]]
        tx_train = tx[[i not in k_indices[k,:] for i in range(tx.shape[0])],:]
        y_train = y[[i not in k_indices[k,:] for i in range(tx.shape[0])]]
        #raise NotImplementedError
                                
    # Perform logistic regression with gradient descent
        w_lr, loss_lr = logistic_regression_gd(y_train, tx_train, initial_w, max_iters, initial_gamma)
        w.append(w_lr)
    w = np.mean(w)
    return w

def log_reg_ridge(y, tx, w, lambda_):
    loss, grad, hess = compute_log_likeli(y, tx, w), compute_grad_log_likeli(y, tx, w), compute_hess_log_likeli(y, tx, w)
    loss = loss + lambda_*np.linalg.norm(w)**2
    grad = grad + 2*lambda_*w
    hess = hess + 2*lambda_*np.eye(hess.shape[0])
    return loss, grad, hess

def logistic_regression_ridge(y, tx, lambda_, ini_w, max_iters=10, gamma=1e-4, tol = 1e-6):
    k = 1
    w = ini_w
    _, grad, hess = log_reg_ridge(y, tx, w, lambda_)
    while k <= max_iters and np.linalg.norm(grad) > tol:
        w = w - gamma*np.linalg.lstsq(hess, grad, rcond = None)[0]
        _, grad, hess = log_reg_ridge(y, tx, w, lambda_)
        k = k+1
    loss, _, _ = log_reg_ridge(y, tx, w, lambda_)
    return w, loss