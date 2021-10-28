# -*- coding: utf-8 -*-
"""Gradient Descent"""
import numpy as np

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = y.shape[0]
    e = y - np.dot(tx,w)
    return -1/N*np.dot(tx.T,e)
    raise NotImplementedError
    
def compute_loss(y, tx, w, mse = True): # set false to use MAE instead
    """Calculate the loss."""
    N = y.shape[0]
    if mse:
        L = (np.linalg.norm(y-np.dot(tx,w)))**2/(2*N)
    else:
        e = y - np.dot(tx,w)
        L = 1/N*np.sum(np.absolute(e))
    return L
    raise NotImplementedError

def line_search(y, tx, w, gamma_ini = 1/4, rho = 0.5):
    gamma = gamma_ini
    g = compute_gradient(y, tx, w)
    loss = compute_loss(y, tx, w)
    f_suiv = compute_loss(y, tx, w-gamma*g)
    k = 1
    while (f_suiv > loss - 1e-4*gamma*np.dot(g.T,g) and k<20): # restriction on the number of iterations to save time
        gamma = rho*gamma
        f_suiv = compute_loss(y, tx, w-gamma*g)
        k = k+1
    return gamma


def gradient_descent(y, tx, initial_w, max_iters, gamma_ini = 1/4, rho = 0.5):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        # we compute the best descent gamma via line search
        gamma = line_search(y, tx, w, gamma_ini, rho) 
        # raise NotImplementedError
        # INSERT YOUR CODE HERE
        w = w - gamma*grad
        loss = compute_loss(y, tx, w)
        # ***************************************************
        # raise NotImplementedError
        print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return loss, w

