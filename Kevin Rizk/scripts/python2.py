import numpy as np
from proj1_helpers import *


def standardize(x):
    ''' fill your code in here...
    '''
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    
    return std_data

def loss_really(weights,y_te,tx_te):    
    y_pred = predict_labels(weights, tx_te)
    print(y_pred)
    s = y_pred != y_te
    return sum(s)/len(y_te)


def build_poly(tX,d):

    tx = np.ones((tX.shape[0],1))
    for i in range(d):
        tx = np.concatenate((tx,np.power(tX,i+1)),1)
    return tx



def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    e = y - np.dot(tx,w)
    return 0.5*(1/np.size(y))*np.linalg.norm(e)**2


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient and error vector
    
    e = y - np.dot(tx,w)
    return - 1/np.size(y)*np.dot(tx.transpose(),e)

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    # ***************************************************
    e = y - np.dot(tx,w)
    return - 1/np.size(y)*np.dot(tx.transpose(),e)

def least_squares(y, tx):
    """calculate the least squares solution."""
    A = np.dot(tx.T,tx)
    b = np.dot(tx.T,y)
    w = np.linalg.solve(A,b)
    mse = compute_loss(y,tx,w)
    return w, mse


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient descent.
    # ***************************************************
    ws = [initial_w]
    losses = []
    w = initial_w
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_iters):
        
        gradient = compute_gradient(minibatch_y, minibatch_tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma*gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        
    return losses, ws


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        # ***************************************************
        w = w - gamma*gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    
    A = np.dot(tx.T,tx) + lambda_*2*len(y)*np.eye(tx.shape[1])
    b = np.dot(tx.T,y)
    w = np.linalg.solve(A,b)
    mse = compute_loss(y,tx,w)
    return w,mse

def sigmoid(t):

    return 1.0 / (1.0 + np.exp(-t))

def calculate_loss_sigmoid(y, tx, w):
    """compute the loss: negative log likelihood."""
    S = sigmoid(np.dot(tx,w))
    n = len(y)
    return 1./n*(- y @ log_plus(S) - (1-y) @ log_plus(1-S))

def calculate_gradient_sigmoid(y, tx, w):
    n = len(y)
    return 1./n*np.dot(tx.T, sigmoid(np.dot(tx,w)) - y) 


def learning_by_gradient_descent(y, tx, w, gamma):

    loss = calculate_loss_sigmoid(y,tx,w)

    gradient = calculate_gradient_sigmoid(y,tx,w)

    w = w - gamma*gradient

    return loss, w

def logistic_regression(y,tx,initial_w,max_iters,gamma):

    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):

        loss , w = learning_by_gradient_descent(y,tx,w,gamma)
        losses.append(loss)
        ws.append(w)

    return losses, ws[-1]

def log_plus(x):
    y = x
    y[y < 1e-10] = 1e-10
    return(np.log(y))


