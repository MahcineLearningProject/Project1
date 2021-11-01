import numpy as np
from proj1_helpers import *
from support_code import *



# ALL THE LOSS FUNCTION USED HERE 
#***************************************************************************


def compute_loss_least_square(y, tx, w):
    """Calculate the loss.
    of the least sqaure
    """
    # ***************************************************
    e = y - np.dot(tx, w)
    N = y.shape[0]
    loss = 1/(2*N) * np.dot(e.T, e)
    return loss
    # ***************************************************

def compute_gradient_least_sqaure(y, tx, w):
    """Compute the gradient of the least sqaure error"""

    e = y - np.dot(tx, w)
    N = y.shape[0]
    grad = -1/(2*N) * np.dot(tx.T, e)
    return grad
    # ***************************************************


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""

    # ***************************************************
    e = y - np.dot(tx,w)
    return - 1/np.size(y)*np.dot(tx.transpose(),e)
    # ***************************************************


def calculate_loss_sigmoid(y, tx, w):
    """compute the loss: negative log likelihood."""
    m = len(y)
    S = sigmoid(tx @ w)
    return - y @ log_plus(S) - (1-y) @ log_plus(1-S)

#***************************************************************************

# ALL THE GRADIENT FUNCTION USED HERE 

#***************************************************************************



def calculate_gradient_sigmoid(y, tx, w):
    """ compute the gradiant of the sigmoid loss """
    m = len(y)
    return tx.T @ (sigmoid(tx @ w) - y)

def sigmoid(t):
    """apply the sigmoid function on t or a matrix or vector also"""

    return 1.0 / (1.0 + np.exp(-t))

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient, and Hessian: TODO
    loss = calculate_loss(y, tx, w) + lambda_ * np.dot(w.T, w)
    grad = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, grad
    # ***************************************************

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """

    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * grad
    # ***************************************************
    return loss, w



def learning_by_gradient_descent_logistic(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w return loss 
    """

    grad = calculate_gradient_sigmoid(y, tx, w)
    w = w - gamma * grad
    loss = calculate_loss_sigmoid(y, tx, w)

    return loss, w 
    



#***************************************************************************

### HERE WILL BE THE 5 ASKED METHODS TO IMPLEMENT

#***************************************************************************
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm using least sqaure.
    Return the found weught and the last loss"""

    w = initial_w

    for n_iter in range(max_iters):
        grad = compute_gradient_least_sqaure(y, tx, w)
        w = w - gamma*grad
        
    loss = compute_loss_least_square(y, tx, w)

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm with batch size 1"""

    w = initial_w

    for minibatch_y, minibatch_tx in batch_iter(y, tx, 1, max_iters):
        grad = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        w = w - gamma*grad
    loss = compute_loss_least_square(y, tx, w)

    return w, loss
    # ***************************************************


def least_squares(y, tx):
    """calculate the least squares solution using normal equations
    and return the weight and the loss"""

    A = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    w = np.linalg.solve(A, b)
    loss = compute_loss_least_square(y, tx, w)
    return w, loss
    # ***************************************************
    
def ridge_regression(y, tx, lambda_):
    """calculate ridge regression using normal equations
    and return the weight and the loss """

    A = np.dot(tx.T,tx) + lambda_*2*len(y)*np.eye(tx.shape[1])
    b = np.dot(tx.T,y)
    w = np.linalg.solve(A,b)
    loss = compute_loss_least_square(y,tx,w)+ lambda_ * np.dot(w.T, w)

    return w, loss
    # ***************************************************

def logistic_regression(y, tx, initial_w, max_iter, gamma):
    """ logistic regression using gradient method""" 
    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent_logistic(y, tx, w, gamma)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, gamma):
    """ regularized logistic regression using gradient method""" 
    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss


