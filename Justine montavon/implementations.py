import numpy as np
from proj1_helpers import *

def build_poly_plus_missing(tX, degree, missing_vectors):
    N = tX.shape[0]
    polx = build_poly(tX, degree)
    polx = polx.reshape(N, -1)
    polx = np.append(polx, missing_vectors, axis = 1)
    return polx

def missing(tX):
    # change the -999 to 0 in tX and 
    # create new columns missing_vectors = 1 if it misses the data for the actual variable and 0 otherwise
    print(tX[tX == -999].shape)
    missing_vectors = np.array([])
    n_par = tX.shape[1]
    for i in range(n_par):
        col = tX[:,i]
        if (col[col==-999].shape[0] != 0):
            miss = col.copy()
            miss[miss != -999] = 0
            miss[miss == -999] = 1
            col[col == -999] = 0
            miss = miss.reshape(-1, 1)
            if (missing_vectors.shape[0] == 0):
                missing_vectors = miss
            else:
                 missing_vectors = np.append(missing_vectors, miss, axis = 1)
    print(tX[tX == -999].shape)
    missing_vectors = missing_vectors[:, (0, 1, 5)] # only take differents columns of missing_vectors
    return tX, missing_vectors

def loss_really(weights,y_te,tx_te):    
    y_pred = predict_labels(weights, tx_te)
    s = y_pred != y_te
    return sum(s)/len(y_te)


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    N = x.shape[0]
    pol_X = np.ones((N,1))
    for i in range(degree):
        x_ap = x**(i+1)
        pol_X = np.concatenate((pol_X, x_ap), axis = 1)
    return pol_X
    # ***************************************************
    raise NotImplementedError

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x

def compute_loss(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    e = y - np.dot(tx, w)
    N = e.shape[0]
    loss = 1/(2*N) * np.dot(e.T, e)
    return loss
    # ***************************************************

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient and error vector
    e = y - np.dot(tx, w)
    N = e.shape[0]
    grad = -1/(2*N) * np.dot(tx.T, e)
    return grad
    # ***************************************************

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        grad = compute_gradient(y, tx, w)
        # ***************************************************
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        w = w - gamma*grad
        # ***************************************************
        
    loss = compute_loss(y, tx, w)
    return w, loss


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    e = y - np.dot(tx, w)
    N = np.size(y)
    grad = -1/(2*N) * np.dot(tx.T, e)
    return grad
    # ***************************************************


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient descent.
    w = initial_w
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 1, max_iters):
        grad = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        w = w - gamma*grad
    loss = compute_loss(y, tx, w)
    return w, loss
    # ***************************************************
    

def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    A = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    w = np.linalg.solve(A, b)
    loss = compute_loss(y, tx, w)
    return w, loss
    # ***************************************************
    
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    N = len(y)
    d = tx.shape[1]
    A = np.dot(tx.T, tx) + (lambda_*2*N) * np.eye(d)
    b = np.dot(tx.T, y)
    w = np.linalg.solve(A, b)
    loss = compute_loss(y,tx,w)
    return w, loss
    # ***************************************************

def sigmoid(t):
    """apply the sigmoid function on t."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    sigm = 1/(1+np.exp(-t))
    return sigm
    # ***************************************************
    raise NotImplementedError
    
def log_plus(t):
    y = t.copy()
    y[y<10e-10] = 10e-10
    return np.log(y)


def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    sigm = sigmoid(np.dot(tx, w))
    l_sigm1 = log_plus(sigm)
    l_sigm2 = log_plus(1-sigm)
    L = - np.dot(y.T, l_sigm1) - np.dot((1-y).T, l_sigm2)
    return L
    # ***************************************************
    raise NotImplementedError

""" def calculate_loss(y, tx, w):
    compute the loss: negative log likelihood.
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    sigm = sigmoid(np.dot(tx, w))
    l_sigm1 = np.log(sigm)
    l_sigm2 = np.log(1-sigm)
    L = - np.dot(y.T, l_sigm1) - np.dot((1-y).T, l_sigm2)
    return L
    # ***************************************************
    raise NotImplementedError """
    
def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    sigm = sigmoid(np.dot(tx, w))
    grad = np.dot(tx.T, (sigm-y) )
    return grad
    # ***************************************************
    
def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # compute the loss: TODO
    loss = calculate_loss(y, tx, w)
    # ***************************************************
    # ***************************************************
    # INSERT YOUR CODE HERE
    # compute the gradient: TODO
    grad = calculate_gradient(y, tx, w)
    # ***************************************************
    # ***************************************************
    # INSERT YOUR CODE HERE
    # update w: TODO
    w = w - gamma * grad
    # ***************************************************
    return loss, w 
    
def logistic_regression(y, tx, w, max_iter, gamma):
    # init parameters
    threshold = 1e-8
    losses = []
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    return w, loss

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
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient: TODO
    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
    # ***************************************************
    # ***************************************************
    # INSERT YOUR CODE HERE
    # update w: TODO
    w = w - gamma * grad
    # ***************************************************
    return loss, w

def reg_logistic_regression(y, tx, lambda_, w, max_iter, gamma):
    # init parameters
    threshold = 1e-8
    losses = []

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss


