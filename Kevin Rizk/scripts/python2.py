import numpy as np
from proj1_helpers import *


def cleanse_data(tx):
    col =[]
    for i in range(tx.shape[1]):
        if np.std(tx[:,i]) > 10:
            col.append(i)
    return cleanse_data_col(tx,col), col

def cleanse_data_col(tx,cols):
    l = tx.copy()
    for i in cols:
            l[:,i] = np.log(l[:,i]+1)
    return l

def tree_test(tx,y):

    X = []
    Y = []
    Missing_Vectors = []
    for i in range(4):
        s_0 = tx[tx[:,22] == i]
        y_0 = y[tx[:,22] == i]
        ind_c = np.append(0,np.arange(30)[(np.sum(s_0 == -999, 0) == 0)])
        to_remove = np.argwhere(ind_c == 22)
        ind_c = np.delete(ind_c,to_remove)
        if i == 0:
            ind_c = np.delete(ind_c,-1)
        s_0 = s_0[:,ind_c]
        missing_vector = 1*(s_0[:,[0]] == -999)
        s_0 = np.where(s_0 == -999 , 0, s_0)
        X.append(s_0)
        Y.append(y_0)
        Missing_Vectors.append(missing_vector)
    



    return X,Y,Missing_Vectors


def balance_data_stupid(x,y,seed=6):

    np.random.seed(seed)

    shuffle_indices = np.random.permutation(np.arange(len(y)))
    shuffled_y = y[shuffle_indices]
    shuffled_tx = x[shuffle_indices]

    a1 = np.sum(shuffled_y==1)
    count = 0
    indices = []
    for i in range(len(shuffled_y)):
        if shuffled_y[i] == 1:
            indices.append(i)
        else:
            if count < a1:
                indices.append(i)
                count = count +1
    return (shuffled_tx[indices ,: ], shuffled_y[indices]) 


def split_data(x, y, ratio, seed=6):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    data_size = len(y)
    n = int(ratio*data_size)
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_y = y[shuffle_indices]
    shuffled_tx = x[shuffle_indices]
    tx_tr = shuffled_tx[0:n,:]
    tx_te = shuffled_tx[n+1:,:]
    y_tr = shuffled_y[0:n]
    y_te = shuffled_y[n+1:]
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    return tx_tr,tx_te,y_tr,y_te

def standardize_with_info(x, means, stds):
    ''' fill your code in here...

    '''
    centered_data = x - means
    std_data = centered_data / stds

    return std_data



def standardize(x):
    ''' fill your code in here...

    '''
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)


    return standardize_with_info(x,means,stds) , means, stds


def loss_really(weights,y_te,tx_te):    
    y_pred = predict_labels(weights, tx_te)
    s = y_pred != y_te
    return sum(s)/len(y_te)


def new_build(tX):
    n = tX.shape[0]
    tx = np.ones((n,1))
    for i in range(tX.shape[1]):
        print(i)
        tx = np.concatenate((tx,tX[:,i:]*tX[:,i:i+1]),1)
    return tx



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
    m = len(y)
    S = sigmoid(tx @ w)
    return (1/m)*(- y @ log_plus(S) - (1-y) @ log_plus(1-S))

def calculate_gradient_sigmoid(y, tx, w):
    m = len(y)
    return (1/m)*(tx.T @ (sigmoid(tx @ w) - y))


def learning_by_gradient_descent(y, tx, w, gamma):
    loss = calculate_loss_sigmoid(y,tx,w)
    gradient = calculate_gradient_sigmoid(y,tx,w)
    #hessian = calculate_hessian(y,tx,w)
    w = w-gamma*gradient
    #w = np.linalg.solve( hessian, c)

    return loss, w

def logistic_regression(y,tx,initial_w,max_iters,gamma):

    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss , w = learning_by_gradient_descent(y,tx,w,gamma)
        losses.append(loss)
        ws.append(w)
        if (n_iter % 10 == 0):
            print(loss)

    return losses, ws[-1]

def log_plus(x):
    y = x
    y[y < 1e-50] = 1e-50
    return(np.log(y))


def calculate_hessian(y, tx, w):
    S = sigmoid(tx@w).reshape(-1,1)
    S = S*(1-S)
    return tx.T @ (S * tx)

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    loss, gradient, hessian = logistic_regression(y,tx,w)
    loss = loss + lambda_*np.linalg.norm(w,2)
    gradient = gradient + lambda_*w
    return loss,gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient: TODO
    # ***************************************************
    gradient,hessian = penalized_logistic_regression(y, tx, w, lambda_)
    
    w = w - gamma*gradient
    
    return loss, w


def pen_logistic_regression(y,tx,initial_w,max_iters,gamma):

    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss , w = learning_by_gradient_descent(y,tx,w,gamma)
        losses.append(loss)
        ws.append(w)
        if (n_iter % 10 == 0):
            print(loss)

    return losses, ws[-1]


