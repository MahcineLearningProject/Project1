import time
from helpers import *
from gradient_descent import compute_loss
from plots import cross_validation_visualization

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def change_data(tx):
    #22 to get pri_jet
    # 11 columns contain undefined numbers : 0,4,5,6,12,26,27,28, if pri <=1 ; 23,24,25 if pri=0
    missing = tx[:,[0,4,5,6,12,23,24,25,26,27,28]]
    N = tx.shape[0]
    d = tx.shape[1]
    
    for i in range(tx.shape[0]):
        if tx[i,22]==0:
            tx[i,[23,24,25]] = 0
        if tx[i,22]<=1:
            tx[i,[4,5,6,12,26,27,28]] = 0
        if np.abs(tx[i,0])==999:
            tx[i,0]=0
            
    # Create the missing vectors
    for i in range(missing.shape[0]):
        for j in range(missing.shape[1]):
            if np.abs(missing[i,j])==999:
                missing[i,j] = 1
            else:
                missing[i,j] = 0

    # Extract the 3 differents missing vectors
    add_missing_col = missing[:,0].reshape((N,1))
    for i in range(1,missing.shape[1]):
        j = 0
        while j < add_missing_col.shape[1]:
            if np.array_equal(missing[:,i],add_missing_col[:,j]):
                break
            j = j+1
            if j == add_missing_col.shape[1]:
                add_missing_col = np.append(add_missing_col,missing[:,i].reshape((N,1)),axis=1)    
    return tx, add_missing_col

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ run linear regression using gradient descent with step size gamma"""
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        
        raise NotImplementedError
        w = w - gamma*grad
        
    return loss, w
    raise NotImplementedError
    
def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        for mini_batch_y, mini_batch_tx in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
            loss = compute_loss(mini_batch_y, mini_batch_tx, w)
            grad = compute_stoch_gradient(mini_batch_y, mini_batch_tx, w)
            w = w - gamma*grad
        raise NotImplementedError
    # raise NotImplementedError
    return loss, w

def least_squares(y, tx):
    """calculate the least squares solution."""
    w_opt = np.linalg.solve(tx.T @ tx, tx.T @ y)
    N = len(y)
    e = y- tx @ w_opt
    mse = (e.T @ e)/(2*N)
    return w_opt, mse
    raise NotImplementedError
    
def split_data(y, tX, ratio, seed=44):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    shuffled_y = y[shuffle_indices]
    shuffled_tX = tX[shuffle_indices,:]
    # SPLIT
    split = int(np.floor(ratio*len(y)))
    y_train, y_test = shuffled_y[0:split], shuffled_y[split:]
    tX_train, tX_test = shuffled_tX[0:split,:], shuffled_tX[split:,:]
    # ***************************************************
    return y_train, y_test, tX_train, tX_test
    raise NotImplementedError
    
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = len(y)
    w_ri = np.linalg.lstsq(tx.T @ tx + lambda_*(2*N)*np.eye(tx.shape[1]), tx.T @ y, rcond = None)[0]
    loss = 1/(2*N)*np.linalg.norm(y-tx @ w_ri)**2+lambda_*(w_ri.T @ w_ri)
    
    return w_ri, loss
    raise NotImplementedError
    
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold (number of folds)."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    
    return np.array(k_indices) # return a matrix. K-th row contains indices corresponding to the k-th fold
    raise NotImplementedError
    
def build_poly(tx, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    tx_new = tx.copy()
    tx_new, missing = change_data(tx_new)
    N = tx_new.shape[0]
    d = tx_new.shape[1]  
    
    # Create polynomial
    phi = np.ones((N,d*(degree+1)+3))
    k=1
    while k<=degree:
        for i in range(d):
            phi[:,d*k+i] = tx[:,i]**k
        k = k+1
        
    # Add 3 features corresponding to missing vectors
    for i in range(missing.shape[1]):
        phi[:,-i-1] = missing[:,i]
    
    return phi
    # ***************************************************
    raise NotImplementedError
    
def cross_validation(y, tx, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train:
    k_fold = k_indices.shape[0]
    tx_test = tx[k_indices[k,:],:]
    y_test = y[k_indices[k,:]]
    tx_train = tx[[i not in k_indices[k,:] for i in range(tx.shape[0])],:]
    y_train = y[[i not in k_indices[k,:] for i in range(tx.shape[0])]]
    raise NotImplementedError
    
    # ridge regression:
    w_ri, loss_ri = ridge_regression(y_train, tx_train, lambda_)
    raise NotImplementedError
    
    # calculate the loss for train and test data:
    loss_tr = compute_loss(y_train, tx_train, w_ri, mse = True)
    loss_te = compute_loss(y_test, tx_test, w_ri, mse = True)
    raise NotImplementedError
    return loss_tr, loss_te

def cross_validation_demo(y, tx, seed = 1, degree = 7, k_fold = 4):
    lambdas = np.logspace(-4, 0, 40)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # cross validation:
    i = 1
    for lambda_ in lambdas:
        start = time.time()
        sum_tr = []
        sum_te = []
        for k in range(k_fold):
            l_tr, l_te = cross_validation(y, tx, k_indices, k, lambda_, degree)
            sum_tr.append(l_tr), sum_te.append(l_te)
        loss_tr = np.mean(sum_tr)
        loss_te = np.mean(sum_te)
        rmse_tr.append(np.sqrt(2*loss_tr))
        rmse_te.append(np.sqrt(2*loss_te))
        t = time.time() - start
        print("Cross_validation({bi}/{ti}): time taken = {t} seconds".format(bi=i, ti=len(lambdas), t = t))
        i = i+1
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    m = np.argmin(rmse_tr)
    return lambdas[m]
    