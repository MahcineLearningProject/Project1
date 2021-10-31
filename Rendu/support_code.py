import numpy as np
from proj1_helpers import *



def cleanse_data(tx):
    """ A function that will apply log transformation to 
    data which has some a long tail and returns the columns for which these
    transofrmation was done """
    col =[]
    for i in range(tx.shape[1]):
        if np.std(tx[:,i]) > 10:
            col.append(i)
    return cleanse_data_col(tx,col), col



def cleanse_data_col(tx,cols):
    """ Apply log transformation to the specified columns in the argumnent"""
    l = tx.copy()
    for i in cols:
            l[:,i] = np.log(l[:,i]+1)
    return l




def divide_set (tx,y):
    """ Divide the set into 4 smaller depending of the value of the column 22 and 
    removes the column where there is undefinied set. 
    Returb the 4 different set with vectors containing the missing value of the first column """

    X = []
    Y = []
    Missing_Vectors = []

    for i in range(4):

        s = tx[tx[:,22] == i]

        # indices to remove and delete them
        ind_c = np.append(0,np.arange(30)[(np.sum(s == -999, 0) == 0)])
        to_remove = np.argwhere(ind_c == 22)
        ind_c = np.delete(ind_c,to_remove)

        if i == 0:
            #remove the last argument if (i==0) so the set is defined.
            ind_c = np.delete(ind_c,-1)
        s = s[:,ind_c]
        #reokace the missing vakue
        s = np.where(s == -999 , 0, s)

        # create the missing vector
        missing_vector = 1*(s[:,[0]] == -999)
        


        #append the new data
        X.append(s)
        Y.append(s)
        Missing_Vectors.append(missing_vector)
    



    return X,Y,Missing_Vectors


def balance_data(x,y,seed=1):
    """ This code will remove some subset to make the code more balanced, in fact it will 
    return the new baalnced x and y"""

    #randomnly shuffle to make it more random
    np.random.seed(seed)

    shuffle_indices = np.random.permutation(np.arange(len(y)))
    shuffled_y = y[shuffle_indices]
    shuffled_tx = x[shuffle_indices]

    a1 = np.sum(shuffled_y==1)
    count = 0
    indices = []

    #balance the data
    for i in range(len(shuffled_y)):
        if shuffled_y[i] == 1:
            indices.append(i)
        else:
            if count < a1:
                indices.append(i)
                count = count +1

    return (shuffled_tx[indices ,: ], shuffled_y[indices]) 



def standardize_with_info(x, means, stds):
    """
        standarize the data when we know the means that will substract 
        and the deviation that we will divide from
    """
    centered_data = x - means
    std_data = centered_data / stds

    return std_data



def standardize(x):
    ''' 
        standardize the data with the mean and std if each column, and return the means and 
        the standart divation.
    '''
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)


    return standardize_with_info(x,means,stds) , means, stds


def loss_really(weights,y_te,tx_te): 
    '''
        calculate the real accuarcty  
    '''
    y_pred = predict_labels(weights, tx_te)
    s = y_pred != y_te
    return sum(s)/len(y_te)


def build_poly(x, degree):

    """polynomial basis functions for input data x, for j=0 up to j=degree and add offset 
    column """
    N = x.shape[0]
    pol_X = np.ones((N,1))
    for i in range(degree):
        pol_X = np.concatenate((pol_X, np.power(x,i+1)), axis = 1)
    return pol_X




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

    #create and divide data
    tx_tr = shuffled_tx[0:n,:]
    tx_te = shuffled_tx[n+1:,:]
    y_tr = shuffled_y[0:n]
    y_te = shuffled_y[n+1:]

    return tx_tr,tx_te,y_tr,y_te


def log_plus(x):
    """
    if x is too small replace by another value
    """
    y = x
    y[y < 1e-50] = 1e-50
    return(np.log(y))


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

