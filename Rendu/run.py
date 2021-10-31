import numpy as np
from implementations import *
from support_code import *
from proj1_helpers import *
import os.path



def main():

    DATA_TRAIN_PATH = './data/train.csv' # TODO: download train data and supply path here 
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)


    X, Y, missing_vectors = divide_set(tX,y)


    weightss = []
    degrees = []
    errors = []
    cols = []
    Means = []
    STDS = []

    degress = np.arange(4,17)
    lambdas = np.logspace(-12,-3,num = 40)

    for i in range(4):
        print("index :" + str(i))
        x_n = X[i]
        y_n = Y[i]
        x,col = cleanse_data(x_n)
        x,mean,std = standardize(x)

        min_degree, min_weight,min_error = find_best(x,y_n,degress,lambdas,missing_vectors[i])
        print("found the min error" + str(min_error) + "  for this index")
        weightss.append(min_weight)
        degrees.append(min_degree)
        errors.append(min_error)
        cols.append(col)
        Means.append(mean)
        STDS.append(std)


    DATA_TEST_PATH = './data/test.csv' # TODO: download train data and supply path here 
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

    Y_pred = np.zeros(tX_test.shape[0])
    X_t, _,_ = divide_set(tX_test,Y_pred)

    for i in range(4):
        X_test = cleanse_data_col(X_t[i],cols[i])
        X_test = standardize_with_info(X_test,Means[i],STDS[i])
        X_test_poly = build_poly(X_test,degrees[i])
        print(X_test_poly.shape)
        y_a = predict_labels(weightss[i], X_test_poly)
        print(y_a.shape)
        Y_pred[ tX_test[:,22] == i] = y_a


    OUTPUT_PATH = './final_submission.csv'
    create_csv_submission(ids_test, Y_pred, OUTPUT_PATH)
    print('FINISHED')


def find_best(tx,y,degrees, lambdas,missing_vector):

    min_degree = degrees[0]
    min_weight = np.array({})
    min_error = 1

    for deg in degrees:
        tx_poly = build_poly(tx,deg)
        #tx_poly = np.concatenate((missing_vector, tx_poly),1)
        tx_tr,tx_te,y_tr,y_te = split_data(tx_poly, y, 0.9,4)
        index = 0
        for lambda_ in lambdas:
            weights_r, error_r =  ridge_regression(y_tr,tx_tr,lambda_)
            error = loss_really(weights_r,y_te,tx_te)
            if error < min_error:
                min_degree = deg
                min_error = error
                min_weight = weights_r

    return min_degree, min_weight,min_error



main()