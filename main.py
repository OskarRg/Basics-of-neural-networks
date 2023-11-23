import numpy as np
from data_preparation import P_cleaned as P, T_cleaned as T
from functions import sigmoid, init2 as init, sim2, train2 as train2, init_weights, sim, train

if __name__ == '__main__':

    n = 5000
    Y_before_list = []
    W = init_weights(13, 3, 1)
    print("W")
    print(W)
    print(len(W[0]), len(W[1]))
    for column in range(P.shape[1]):
        Y2 = sim(P[:, column], W)
        Y_before_list.append(Y2[-1][0])
    print("Y_before_list")
    print(len(Y_before_list))
    print(Y_before_list)

    print("W1 ", len(W[0]))
    print("W2 ", len(W[1]))
    # Do tego punktu chyba działa dobrze, train() jest zepsute
    Wafter = train(W, P, T, 10)
    for column in range(P.shape[1]):
        Y2 = sim(P[:, column], Wafter)
        Y_before_list.append(Y2[0])
    print("Y_before_list")
    print(len(Y_before_list))
    print(Y_before_list)
    '''
    W1after, W2after = train(W1before, W2before, P, T, 10)
    Y_after_list = []
    for column in range(P.shape[1]):
        Y1, Y2 = sim(W1before, W2before, P[:, column])
        Y_after_list.append(Y2)
    print(Y_after_list)
    '''
