import numpy as np
from data_preparation_two_layers import T_5 as T, P_5 as P


def sigmoid(x, beta=1):
    return 1 / (1 + np.exp(-beta * x))


def init2(S, K1, K2):
    # funkcja tworzy sieć jednowarstwową
    # i wypełnia jej macierz wag wartościami losowymi
    # z zakresu od -0.1 do 0.1
    # parametry: S - liczba wejść do sieci
    # K - liczba neuronów w warstwie
    # wynik: W - macierz wag sieci
    W1 = np.random.uniform(-0.1, 0.1, (S + 1, K1))
    W2 = np.random.uniform(-0.1, 0.1, (K1 + 1, K2))
    return W1, W2


def sim2(W1, W2, X):
    # calculates output of a two-layer net for a given input
    # parameters: W1 – weight matrix for layer 1
    # W2 – weight matrix for layer 2
    # X – input vector ( to the net / layer 1 )
    # result: Y1 – output vector for layer 1 ( useful for training )
    # Y2 – output vector for layer 2 / the net
    beta = 5
    X1 = np.insert(X, 0, -1)
    U1 = np.dot(W1.T, X1)
    Y1 = 1 / (1 + np.exp(-beta * U1))
    X2 = np.insert(Y1, 0, -1)
    U2 = np.dot(W2.T, X2)
    Y2 = 1 / (1 + np.exp(-beta * U2))
    return Y1, Y2

def train2(W1before, W2before, P, T, n):
    noExamples = P.shape[1]
    W1 = W1before.copy()
    W2 = W2before.copy()
    lr = 0.1
    beta = 5

    print(len(W1))
    print(len(W2))
    for i in range(n):
        exampleNo = np.random.randint(1, noExamples + 1)  # draw a random example number
        X = P[:, exampleNo - 1]  # present the chosen example and calculate output
        X1 = np.insert(X, 0, -1)
        Y1, Y2 = sim2(W1, W2, X)
        X2 = np.insert(Y1, 0, -1)
        D2 = T[exampleNo - 1] - Y2
        #  D2 = T[:, exampleNo - 1] - Y2  # calculate errors for layer 2
        E2 = beta * D2 * Y2 * (1 - Y2)  # "internal" error for layer 2
        D1 = np.dot(W2[1:, :], E2)  # calculate errors for layer 1 (backpropagation)
        E1 = beta * D1 * Y1 * (1 - Y1)  # "internal" error for layer 1

        dW1 = lr * np.outer(X1, E1)  # calculate weight adjustments for layers 1 & 2
        dW2 = lr * np.outer(X2, E2)

        W1 += dW1  # add adjustments to weights in both layers
        W2 += dW2

    W1after = W1
    W2after = W2

    return W1after, W2after


def init_weights(S, *K):
    # funkcja tworzy sieć wielowarstwową
    # i wypełnia jej macierze wag wartościami losowymi
    # z zakresu od -0.1 do 0.1
    # parametry: S - liczba wejść do sieci
    # *K - liczba neuronów w każdej warstwie (można przekazać dowolną ilość argumentów)
    # wynik: lista macierzy wag sieci
    num_layers = len(K)
    weights = [np.random.uniform(-0.1, 0.1, (S + 1, K[0]))]

    for i in range( num_layers - 1):
        W = np.random.uniform(-0.1, 0.1, (K[i] + 1, K[i + 1]))
        weights.append(W)
        # print('i ', i, weights[0])
    return weights



def sim(X, W):
    # calculates output of a multi-layer net for a given input
    # parameters: X – input vector to the net / layer 1
    # *W – list of weight matrices for each layer
    # result: Y – output vector for the final layer / the net

    beta = 5
    input_vector = X.copy()
    # input_vector = np.insert(X, 0, -1)
    for i, weights in enumerate(W):

        input_vector = np.insert(input_vector, 0, -1)
        net_input = np.dot(weights.T, input_vector)
        output_vector = 1 / (1 + np.exp(-beta * net_input))
        input_vector = output_vector

    return output_vector



if __name__ == '__main__':
    n = 5000
    '''
    P = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    T = np.array([0, 1, 1, 0])
    '''
    Y_before_list = []
    #W1before, W2before = init2(2, 3, 1)
    W1before, W2before = init2(2, 3, 1)
    W = init_weights(2, 3, 1)

    for column in range(P.shape[1]):
        Y2 = sim(P[:, column], W)
        Y_before_list.append(Y2[0])
    print("Y_before_list")
    print(Y_before_list)

    print("W1 ", len(W[0]))
    print("W2 ", len(W[1]))

    W1after, W2after = train2(W[0], W[1], P, T, n)
    Y1, Y2a = sim2(W1after, W2after, P[:, 0])
    Y1, Y2b = sim2(W1after, W2after, P[:, 1])
    Y1, Y2c = sim2(W1after, W2after, P[:, 2])
    Y1, Y2d = sim2(W1after, W2after, P[:, 3])
    Yafter = [Y2a, Y2b, Y2c, Y2d]
    print(Yafter)

    for i in range(P.shape[1]):
        X = P[:, i]
        R = T[i]
        Y1, Y2 = sim2(W1after, W2after, X)
        print(f"Input: {X}, Predicted Output: {Y2}, Real Output {R}")


    W1before, W2before = W[0], W[1]
    Y1, Y2a = sim2(W1before, W2before, P[:, 0])

    Y1, Y2b = sim2(W1before, W2before, P[:, 1])
    Y1, Y2c = sim2(W1before, W2before, P[:, 2])
    Y1, Y2d = sim2(W1before, W2before, P[:, 3])
    Ybefore = [Y2a, Y2b, Y2c, Y2d]
    W1after, W2after = train2(W1before, W2before, P, T, n)
    Y1, Y2a = sim2(W1after, W2after, P[:, 0])
    Y1, Y2b = sim2(W1after, W2after, P[:, 1])
    Y1, Y2c = sim2(W1after, W2after, P[:, 2])
    Y1, Y2d = sim2(W1after, W2after, P[:, 3])
    Yafter = [Y2a, Y2b, Y2c, Y2d]
    print(Yafter)
'''
    for i in range(P.shape[1]):
        X = P[:, i]
        Y1, Y2 = sim2(W1after, W2after, X)
        print(f"Input: {X}, Predicted Output: {Y2}")
'''