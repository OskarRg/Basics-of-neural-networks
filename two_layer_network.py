import numpy as np
from functions import sigmoid, init2, sim2, train2



'''  # To są uogólnione funkcje, działają poprawnie, ale używam na razie tych specialnie do 2 warstw.
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
'''


if __name__ == '__main__':
    n = 5000
    P = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    T = np.array([0, 1, 1, 0])
    Y_before_list = []
    #W1before, W2before = init2(2, 3, 1)
    W1before, W2before = init2(2, 3, 1)
    W = init2(2, 3, 1)

    for column in range(P.shape[1]):
        Y2 = sim2(W[0], W[1], P[:, column])
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
        Y1, Y2 = sim2(W1after, W2after, X)
        print(f"Input: {X}, Predicted Output: {Y2}")


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