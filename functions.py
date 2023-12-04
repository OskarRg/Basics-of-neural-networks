import numpy as np


def sigmoid(x, beta=1):
    return 1 / (1 + np.exp(-beta * x))


def relu(x):  # Nw czy dobrze - psrawdze kiedyś
    return np.maximum(0, x)


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
    Y1 = 1 / (1 + np.exp(-beta * U1)) # używamy funkcji aktywacji ReLU dla warstwy ukrytej
    # Y1 = np.maximum(0, U1)
    X2 = np.insert(Y1, 0, -1)
    U2 = np.dot(W2.T, X2)
    # Y2 = 1 / (1 + np.exp(-beta * U2)) # używamy funkcji aktywacji softmax dla warstwy wyjściowej
    Y2 = np.exp(U2) / np.sum(np.exp(U2))
    return Y1, Y2


def train2(W1before, W2before, P, T, n):
    noExamples = P.shape[1]
    W1 = W1before.copy()
    W2 = W2before.copy()
    lr = 0.1
    beta = 5
    CE_list = []
    W1_list = []
    W2_list = []
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

        # obliczanie błędu klasyfikacji dla całego zbioru uczącego
        # tworzymy puste listy do przechowywania wyjść sieci dla każdego wektora wejściowego
        Y1_all = []
        Y2_all = []
        # iterujemy po każdym wektorze wejściowym i obliczamy wyjście sieci dla niego
        if n % 50 == 0:
            for j in range(noExamples):
                X = P[:, j]
                Y1, Y2 = sim2(W1, W2, X)
                # dodajemy wyjście sieci do list
                Y1_all.append(Y1)
                Y2_all.append(Y2)
            # konwertujemy listy na tablice numpy
            Y2_all = np.array(Y2_all)
            # zakładamy, że jeśli wyjście sieci jest większe niż 0.5, to sieć przewiduje klasę 1, a jeśli jest mniejsze, to sieć przewiduje klasę 0
            Y2_pred = np.where(Y2_all > 0.5, 1, 0)
            # porównujemy przewidywania sieci z wartościami oczekiwanymi i liczymy liczbę błędów
            CE = 0
            '''
            for i in range(len(T)):
                if Y2_pred[i][0] != T[i]:
                    CE += 1
            '''
            CE = np.sum(np.abs(Y2_pred.flatten() - T))
            # CE = np.sum(np.abs(Y2_pred.T - T))
            # dodajemy błąd do listy
            CE_list.append(CE / len(T) * 100)

        W1 += dW1  # add adjustments to weights in both layers
        W2 += dW2

        W1_list.append(W1[0][0])
        W2_list.append(W2[0][0])

    W1after = W1
    W2after = W2


    return W1after, W2after, CE_list, W1_list, W2_list


def train3(W1before, W2before, P, T, n):
    noExamples = P.shape[1]
    W1 = W1before.copy()
    W2 = W2before.copy()
    lr = 0.1
    beta = 5
    epsilon = 1e-8  # small constant to share
    CE_list = []
    W1_list = []
    W2_list = []

    # Matrix initialization for storing gradient squares
    square_gradients_W1 = np.zeros_like(W1)
    square_gradients_W2 = np.zeros_like(W2)

    for i in range(n):
        exampleNo = np.random.randint(1, noExamples + 1)
        X = P[:, exampleNo - 1]
        X1 = np.insert(X, 0, -1)
        Y1, Y2 = sim2(W1, W2, X)
        X2 = np.insert(Y1, 0, -1)
        D2 = T[exampleNo - 1] - Y2

        E2 = beta * D2 * Y2 * (1 - Y2)
        D1 = np.dot(W2[1:, :], E2)
        E1 = beta * D1 * Y1 * (1 - Y1)

        # Calculation of gradient squares
        square_gradients_W1 += np.outer(X1, E1) ** 2
        square_gradients_W2 += np.outer(X2, E2) ** 2

        # Adaptacyjny współczynnik uczenia
        adjusted_learning_rate_W1 = lr / (np.sqrt(square_gradients_W1) + epsilon)
        adjusted_learning_rate_W2 = lr / (np.sqrt(square_gradients_W2) + epsilon)

        # calculate weight adjustments for layers 1 & 2
        dW1 = adjusted_learning_rate_W1 * np.outer(X1, E1)
        dW2 = adjusted_learning_rate_W2 * np.outer(X2, E2)

        # obliczanie błędu klasyfikacji dla całego zbioru uczącego
        if i % 10 == 0:
            Y2_all = np.array([sim2(W1, W2, P[:, j])[1] for j in range(noExamples)])
            Y2_pred = np.eye(3)[np.argmax(Y2_all, axis=1)]
            CE = np.sum(Y2_pred != T)
            CE_list.append(CE / (len(T) * len(T[0])) * 100)

        W1_list.append(W1.copy())
        W2_list.append(W2.copy())

        W1 += dW1
        W2 += dW2

    W1after = W1
    W2after = W2

    return W1after, W2after, CE_list, W1_list, W2_list

# TODO Połączyć wszystkie plotujące train2 w jedno
# TODO Zrobić fajne wykresy dla różnych stylów trenowania - to co ty roibłeś zaimplementować tutaj, ale to na sam koniec
def train2_with_CE(W1before, W2before, P, T, n):
    noExamples = P.shape[1]
    W1 = W1before.copy()
    W2 = W2before.copy()
    lr = 0.1
    beta = 5
    # lista do przechowywania błędów klasyfikacji
    CE_list = []
    W1_list = []
    W2_list = []

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

        # obliczanie błędu klasyfikacji dla całego zbioru uczącego
        # tworzymy puste listy do przechowywania wyjść sieci dla każdego wektora wejściowego
        Y1_all = []
        Y2_all = []
        # iterujemy po każdym wektorze wejściowym i obliczamy wyjście sieci dla niego

        if n % 50 == 0:
            for j in range(noExamples):
                X = P[:, j]
                Y1, Y2 = sim2(W1, W2, X)
                # dodajemy wyjście sieci do list
                Y1_all.append(Y1)
                Y2_all.append(Y2)
            # konwertujemy listy na tablice numpy
            Y2_all = np.array(Y2_all)
            # zakładamy, że jeśli wyjście sieci jest większe niż 0.5, to sieć przewiduje klasę 1, a jeśli jest mniejsze, to sieć przewiduje klasę 0
            Y2_pred = np.eye(3)[np.argmax(Y2_all, axis=1)]
            # porównujemy przewidywania sieci z wartościami oczekiwanymi i liczymy liczbę błędów
            CE = 0

            for i in range(len(T)):
                for j in range(len(T[0])):
                    if Y2_pred[i][j] != T[i][j]:
                        CE += 1

            # CE = np.sum(np.abs(Y2_pred - T))
            # CE = np.sum(np.abs(Y2_pred.T - T))
            # dodajemy błąd do listy
            CE_list.append( CE/(len(T) * len(T[0])) * 100)

        W1 += dW1  # add adjustments to weights in both layers
        W2 += dW2

        W1_list.append(W1[0][0])
        W2_list.append(W2[0][0])



    W1after = W1
    W2after = W2

    return W1after, W2after, CE_list, W1_list, W2_list



def train2_with_plots(W1before, W2before, P, T, n):
    noExamples = P.shape[1]
    W1 = W1before.copy()
    W2 = W2before.copy()
    lr = 0.1
    beta = 5

    # listy do przechowywania błędów MSE
    MSE1_list = []
    MSE2_list = []
    MSE1_total = []
    MSE2_total = []
    Y1_all, Y2_all = [], []

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

        # obliczanie błędów MSE dla obu warstw
        MSE1 = np.mean((D1 - Y1)**2)
        MSE2 = np.mean((D2 - Y2)**2)
        # dodawanie błędów do list
        MSE1_list.append(MSE1)
        MSE2_list.append(MSE2)
        '''
        # obliczanie błędów MSE dla całego zbioru uczącego
        for column in range(P.shape[1]):
            Y1, Y2 = sim2(W1, W2, P[:, column])
            Y1_all.append(Y1)
            Y2_all.append(Y2)

        MSE2_all = np.mean((T - Y2_all)**2)
        # dodawanie błędów do list
        MSE2_total.append(MSE2_all)
        '''
    W1after = W1
    W2after = W2

    return W1after, W2after, MSE1_list, MSE2_list, MSE1_total, MSE2_total


def sim(X, W):
    # calculates output of a multi-layer net for a given input
    # parameters: X – input vector to the net / layer 1
    # *W – list of weight matrices for each layer
    # result: Y – output vector for the final layer / the net
    beta = 5
    input_vector = X.copy()
    # input_vector = np.insert(X, 0, -1)
    output_vector_list = []
    for i, weights in enumerate(W):
        input_vector = np.insert(input_vector, 0, -1)
        net_input = np.dot(weights.T, input_vector)
        output_vector = 1 / (1 + np.exp(-beta * net_input))
        input_vector = output_vector
        output_vector_list.append(output_vector)
    return output_vector_list


def train(W, P, T, n):
    # trains a multi-layer net using backpropagation
    # parameters: W – list of weight matrices for each layer
    # P – matrix of input vectors (one column per example)
    # T – matrix of target vectors (one column per example)
    # n – number of iterations
    # result: Wafter – list of updated weight matrices for each layer

    noExamples = P.shape[1] # number of examples
    noLayers = len(W) # number of layers
    Wafter = W.copy() # copy the initial weights
    lr = 0.1 # learning rate
    beta = 5 # sigmoid parameter
    Y = []
    X = []
    for i in range(n):
        exampleNo = np.random.randint(1, noExamples + 1) # draw a random example number
        D = [None] * noLayers
        E = [None] * noLayers
        dW = [None] * noLayers
        X1 = P[:, exampleNo - 1] # present the chosen example and calculate output
        # X1 = np.insert(X1, 0, -1)
        print("X1: ", X1.shape)

        print("Wafter: ", len(Wafter[0]))

        X.append(X1)
        Y.append(sim(X1, Wafter))  # list of output vectors for each layer
        for j in range(1, noLayers): # Nie potrzebna - trzeba zrobić inaczej
            Y.append(sim(Y[noLayers - j - 1][0], Wafter)) # list of output vectors for each layer
            X2 = np.insert(Y[noLayers - j][0], 0, -1)
            X.append(X2)
        D2 = T[exampleNo - 1] - Y[-1]
        D[-1] = D2
        E2 = beta * D[-1] * Y[-1] * (1 - Y[-1])
        for j in range(1, noLayers):
            D[noLayers - j] = np.dot(Wafter[noLayers - j][1:, :], E2)

            E2 = beta * D[noLayers - j] * Y[noLayers - j] * (1 - Y[noLayers - j])
            E[noLayers - j] = E2

        for j in range(1, noLayers):

            dW1 = lr * np.outer(X[j], E[j])  # calculate weight adjustments for layers 1 & 2
            dW[j] = dW1
            Wafter[j] += dW[j]
        return Wafter
