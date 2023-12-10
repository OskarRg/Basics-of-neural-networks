import random

import numpy as np


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
    Y1 = 1 / (1 + np.exp(-beta * U1)) # używamy funkcji aktywacji ReLU dla warstwy ukrytej
    # Y1 = np.maximum(0, U1)
    X2 = np.insert(Y1, 0, -1)
    U2 = np.dot(W2.T, X2)
    # Y2 = 1 / (1 + np.exp(-beta * U2)) # używamy funkcji aktywacji softmax dla warstwy wyjściowej
    # Y2 = np.exp(U2) / np.sum(np.exp(U2))
    Y2 = 1 / (1 + np.exp(-beta * U2))
    return Y1, Y2


def train4(W1before, W2before, P, T, num_epochs):
    batch_size = 16
    noExamples = P.shape[1]
    W1 = W1before.copy()
    W2 = W2before.copy()
    lr = 0.1
    beta = 5
    epsilon = 1e-8  # small constant to share
    # Matrix initialization for storing gradient squares
    square_gradients_W1 = np.zeros_like(W1)
    square_gradients_W2 = np.zeros_like(W2)

    # List initialization for storing classification errors
    classification_errors = []
    mse_total_errors = []
    mse_layer1_list = []
    mse_layer2_list = []
    W1_values = [W1before.copy()]
    W2_values = [W2before.copy()]
    # early stopping: split the data into train and validation sets, e.g. 80:20
    split = int(noExamples * 0.8)
    P_train = P[:, :split]
    T_train = T[:split]

    for epoch in range(num_epochs):
        # Variable initialization for counting incorrect predictions
        incorrect_predictions = 0
        num = 0

        for i in range(0, split, batch_size):  # early stopping: use only the train set for updating weights
            mini_batch_X = P_train[:, i:i + batch_size]
            mini_batch_T = T_train[i:i + batch_size]
            for j in range(mini_batch_X.shape[1]):
                X = mini_batch_X[:, j]
                X1 = np.insert(X, 0, -1)
                Y1, Y2 = sim2(W1, W2, X)
                X2 = np.insert(Y1, 0, -1)
                D2 = mini_batch_T[j] - Y2
                # Increment incorrect predictions if the output is not equal to the target
                E2 = beta * D2 * Y2 * (1 - Y2)  # "internal" error for layer 2
                D1 = np.dot(W2[1:, :], E2)  # calculate errors for layer 1 (backpropagation)
                E1 = beta * D1 * Y1 * (1 - Y1)  # "internal" error for layer 1
                # Calculation of gradient squares
                square_gradients_W1 += np.outer(X1, E1) ** 2
                square_gradients_W2 += np.outer(X2, E2) ** 2
                # Adaptacyjny współczynnik uczenia
                adjusted_learning_rate_W1 = lr / (np.sqrt(square_gradients_W1) + epsilon)
                adjusted_learning_rate_W2 = lr / (np.sqrt(square_gradients_W2) + epsilon)
                # calculate weight adjustments for layers 1 & 2
                dW1 = adjusted_learning_rate_W1 * np.outer(X1, E1)
                dW2 = adjusted_learning_rate_W2 * np.outer(X2, E2)
                # add adjustments to weights in both layers
                W1 += dW1
                W2 += dW2
                W1_values.append(W1.copy())
                W2_values.append(W2.copy())
            # -------- MAKING PLOTS - WITH TESTING DATA --------
            Y_all = []
            mse_total_temp = []
            mse_layer1 = 0
            mse_layer2 = 0

            for i in range(split, P.shape[1], batch_size):
                X = P[:, i]
                Y1, Y2 = sim2(W1, W2, X)

                D2 = (T[i] - Y2)
                E2 = beta * D2 * Y2 * (1 - Y2)
                D1 = np.dot(W2[1:, :], E2)
                E1 = beta * D1 * Y1 * (1 - Y1)
                mse_total_temp.append(np.mean(np.power(D2, 2)))
                mse_layer1 = (np.mean(np.power(D1, 2)))
                mse_layer2 = (np.mean(np.power(D2, 2)))
                Y_all.append(Y2)
                num += 1
                Y2_pred = np.argmax(Y2)
                true_class = np.argmax(T[i])
                if Y2_pred != true_class:
                    incorrect_predictions += 1

            mse_layer1_list.append(mse_layer1)
            mse_layer2_list.append(mse_layer2)
            mse_total_errors.append(np.mean(mse_total_temp))

        # Calculate and append the classification error for the current epoch
        classification_error = incorrect_predictions / (num) * 100
        classification_errors.append(classification_error)

    W1after = W1
    W2after = W2

    return W1after, W2after, classification_errors, mse_total_errors, mse_layer1_list, mse_layer2_list, W1_values, W2_values


def train4_with_stop(W1before, W2before, P, T, num_epochs, desired_error):
    batch_size=16
    noExamples = P.shape[1]
    W1 = W1before.copy()
    W2 = W2before.copy()
    lr = 0.1
    beta = 1
    epsilon = 1e-8  # small constant to share
    # Matrix initialization for storing gradient squares
    square_gradients_W1 = np.zeros_like(W1)
    square_gradients_W2 = np.zeros_like(W2)

    # List initialization for storing classification errors
    classification_errors = []
    mse_total_errors = []
    mse_layer1_list = []
    mse_layer2_list = []
    W1_values = [W1before.copy()]
    W2_values = [W2before.copy()]
    # early stopping: split the data into train and validation sets, e.g. 80:20
    split = int(noExamples * 0.7)
    P_train = P[:, :split]
    T_train = T[:split]

    for epoch in range(num_epochs):
        # Variable initialization for counting incorrect predictions

        for i in range(0, split, batch_size): # early stopping: use only the train set for updating weights
            incorrect_predictions = 0
            num = 0
            mini_batch_X = P_train[:, i:i+batch_size]
            mini_batch_T = T_train[i:i+batch_size]
            for j in range(mini_batch_X.shape[1]):
                X = mini_batch_X[:, j]
                X1 = np.insert(X, 0, -1)
                Y1, Y2 = sim2(W1, W2, X)
                X2 = np.insert(Y1, 0, -1)
                D2 = mini_batch_T[j] - Y2
                # Increment incorrect predictions if the output is not equal to the target
                E2 = beta * D2 * Y2 * (1 - Y2)  # "internal" error for layer 2
                D1 = np.dot(W2[1:, :], E2)  # calculate errors for layer 1 (backpropagation)
                E1 = beta * D1 * Y1 * (1 - Y1)  # "internal" error for layer 1
                # Calculation of gradient squares
                square_gradients_W1 += np.outer(X1, E1)**2
                square_gradients_W2 += np.outer(X2, E2)**2
                # Adaptacyjny współczynnik uczenia
                adjusted_learning_rate_W1 = lr / (np.sqrt(square_gradients_W1) + epsilon)
                adjusted_learning_rate_W2 = lr / (np.sqrt(square_gradients_W2) + epsilon)
                # calculate weight adjustments for layers 1 & 2
                dW1 = adjusted_learning_rate_W1 * np.outer(X1, E1)
                dW2 = adjusted_learning_rate_W2 * np.outer(X2, E2)
                # add adjustments to weights in both layers
                W1 += dW1
                W2 += dW2
            W1_values.append(W1.copy())
            W2_values.append(W2.copy())
            # -------- MAKING PLOTS - WITH TESTING DATA --------
            Y_all = []
            mse_total_temp = []
            mse_layer1_rand = []
            mse_layer2_rand = []
            mse_layer1 = 0
            mse_layer2 = 0

            for p in range(split, P.shape[1]):

                X = P[:, p]
                Y1, Y2 = sim2(W1, W2, X)

                D2 = (T[p] - Y2)
                E2 = beta * D2 * Y2 * (1 - Y2)
                D1 = np.dot(W2[1:, :], E2)
                E1 = beta * D1 * Y1 * (1 - Y1)
                mse_total_temp.append(np.mean(np.power(D2, 2)))
                mse_layer1 = (np.mean(np.power(D1, 2)))
                mse_layer2 = (np.mean(np.power(D2, 2)))
                mse_layer1_rand.append(mse_layer1)
                mse_layer2_rand.append(mse_layer2)  # useless we can use mse_total_temp
                Y_all.append(Y2)
                num += 1
                Y2_pred = np.argmax(Y2)
                true_class = np.argmax(T[p])
                if Y2_pred != true_class:
                    incorrect_predictions += 1
            print(num)
            mse_layer1_list.append(mse_layer1_rand[random.randint(0, len(mse_layer1_rand)-1)])
            mse_layer2_list.append(mse_layer2_rand[random.randint(0, len(mse_layer2_rand)-1)])
            # mse_layer2_list.append(mse_layer2)
            mse_total_errors.append(np.mean(mse_total_temp))

            # Calculate and append the classification error for the current epoch
            classification_error = incorrect_predictions / ( num ) * 100
            classification_errors.append(classification_error)

        if np.mean(mse_total_errors[-10:]) < desired_error:
            print(f"Training stopped after {epoch + 1} epochs. Desired error reached.")
            break

    W1after = W1
    W2after = W2

    return W1after, W2after, classification_errors, mse_total_errors, mse_layer1_list, mse_layer2_list, W1_values, W2_values, epoch


def train_with_stop_without_adaptive(W1before, W2before, P, T, num_epochs, desired_error):
    batch_size=16
    noExamples = P.shape[1]
    W1 = W1before.copy()
    W2 = W2before.copy()
    lr = 0.1
    beta = 5
    # Matrix initialization for storing gradient squares
    # List initialization for storing classification errors
    classification_errors = []
    mse_total_errors = []
    mse_layer1_list = []
    mse_layer2_list = []
    W1_values = [W1before.copy()]
    W2_values = [W2before.copy()]
    # early stopping: split the data into train and validation sets, e.g. 80:20
    split = int(noExamples * 0.7)
    P_train = P[:, :split]
    T_train = T[:split]

    for epoch in range(num_epochs):
        # Variable initialization for counting incorrect predictions

        for i in range(0, split, batch_size): # early stopping: use only the train set for updating weights
            incorrect_predictions = 0
            num = 0
            mini_batch_X = P_train[:, i:i+batch_size]
            mini_batch_T = T_train[i:i+batch_size]
            for j in range(mini_batch_X.shape[1]):
                X = mini_batch_X[:, j]
                X1 = np.insert(X, 0, -1)
                Y1, Y2 = sim2(W1, W2, X)
                X2 = np.insert(Y1, 0, -1)
                D2 = mini_batch_T[j] - Y2
                # Increment incorrect predictions if the output is not equal to the target
                E2 = beta * D2 * Y2 * (1 - Y2)  # "internal" error for layer 2
                D1 = np.dot(W2[1:, :], E2)  # calculate errors for layer 1 (backpropagation)
                E1 = beta * D1 * Y1 * (1 - Y1)  # "internal" error for layer 1
                # calculate weight adjustments for layers 1 & 2
                dW1 = lr * np.outer(X1, E1)
                dW2 = lr * np.outer(X2, E2)
                # add adjustments to weights in both layers
                W1 += dW1
                W2 += dW2
            W1_values.append(W1.copy())
            W2_values.append(W2.copy())
            # -------- MAKING PLOTS - WITH TESTING DATA --------
            Y_all = []
            mse_total_temp = []
            mse_layer1_rand = []
            mse_layer2_rand = []
            mse_layer1 = 0
            mse_layer2 = 0

            for p in range(split, P.shape[1]):

                X = P[:, p]
                Y1, Y2 = sim2(W1, W2, X)

                D2 = (T[p] - Y2)
                E2 = beta * D2 * Y2 * (1 - Y2)
                D1 = np.dot(W2[1:, :], E2)
                E1 = beta * D1 * Y1 * (1 - Y1)
                mse_total_temp.append(np.mean(np.power(D2, 2)))
                mse_layer1 = (np.mean(np.power(D1, 2)))
                mse_layer2 = (np.mean(np.power(D2, 2)))
                mse_layer1_rand.append(mse_layer1)
                mse_layer2_rand.append(mse_layer2)  # useless we can use mse_total_temp
                Y_all.append(Y2)
                num += 1
                Y2_pred = np.argmax(Y2)
                true_class = np.argmax(T[p])
                if Y2_pred != true_class:
                    incorrect_predictions += 1
            print(num)
            mse_layer1_list.append(mse_layer1_rand[random.randint(0, len(mse_layer1_rand)-1)])
            mse_layer2_list.append(mse_layer2_rand[random.randint(0, len(mse_layer2_rand)-1)])
            # mse_layer2_list.append(mse_layer2)
            mse_total_errors.append(np.mean(mse_total_temp))

            # Calculate and append the classification error for the current epoch
            classification_error = incorrect_predictions / ( num ) * 100
            classification_errors.append(classification_error)

        if np.mean(mse_total_errors[-10:]) < desired_error:
            print(f"Training stopped after {epoch + 1} epochs. Desired error reached.")
            break

    W1after = W1
    W2after = W2

    return W1after, W2after, classification_errors, mse_total_errors, mse_layer1_list, mse_layer2_list, W1_values, W2_values, epoch


# THERE IS NO beta PARAMETER ALE DZIAŁA
def train_with_adaptive_without_momentum(W1before, W2before, P, T, num_epochs, desired_error):
    batch_size = 16
    noExamples = P.shape[1]
    W1 = W1before.copy()
    W2 = W2before.copy()
    lr = 0.1

    epsilon = 1e-8  # small constant to avoid division by zero

    # Matrix initialization for storing gradient squares
    square_gradients_W1 = np.zeros_like(W1)
    square_gradients_W2 = np.zeros_like(W2)

    # List initialization for storing classification errors
    classification_errors = []
    mse_total_errors = []
    mse_layer1_list = []
    mse_layer2_list = []
    W1_values = [W1before.copy()]
    W2_values = [W2before.copy()]

    # early stopping: split the data into train and validation sets, e.g. 80:20
    split = int(noExamples * 0.7)
    P_train = P[:, :split]
    T_train = T[:split]

    for epoch in range(num_epochs):
        # Variable initialization for counting incorrect predictions
        incorrect_predictions = 0
        num = 0

        for i in range(0, split, batch_size):  # early stopping: use only the train set for updating weights
            mini_batch_X = P_train[:, i:i+batch_size]
            mini_batch_T = T_train[i:i+batch_size]

            for j in range(mini_batch_X.shape[1]):
                X = mini_batch_X[:, j]
                X1 = np.insert(X, 0, -1)
                Y1, Y2 = sim2(W1, W2, X)
                X2 = np.insert(Y1, 0, -1)
                D2 = mini_batch_T[j] - Y2

                E2 = D2 * Y2 * (1 - Y2)  # "internal" error for layer 2
                D1 = np.dot(W2[1:, :], E2)  # calculate errors for layer 1 (backpropagation)
                E1 = D1 * Y1 * (1 - Y1)  # "internal" error for layer 1
                E2 = E2.reshape(3, 1)
                X2 = X2.reshape(6, 1)
                # Calculation of gradient squares
                square_gradients_W1 += np.dot(X1, E1.T)**2
                square_gradients_W2 += np.dot(X2, E2.T)**2

                # Adaptacyjny współczynnik uczenia
                adjusted_learning_rate_W1 = lr / (np.sqrt(square_gradients_W1) + epsilon)
                adjusted_learning_rate_W2 = lr / (np.sqrt(square_gradients_W2) + epsilon)

                # calculate weight adjustments for layers 1 & 2

                dW1 = adjusted_learning_rate_W1 * np.dot(X1, E1.T)
                dW2 = adjusted_learning_rate_W2 * np.dot(X2, E2.T)

                # add adjustments to weights in both layers
                W1 += dW1
                W2 += dW2

            W1_values.append(W1.copy())
            W2_values.append(W2.copy())

        # -------- MAKING PLOTS - WITH TESTING DATA --------
        Y_all = []
        mse_total_temp = []
        mse_layer1_rand = []
        mse_layer2_rand = []
        mse_layer1 = 0
        mse_layer2 = 0

        for p in range(split, P.shape[1]):
            X = P[:, p]
            Y1, Y2 = sim2(W1, W2, X)

            D2 = (T[p] - Y2)
            E2 = D2 * Y2 * (1 - Y2)
            D1 = np.dot(W2[1:, :], E2)
            E1 = D1 * Y1 * (1 - Y1)

            mse_total_temp.append(np.mean(np.power(D2, 2)))
            mse_layer1 = (np.mean(np.power(D1, 2)))
            mse_layer2 = (np.mean(np.power(D2, 2)))
            mse_layer1_rand.append(mse_layer1)
            mse_layer2_rand.append(mse_layer2)
            Y_all.append(Y2)
            num += 1
            Y2_pred = np.argmax(Y2)
            true_class = np.argmax(T[p])
            if Y2_pred != true_class:
                incorrect_predictions += 1

            mse_layer1_list.append(mse_layer1_rand[random.randint(0, len(mse_layer1_rand) - 1)])
            mse_layer2_list.append(mse_layer2_rand[random.randint(0, len(mse_layer2_rand) - 1)])
            mse_total_errors.append(np.mean(mse_total_temp))

            # Calculate and append the classification error for the current epoch
            classification_error = incorrect_predictions / num * 100
            classification_errors.append(classification_error)

        if np.mean(mse_total_errors[-10:]) < desired_error:
            print(f"Training stopped after {epoch + 1} epochs. Desired error reached.")
            break

    W1after = W1
    W2after = W2

    return W1after, W2after, classification_errors, mse_total_errors, mse_layer1_list, mse_layer2_list, W1_values, W2_values, epoch


# TO PO ZMIANACH W ADAPTIVE NIE DZIAŁA
def train_with_adaptive_without_momentum_not_working(W1before, W2before, P, T, num_epochs, desired_error):
    batch_size = 16
    noExamples = P.shape[1]
    W1 = W1before.copy()
    W2 = W2before.copy()
    lr = 0.1
    epsilon = 1e-8  # small constant to avoid division by zero

    # List initialization for storing classification errors
    classification_errors = []
    mse_total_errors = []
    mse_layer1_list = []
    mse_layer2_list = []
    W1_values = [W1before.copy()]
    W2_values = [W2before.copy()]
    D2_prev = 0
    dW1pokaz = 0
    dW2pokaz = 0
    D2_check = 0
    D2_squared = 0
    # early stopping: split the data into train and validation sets, e.g. 80:20
    split = int(noExamples * 0.7)
    P_train = P[:, :split]
    T_train = T[:split]

    for epoch in range(num_epochs):
        # Variable initialization for counting incorrect predictions
        incorrect_predictions = 0
        num = 0

        for i in range(0, split, batch_size):  # early stopping: use only the train set for updating weights
            mini_batch_X = P_train[:, i:i + batch_size]
            mini_batch_T = T_train[i:i + batch_size]

            for j in range(mini_batch_X.shape[1]):
                X = mini_batch_X[:, j]
                X1 = np.insert(X, 0, -1)
                Y1, Y2 = sim2(W1, W2, X)
                X2 = np.insert(Y1, 0, -1)
                D2 = mini_batch_T[j] - Y2

                E2 = D2 * Y2 * (1 - Y2)  # "internal" error for layer 2
                D1 = np.dot(W2[1:, :], E2)  # calculate errors for layer 1 (backpropagation)

                E1 = D1 * Y1 * (1 - Y1)  # "internal" error for layer 1

                E2 = E2.reshape(3, 1)
                X2 = X2.reshape(6, 1)
                dW1 = lr * np.outer(X1, E1)
                dW2 = lr * np.outer(X2 * E2)

                dW1pokaz += dW1
                dW2pokaz += dW2
                D2_squared += (np.sum(np.power(D2, 2)) / 2)
                # zastosuj poprawkę do wag sieci; reset zmiennych pomocniczych
            W1 += dW1pokaz / mini_batch_X.shape[1]
            W2 += dW2pokaz / mini_batch_X.shape[1]

            dW1pokaz = 0
            dW2pokaz = 0
            D2_check = D2_squared / mini_batch_X.shape[1]
            D2_squared = 0
            '''
            # update weights using momentum
            W1 += lr * np.dot(X1, E1.T)
            E2 = E2.reshape(3, 1)
            X2 = X2.reshape(6, 1)
            W2 += lr * np.dot(X2, E2.T)
            '''
            # D2_check = (np.mean(np.power(D2, 2)))
            if D2_check > (1.05 * D2_prev) and 0.7 * lr >= 0.15:
                lr = 0.7 * lr
            else:
                lr = 1.05 * lr
            D2_prev = D2_check
            W1_values.append(W1.copy())
            W2_values.append(W2.copy())

        # -------- MAKING PLOTS - WITH TESTING DATA --------
        Y_all = []
        mse_total_temp = []
        mse_layer1_rand = []
        mse_layer2_rand = []
        mse_layer1 = 0
        mse_layer2 = 0

        for p in range(split, P.shape[1]):
            X = P[:, p]
            Y1, Y2 = sim2(W1, W2, X)

            D2 = (T[p] - Y2)
            E2 = D2 * Y2 * (1 - Y2)
            D1 = np.dot(W2[1:, :], E2)
            E1 = D1 * Y1 * (1 - Y1)

            mse_total_temp.append(np.mean(np.power(D2, 2)))
            mse_layer1 = (np.mean(np.power(D1, 2)))
            mse_layer2 = (np.mean(np.power(D2, 2)))
            mse_layer1_rand.append(mse_layer1)
            mse_layer2_rand.append(mse_layer2)
            Y_all.append(Y2)
            num += 1
            Y2_pred = np.argmax(Y2)
            true_class = np.argmax(T[p])
            if Y2_pred != true_class:
                incorrect_predictions += 1

        mse_layer1_list.append(mse_layer1_rand[random.randint(0, len(mse_layer1_rand) - 1)])
        mse_layer2_list.append(mse_layer2_rand[random.randint(0, len(mse_layer2_rand) - 1)])
        mse_total_errors.append(np.mean(mse_total_temp))

        # Calculate and append the classification error for the current epoch
        classification_error = incorrect_predictions / num * 100
        classification_errors.append(classification_error)

        if np.mean(mse_total_errors[-10:]) < desired_error:
            print(f"Training stopped after {epoch + 1} epochs. Desired error reached.")
            break

    W1after = W1
    W2after = W2

    return W1after, W2after, classification_errors, mse_total_errors, mse_layer1_list, mse_layer2_list, W1_values, W2_values, epoch


def train_adaptive_with_momentum(W1before, W2before, P, T, num_epochs, desired_error, momentum_rate=0.7):
    batch_size = 16
    noExamples = P.shape[1]
    W1 = W1before.copy()
    W2 = W2before.copy()
    lr = 0.1
    beta = 5
    wspMomentum = 0.9
    epsilon = 1e-8  # small constant to avoid division by zero

    # List initialization for storing classification errors
    classification_errors = []
    mse_total_errors = []
    mse_layer1_list = []
    mse_layer2_list = []
    W1_values = [W1before.copy()]
    W2_values = [W2before.copy()]
    D2_prev = 0
    dW1 = 0
    dW2 = 0
    dW1_prev = 0
    dW2_prev = 0

    dW1pokaz = 0
    dW2pokaz = 0
    D2_check = 0
    D2_squared = 0
    momentum_off = 0.05
    # early stopping: split the data into train and validation sets, e.g. 80:20
    split = int(noExamples * 0.7)
    P_train = P[:, :split]
    T_train = T[:split]

    for epoch in range(num_epochs):
        # Variable initialization for counting incorrect predictions
        incorrect_predictions = 0
        num = 0

        for i in range(0, split, batch_size):  # early stopping: use only the train set for updating weights
            mini_batch_X = P_train[:, i:i+batch_size]
            mini_batch_T = T_train[i:i+batch_size]

            for j in range(mini_batch_X.shape[1]):
                X = mini_batch_X[:, j]
                X1 = np.insert(X, 0, -1)
                Y1, Y2 = sim2(W1, W2, X)
                X2 = np.insert(Y1, 0, -1)
                D2 = mini_batch_T[j] - Y2

                E2 = D2 * Y2 * (1 - Y2)  # "internal" error for layer 2
                D1 = np.dot(W2[1:, :], E2)  # calculate errors for layer 1 (backpropagation)

                E1 = D1 * Y1 * (1 - Y1)  # "internal" error for layer 1

                E2 = E2.reshape(3, 1)
                X2 = X2.reshape(6, 1)
                dW1 = lr * np.dot(X1, E1.T) + wspMomentum * dW1_prev
                dW2 = lr * np.dot(X2, E2.T) + wspMomentum * dW2_prev
                dW1_prev = dW1
                dW2_prev = dW2
                dW1pokaz += dW1
                dW2pokaz += dW2
                D2_squared += (np.sum(np.power(D2, 2)/2))
                # zastosuj poprawkę do wag sieci; reset zmiennych pomocniczych
            W1 += dW1pokaz / mini_batch_X.shape[1]
            W2 += dW2pokaz / mini_batch_X.shape[1]

            dW1pokaz = 0
            dW2pokaz = 0
            D2_check = D2_squared / mini_batch_X.shape[1]
            D2_squared = 0
            '''
            # update weights using momentum
            W1 += lr * np.dot(X1, E1.T)
            E2 = E2.reshape(3, 1)
            X2 = X2.reshape(6, 1)
            W2 += lr * np.dot(X2, E2.T)
            '''
            # D2_check = (np.mean(np.power(D2, 2)))
            if D2_check > (1.05 * D2_prev) and 0.7 * lr >= 0.15:
                lr = 0.7 * lr
            else:
                lr = 1.05 * lr
            D2_prev = D2_check
            W1_values.append(W1.copy())
            W2_values.append(W2.copy())


        # -------- MAKING PLOTS - WITH TESTING DATA --------
        Y_all = []
        mse_total_temp = []
        mse_layer1_rand = []
        mse_layer2_rand = []
        mse_layer1 = 0
        mse_layer2 = 0

        for p in range(split, P.shape[1]):
            X = P[:, p]
            Y1, Y2 = sim2(W1, W2, X)

            D2 = (T[p] - Y2)
            E2 = beta * D2 * Y2 * (1 - Y2)
            D1 = np.dot(W2[1:, :], E2)
            E1 = beta * D1 * Y1 * (1 - Y1)

            mse_total_temp.append(np.mean(np.power(D2, 2)))
            mse_layer1 = (np.mean(np.power(D1, 2)))
            mse_layer2 = (np.mean(np.power(D2, 2)))
            mse_layer1_rand.append(mse_layer1)
            mse_layer2_rand.append(mse_layer2)
            Y_all.append(Y2)
            num += 1
            Y2_pred = np.argmax(Y2)
            true_class = np.argmax(T[p])
            if Y2_pred != true_class:
                incorrect_predictions += 1

        mse_layer1_list.append(mse_layer1_rand[random.randint(0, len(mse_layer1_rand) - 1)])
        mse_layer2_list.append(mse_layer2_rand[random.randint(0, len(mse_layer2_rand) - 1)])
        mse_total_errors.append(np.mean(mse_total_temp))

        # Calculate and append the classification error for the current epoch
        classification_error = incorrect_predictions / num * 100
        classification_errors.append(classification_error)
        if len(mse_total_errors) > 2:
            if (mse_total_errors[-1] - mse_total_errors[-2]) > momentum_off:
                dW1_prev = 0
                dW2_prev = 0

        if np.mean(mse_total_errors[-10:]) < desired_error:
            print(f"Training stopped after {epoch + 1} epochs. Desired error reached.")
            break

    W1after = W1
    W2after = W2

    return W1after, W2after, classification_errors, mse_total_errors, mse_layer1_list, mse_layer2_list, W1_values, W2_values, epoch
