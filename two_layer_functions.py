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
    Y2 = np.exp(U2) / np.sum(np.exp(U2))
    return Y1, Y2


def train4(W1before, W2before, P, T, num_epochs):
    batch_size=16
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

    for epoch in range(num_epochs):
        # Variable initialization for counting incorrect predictions
        incorrect_predictions = 0
        num = 0
        for i in range(0, noExamples, batch_size):
            mini_batch_X = P[:, i:i+batch_size]
            mini_batch_T = T[i:i+batch_size]

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
                num += 1
                #print("T: ", T[j])
                Y2_pred = np.argmax(Y2)
                true_class = np.argmax(mini_batch_T[j])
                #print("Y2: ", Y2_pred)
                if Y2_pred != true_class:
                    incorrect_predictions += 1

        # Calculate and append the classification error for the current epoch
        classification_error = incorrect_predictions / ( num ) * 100
        classification_errors.append(classification_error)

    W1after = W1
    W2after = W2

    return W1after, W2after, classification_errors


def mse2(W1, W2, P, T):
    # W1, W2: weights of the neural network
    # P: input matrix
    # T: target matrix
    # returns: the mean squared error between the network output and the target

    # initialize the mse as 0
    mse = 0

    # loop over the input and target pairs
    for i in range(P.shape[1]):
        X = P[:, i] # get the input vector
        Y2 = sim2(W1, W2, X)[1] # get the output vector using the sim2 function
        D2 = T[i] - Y2 # get the difference vector
        mse += np.dot(D2, D2) # add the squared norm of the difference to the mse

    # divide the mse by the number of input and target pairs
    mse /= P.shape[1]

    # return the mse
    return mse


def train4_with_stop(W1before, W2before, P, T, num_epochs, desired_error):
    batch_size=16
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
    mse_total_end_list = []
    # early stopping: split the data into train and validation sets, e.g. 80:20

    split = int(noExamples * 0.8)
    P_train = P[:, :split]
    P_val = P[:, split:]
    T_train = T[:split]
    T_val = T[split:]


    for epoch in range(num_epochs):
        # Variable initialization for counting incorrect predictions
        incorrect_predictions = 0
        num = 0

        for i in range(0, split, batch_size): # early stopping: use only the train set for updating weights
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
                num += 1
                #print("T: ", T[j])
                Y2_pred = np.argmax(Y2)
                true_class = np.argmax(mini_batch_T[j])
                #print("Y2: ", Y2_pred)
                if Y2_pred != true_class:
                    incorrect_predictions += 1

            # MAKING PLOTS - WITH TESTING DATA
            Y_all = []
            mse_total_end = 0
            mse_total_temp = []
            mse_layer1 = 0
            mse_layer2 = 0

            for i in range(split, P.shape[1], batch_size):
                X = P[:, i]
                Y1, Y2 = sim2(W1, W2, X)
                E2 = beta * (T[i] - Y2) * Y2 * (1 - Y2)

                D1 = np.dot(W2[1:, :], E2)
                E1 = beta * D1 * Y1 * (1 - Y1)
                mse_total_temp.append(np.mean(np.power(E2, 2)))
                mse_layer1 = (np.mean(np.power(E1, 2)))
                mse_layer2 = (np.mean(np.power(E2, 2)))
                Y_all.append(Y2)
            mse_layer1_list.append(mse_layer1)
            mse_layer2_list.append(mse_layer2)
            for i in range(split, split + len(Y_all)):
                for j in range(len(T_val[0])):
                    mse_total_end += (Y_all[i - split][j] - T[i][j]) ** 2
            mse_total_errors.append(np.mean(mse_total_temp))
            mse_total_end /= (len(T_val) * len(T_val[0]))
            mse_total_end_list.append(mse_total_end)  # To i mse_total_errors chyba licza to samo więc nw można spróbować 1
            # mse_layer2 = np.mean((T - Y_all) ** 2)
            # Check if MSE is below the desired error



        # Calculate and append the classification error for the current epoch
        classification_error = incorrect_predictions / ( num ) * 100
        #print("CE: ", classification_error)
        #print(incorrect_predictions, "/", num)
        classification_errors.append(classification_error)
        # Calculate MSE for layer 2
        #if len(mse_total_errors) > 50:
        print(np.mean(mse_total_errors[-100:]))
        if np.mean(mse_total_errors[-100:]) < desired_error:
            print(f"Training stopped after {epoch + 1} epochs. Desired error reached.")
            print(f"DŁUGOŚĆ List mse: {len(mse_total_errors)}")
            break


    W1after = W1
    W2after = W2

    return W1after, W2after, classification_errors, mse_total_errors, mse_layer1_list, mse_layer2_list
