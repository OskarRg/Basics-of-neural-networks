import random
import numpy as np


def sigmoid(x, beta=1):
    return 1 / (1 + np.exp(-beta * x))


def init2(S, K1, K2):
    W1 = np.random.uniform(-0.1, 0.1, (S + 1, K1))
    W2 = np.random.uniform(-0.1, 0.1, (K1 + 1, K2))
    return W1, W2


def sim2(W1, W2, X):
    beta = 5
    X1 = np.insert(X, 0, -1)
    U1 = np.dot(W1.T, X1)
    Y1 = sigmoid(U1, beta)
    X2 = np.insert(Y1, 0, -1)
    U2 = np.dot(W2.T, X2)
    Y2 = sigmoid(U2, beta)
    return Y1, Y2


def train_with_stop_without_adaptive(W1before, W2before, P, T, num_epochs, desired_error, batch_size=16):
    noExamples = P.shape[1]
    W1 = W1before.copy()
    W2 = W2before.copy()
    lr = 0.1
    beta = 5
    dW1_sum = 0
    dW2_sum = 0
    classification_errors = []
    mse_total_errors = []
    mse_layer1_list = []
    mse_layer2_list = []
    W1_values = [W1before.copy()]
    W2_values = [W2before.copy()]
    split = int(noExamples * 0.7)
    P_train = P[:, :split]
    T_train = T[:split]

    for epoch in range(num_epochs):
        for i in range(0, split, batch_size):
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

                E2 = beta * D2 * Y2 * (1 - Y2)
                D1 = np.dot(W2[1:, :], E2)
                E1 = beta * D1 * Y1 * (1 - Y1)
                dW1 = lr * np.outer(X1, E1)
                dW2 = lr * np.outer(X2, E2)
                dW1_sum += dW1
                dW2_sum += dW2

            W1 += dW1_sum / mini_batch_X.shape[1]
            W2 += dW2_sum / mini_batch_X.shape[1]

            dW1_sum = 0
            dW2_sum = 0
            W1_values.append(W1.copy())
            W2_values.append(W2.copy())

            # -------- MAKING PLOTS - WITH TESTING DATA --------
            Y_all = []
            mse_total_temp = []
            mse_layer1_rand = []
            mse_layer2_rand = []

            for p in range(split, P.shape[1]):

                X = P[:, p]
                Y1, Y2 = sim2(W1, W2, X)
                D2 = (T[p] - Y2)
                E2 = beta * D2 * Y2 * (1 - Y2)
                D1 = np.dot(W2[1:, :], E2)
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

            mse_layer1_list.append(mse_layer1_rand[random.randint(0, len(mse_layer1_rand)-1)])
            mse_layer2_list.append(mse_layer2_rand[random.randint(0, len(mse_layer2_rand)-1)])
            mse_total_errors.append(np.mean(mse_total_temp))

            classification_error = incorrect_predictions / num * 100
            classification_errors.append(classification_error)

        if np.mean(mse_total_errors[-50:]) < desired_error:
            print(f"Training stopped after {epoch + 1} epochs. Desired error reached.")
            break

    W1after = W1
    W2after = W2

    return W1after, W2after, classification_errors, mse_total_errors, mse_layer1_list, mse_layer2_list, W1_values, W2_values, epoch


def train_with_adaptive_without_momentum(W1before, W2before, P, T, num_epochs, desired_error, batch_size=16):
    noExamples = P.shape[1]
    W1 = W1before.copy()
    W2 = W2before.copy()
    lr = 0.1
    beta = 2
    classification_errors = []
    mse_total_errors = []
    mse_layer1_list = []
    mse_layer2_list = []
    W1_values = [W1before.copy()]
    W2_values = [W2before.copy()]
    D2_prev = 0
    dW1_sum = 0
    dW2_sum = 0
    D2_squared_sum = 0
    split = int(noExamples * 0.7)
    P_train = P[:, :split]
    T_train = T[:split]

    for epoch in range(num_epochs):
        incorrect_predictions = 0
        num = 0

        for i in range(0, split, batch_size):
            mini_batch_X = P_train[:, i:i + batch_size]
            mini_batch_T = T_train[i:i + batch_size]

            for j in range(mini_batch_X.shape[1]):
                X = mini_batch_X[:, j]
                X1 = np.insert(X, 0, -1)
                Y1, Y2 = sim2(W1, W2, X)
                X2 = np.insert(Y1, 0, -1)
                D2 = mini_batch_T[j] - Y2
                E2 = beta * D2 * Y2 * (1 - Y2)
                D1 = np.dot(W2[1:, :], E2)
                E1 = beta * D1 * Y1 * (1 - Y1)
                dW1 = lr * np.outer(X1, E1)
                dW2 = lr * np.outer(X2, E2)
                dW1_sum += dW1
                dW2_sum += dW2
                D2_squared_sum += (np.sum(np.power(D2, 2)) / 2)
            W1 += dW1_sum / mini_batch_X.shape[1]
            W2 += dW2_sum / mini_batch_X.shape[1]

            dW1_sum = 0
            dW2_sum = 0
            D2_check = D2_squared_sum / mini_batch_X.shape[1]
            D2_squared_sum = 0


            W1_values.append(W1.copy())
            W2_values.append(W2.copy())

        if D2_check > (1.04 * D2_prev) and 0.7 * lr >= 0.15:
            lr = 0.7 * lr
        else:
            lr = 1.05 * lr
        D2_prev = D2_check
        # -------- MAKING PLOTS - WITH TESTING DATA --------
        Y_all = []
        mse_total_temp = []
        mse_layer1_rand = []
        mse_layer2_rand = []

        for p in range(split, P.shape[1]):
            X = P[:, p]
            Y1, Y2 = sim2(W1, W2, X)

            D2 = (T[p] - Y2)
            E2 = beta * D2 * Y2 * (1 - Y2)
            D1 = np.dot(W2[1:, :], E2)

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

        classification_error = incorrect_predictions / num * 100
        classification_errors.append(classification_error)

        if np.mean(mse_total_errors[-50:]) < desired_error:
            print(f"Training stopped after {epoch + 1} epochs. Desired error reached.")
            break

    W1after = W1
    W2after = W2

    return W1after, W2after, classification_errors, mse_total_errors, mse_layer1_list, mse_layer2_list, W1_values, W2_values, epoch


def train_with_adaptive_with_momentum(W1before, W2before, P, T, num_epochs, desired_error, batch_size=16):
    noExamples = P.shape[1]
    W1 = W1before.copy()
    W2 = W2before.copy()
    lr = 0.1
    beta = 1
    momentum = 0.9
    momentum_off = 0.01
    classification_errors = []
    mse_total_errors = []
    mse_layer1_list = []
    mse_layer2_list = []
    W1_values = [W1before.copy()]
    W2_values = [W2before.copy()]
    D2_prev = 0
    dW1_sum = 0
    dW2_sum = 0
    D2_squared = 0
    dW1_prev, dW2_prev = 0, 0

    split = int(noExamples * 0.7)
    P_train = P[:, :split]
    T_train = T[:split]

    for epoch in range(num_epochs):
        incorrect_predictions = 0
        num = 0

        for i in range(0, split, batch_size):
            mini_batch_X = P_train[:, i:i + batch_size]
            mini_batch_T = T_train[i:i + batch_size]

            for j in range(mini_batch_X.shape[1]):

                X = mini_batch_X[:, j]
                X1 = np.insert(X, 0, -1)
                Y1, Y2 = sim2(W1, W2, X)
                X2 = np.insert(Y1, 0, -1)
                D2 = mini_batch_T[j] - Y2
                E2 = beta * D2 * Y2 * (1 - Y2)
                D1 = np.dot(W2[1:, :], E2)
                E1 = beta * D1 * Y1 * (1 - Y1)
                dW1 = lr * np.outer(X1, E1) + momentum * dW1_prev
                dW2 = lr * np.outer(X2, E2) + momentum * dW2_prev
                dW1_prev = dW1
                dW2_prev = dW2
                dW1_sum += dW1
                dW2_sum += dW2
                D2_squared += (np.sum(np.power(D2, 2)) / 2)
            W1 += dW1_sum / mini_batch_X.shape[1]
            W2 += dW2_sum / mini_batch_X.shape[1]
            dW1_sum = 0
            dW2_sum = 0
            D2_check = D2_squared / mini_batch_X.shape[1]
            D2_squared = 0


            W1_values.append(W1.copy())
            W2_values.append(W2.copy())
        if D2_check > (1.04 * D2_prev) and 0.7 * lr >= 0.15:
            lr = 0.7 * lr
        else:
            lr = 1.05 * lr
        D2_prev = D2_check
        momentum = 1 - lr
        # -------- MAKING PLOTS - WITH TESTING DATA --------
        Y_all = []
        mse_total_temp = []
        mse_layer1_rand = []
        mse_layer2_rand = []

        for p in range(split, P.shape[1]):
            X = P[:, p]
            Y1, Y2 = sim2(W1, W2, X)

            D2 = (T[p] - Y2)
            E2 = D2 * Y2 * (1 - Y2)
            D1 = np.dot(W2[1:, :], E2)

            mse_total_temp.append(np.mean(np.power(D2, 2))/2)
            mse_layer1 = (np.mean(np.power(D1, 2)))/2
            mse_layer2 = (np.mean(np.power(D2, 2)))/2
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

        classification_error = incorrect_predictions / num * 100
        classification_errors.append(classification_error)
        if len(mse_total_errors) > 2:
            if (mse_total_errors[-1] - mse_total_errors[-2]) > momentum_off:
                dW1_prev = 0
                dW2_prev = 0

        if np.mean(mse_total_errors[-50:]) < desired_error:
            print(f"Training stopped after {epoch + 1} epochs. Desired error reached.")
            break

    W1after = W1
    W2after = W2

    return W1after, W2after, classification_errors, mse_total_errors, mse_layer1_list, mse_layer2_list, W1_values, W2_values, epoch
