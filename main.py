import numpy as np
from functions import init2, sim2, train_with_stop_without_adaptive, train_with_adaptive_without_momentum, train_with_adaptive_with_momentum
from data_preparation import y as T, X as P
from plot_functions import plot_everything, plot_mse, plot_weights, plot_CE


if __name__ == '__main__':
    n = 5000
    e = 0.001
    MSE1_list, MSE2_list, MSE2_total = [[], [], []]

    Y_before_list = []
    Y_after_list = []
    # W = init2(4, 5, 3)
    W = init2(P.shape[0], P.shape[0] + 2, len(T[0]))

    for column in range(P.shape[1]):
        Y2 = sim2(W[0], W[1], P[:, column])
        Y_before_list.append(Y2[1])

    W_after = [0, 0]
    W_after[0], W_after[1], CE_error, mse2_total_error, mse_layer1, mse_layer2,\
        W1_values, W2_values, epoch = train_with_adaptive_with_momentum(W[0], W[1], P, T, n, e)

    for column in range(P.shape[1]):
        Y2 = sim2(W_after[0], W_after[1], P[:, column])
        Y_after_list.append(Y2[1])

    plot_CE(CE_error, epoch)
    plot_weights(W1_values, W2_values, epoch)
    plot_mse(mse_layer1, mse_layer2, mse2_total_error, epoch)
    plot_everything(mse_layer1, mse_layer2, mse2_total_error, CE_error, epoch)

    bad_output = 0
    good_output = 0
    dick = {0:"Iris-setosa", 1:"Iris-versicolor", 2:"Iris-virginica"}
    for i in range(P.shape[1]):
        X = P[:, i]
        R = T[i]
        Y1, Y2 = sim2(W_after[0], W_after[1], X)

        predicted_class = np.argmax(Y2)
        true_class = np.argmax(R)

        if predicted_class != true_class:
            bad_output += 1

            print(f"Predicted class: {dick[predicted_class]}, example: {X}")
            print(f"True class: {dick[true_class]}")
        else:
            good_output += 1

    print("Bad: ", bad_output)
    print("Good: ", good_output)
    print("It went good {}% of the time".format(int(good_output / (good_output + bad_output) * 100)))
