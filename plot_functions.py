import matplotlib.pyplot as plt
import numpy as np


def plot_everything(mse_layer1, mse_layer2, mse2_total_error, CE_error, epoch):
    epoch_number = np.linspace(0, epoch, len(mse_layer1), dtype=int)

    # Subplots init
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # First Layer
    axs[0].plot(epoch_number, mse_layer1, label='MSE na przykładach uczących')
    axs[0].set_xlabel('Epoka')
    axs[0].set_ylabel('Błąd MSE')
    axs[0].set_title('Błąd MSE dla warstwy 1')
    axs[0].legend()

    # Second Layer
    axs[1].plot(epoch_number, mse_layer2, label='MSE na przykładach uczących')
    axs[1].plot(epoch_number, mse2_total_error, label='MSE na całym zbiorze uczącym')
    axs[1].set_xlabel('Epoka')
    axs[1].set_ylabel('Błąd MSE')
    axs[1].set_title('Błąd MSE dla warstwy 2')
    axs[1].legend()

    # Classification Error
    axs[2].plot(epoch_number, CE_error, label='Błąd klasyfikacji na całym zbiorze uczącym')
    axs[2].set_xlabel('Epoka')
    axs[2].set_ylabel('Błąd klasyfikacji')
    axs[2].set_title('Błąd klasyfikacji na całym ciągu uczącym')
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def plot_CE(CE_error, epoch):
    epoch_number = np.linspace(0, epoch - 1, len(CE_error), dtype=int)
    plt.figure()
    plt.plot(epoch_number, CE_error, label='Błąd klasyfikacji na całym zbiorze uczącym')
    plt.xlabel('Epoka')
    plt.ylabel('Błąd klasyfikacji')
    plt.title('Błąd klasyfikacji na całym ciągu uczącym')
    plt.legend()
    plt.show()


def plot_weights(W1_list, W2_list, epoch):
    epoch_number = np.linspace(0, epoch - 1, len(W1_list), dtype=int)
    plt.figure()
    W1_values_list = np.array([W[:, 0] for W in W1_list])
    for i in range(W1_values_list.shape[1]):
        plt.plot(epoch_number, W1_values_list[:, i], label=f'Waga dla warstwy 1, neuron {i + 1}')

    plt.xlabel('Epoka')
    plt.ylabel('Wartość wagi')
    plt.title('Wykresy wag dla warstwy 1')
    plt.legend()
    plt.show()

    # Second Layer
    plt.figure()
    W2_values_list = np.array([W[:, 0] for W in W2_list])
    for i in range(W2_values_list.shape[1]):
        plt.plot(epoch_number, W2_values_list[:, i], label=f'Waga dla warstwy 2, neuron {i + 1}')

    plt.xlabel('Epoka')
    plt.ylabel('Wartość wagi')
    plt.title('Wykresy wag dla warstwy 2')
    plt.legend()
    plt.show()


def plot_mse(mse_layer1, mse_layer2, mse2_total_error, epoch):
    epoch_number = np.linspace(0, epoch - 1, len(mse_layer1), dtype=int)

    plt.figure()
    plt.plot(epoch_number, mse_layer1, label='MSE na przykładach uczących')
    plt.xlabel('Epoka')
    plt.ylabel('Błąd MSE')
    plt.title('Błąd MSE dla warstwy 1')
    plt.legend()
    plt.show()

    # Second Layer
    plt.figure()
    plt.plot(epoch_number, mse_layer2, label='MSE na przykładach uczących')
    plt.plot(epoch_number, mse2_total_error, label='MSE na całym zbiorze uczącym')
    plt.xlabel('Epoka')
    plt.ylabel('Błąd MSE')
    plt.title('Błąd MSE dla warstwy 2')
    plt.legend()
    plt.show()