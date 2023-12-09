import numpy as np
import matplotlib.pyplot as plt
from two_layer_functions import sigmoid, init2, sim2, train4, train4_with_stop
from functions import train_with_momentum
from data_preparation_two_layers import y as T, X as P
# TODO Zrobić wykresy MSE - wag - już wiem jak - raczej proste ❌ XDDDDDD - NIE BYŁO PROSTE 09.12 01:22
# TODO Momentum ❌ - jest na prezentacji ewentualnie o tym mowa i w KSIĄŻCE
# TODO Zminny rozmiar kroku uczenia - ✔ (?) - jeszcze nie wiem jak działa i te adaptacyjne też
# TODO Wykres błędu klasyfikacji - ✔ - trzeba połączyć train_with w jedność jak będą wszystkie 3 (tak, żeby pętle się nie łączyły - 1 wystarczy)
# TODO Opcja szybszego kończenia uczenia ✔ - po porstu jeden if mądrze umieszczony(?) sprawdzjący Error - nie wiem jaka wartość errora byłaby dobra TEŻ NIE BYŁO PROSTE
# TODO Opis problemu, danych uczących, Obserwacje.
# TODO Zapytać o:
# ❔Czy błąd klasyfikacji musi zejść do 0 (nam schodzi do maksymalnie 5% nawet przy 20000 obiegach nie ważne ile danych)
# ❔Ile powinnieśmy mieć danych wejściowych(zmiennych) 2, 3 wystarczą(chyba) czy więcej?❔
# ❔❔
# ❗Ja się zajmę teraz Wykresami aby ładnie je połączyć w jedność,
# ❗zrobić wogle wszystkie 3 typy bo na razie jest 1, na razie jest syf straszny

def plot_everything(): # zrobimy na ładne kiedyś
    lista_wartosci = np.linspace(0, len(CE_error) - 1, len(mse_layer1), dtype=int)

    plt.figure()
    plt.plot(lista_wartosci, mse_layer1, label='MSE na przykładach uczących')
    plt.xlabel('Numer iteracji')
    plt.ylabel('Błąd MSE')
    plt.title('Błąd MSE dla warstwy 1')

    plt.legend()
    plt.show()

    # rysowanie wykresów błędu MSE dla warstwy 2
    plt.figure()

    plt.plot(lista_wartosci, mse_layer2, label='MSE na przykładach uczących')
    plt.plot(lista_wartosci, mse2_total_error, label='MSE na całym zbiorze uczącym')
    # Określenie kroku na osi X

    plt.xlabel('Epoka')
    plt.ylabel('Błąd MSE')
    plt.title('Błąd MSE dla warstwy 2')
    plt.legend()
    plt.show()


def plot_CE():
    # rysowanie wykresu błędu klasyfikacji
    plt.figure()
    plt.plot(CE_error, label='Błąd klasyfikacji na całym zbiorze uczącym')
    plt.xlabel('Numer iteracji')
    plt.ylabel('Błąd klasyfikacji')
    plt.title('Błąd klasyfikacji na całym ciągu uczącym')
    #plt.ylim(0, 10)
    plt.legend()
    plt.show()


def plot_weights(W1_list, W2_list):
    # Rysowanie wykresów wag dla warstwy 1
    plt.figure()
    W1_values = np.array([W[:, 0] for W in W1_list])
    for i in range(W1_values.shape[1]):
        plt.plot(W1_values[:, i], label=f'Waga dla warstwy 1, neuron {i + 1}')

    plt.xlabel('Numer iteracji')
    plt.ylabel('Wartość wagi')
    plt.title('Wykresy wag dla warstwy 1')
    plt.legend()
    plt.show()

    # Rysowanie wykresów wag dla warstwy 2
    plt.figure()
    W2_values = np.array([W[:, 0] for W in W2_list])
    for i in range(W2_values.shape[1]):
        plt.plot(W2_values[:, i], label=f'Waga dla warstwy 2, neuron {i + 1}')

    plt.xlabel('Numer iteracji')
    plt.ylabel('Wartość wagi')
    plt.title('Wykresy wag dla warstwy 2')
    plt.legend()
    plt.show()


def plot_mse():
    '''
    x_values = range(len(MSE2_error))
    # Rysowanie wykresu błędu MSE dla warstwy ukrytej
    plt.scatter(x_values, MSE1_error, label='MSE1', s=1)
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title('MSE error for hidden layer')

    #plt.ylim(0, 0.1)
    plt.legend()
    plt.show()
    '''
    # Rysowanie wykresu błędu MSE dla warstwy wyjściowej
    plt.plot(mse2_total_error, label='MSE2')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title('MSE error for output layer')
    #plt.ylim(0, 0.1)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    n = 2000
    e = 0.05  # error ten do tego wcześniejszego kończenia, nie wiem jaki nam wystarczy
    MSE1_list, MSE2_list, MSE2_total = [[], [], []]
    '''
    # To przykład z otyłością, też działa taki malutki
    P = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                  [.55, .5, .65, .50, .75, .75, 0.9, .70, .72, .68, 0.74, 1.20]])
    T = np.array([[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
                  [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0]])
    T = T.T              
    '''

    Y_before_list = []
    Y_after_list = []
    #W1before, W2before = init2(2, 3, 1)
    W = init2(4, 12, 3)  # Duża liczba chyba pozwala uniknąć najgorszych przypadków, ale nie pomaga jakoś bardzo

    print("W1 size =", len(W[0]))
    print("W2 size =", len(W[1]))
    for column in range(P.shape[1]):
        Y2 = sim2(W[0], W[1], P[:, column])
        Y_before_list.append(Y2[1])

    W_after = [0, 0]
    # W_after[0], W_after[1] = train2(W[0], W[1], P, T, n)
    # W_after[0], W_after[1], MSE1_list, MSE2_list, MSE1_total, MSE2_total = train2(W[0], W[1], P, T, n)
    W_after[0], W_after[1], CE_error, mse2_total_error, mse_layer1, mse_layer2 = train4_with_stop(W[0], W[1], P, T, n, 0.005)

    for column in range(P.shape[1]):
        Y2 = sim2(W_after[0], W_after[1], P[:, column])
        Y_after_list.append(Y2[1])

    plot_CE()
    #plot_weights(W1_list, W2_list)
    # plot_mse()
    plot_everything()
    ninety = []
    bad_output = 0
    good_output = 0

    for i in range(P.shape[1]):
        X = P[:, i]
        R = T[i]
        Y1, Y2 = sim2(W_after[0], W_after[1], X)

        predicted_class = np.argmax(Y2)  # zakładamy, że klasa to indeks neuronu o najwyższym wyjściu
        true_class = np.argmax(R)

        ninety.append(predicted_class)
        if predicted_class != true_class:
            bad_output += 1
            print("Jest źle: ")
            print(f"Input: {X}, Predicted Output: {predicted_class}, Real Output {true_class}")
        else:
            good_output += 1

    print("Bad: ", bad_output)
    print("Good: ", good_output)
    print("It went good {}% of the time".format(int(good_output / (good_output + bad_output) * 100)))

