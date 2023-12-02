import numpy as np
import matplotlib.pyplot as plt
from functions import sigmoid, init2, sim2, train2_with_plots as train2, train2_with_CE as trainCE
from data_preparation_two_layers import y as T, X as P

# TODO Zrobić wykresy MSE - wag - nw jak o co chodzi jeszcze❌
# TODO Momentum ❌ - jest na prezentacji ewentualnie o tym mowa i w KSIĄŻCE
# TODO Zminny rozmiar kroku uczenia - ✔ (?) - jeszcze nie wiem jak działa i te adaptacyjne też
# TODO Wykres błędu klasyfikacji - ✔ - trzeba połączyć train_with w jedność jak będą wszystkie 3 (tak, żeby pętle się nie łączyły - 1 wystarczy)
# TODO Opcja szybszego kończenia uczenia ❌ - po porstu jeden if mądrze umieszczony(?) sprawdzjący Error - nie wiem jaka wartość errora byłaby dobra
# TODO Opis problemu, danych uczących, Obserwacje.
# TODO Zapytać o:
# ❔Czy błąd klasyfikacji musi zejść do 0 (nam schodzi do maksymalnie 5% nawet przy 20000 obiegach nie ważne ile danych)
# ❔Ile powinnieśmy mieć danych wejściowych(zmiennych) 2, 3 wystarczą(chyba) czy więcej?❔
# ❔❔


def plot_everything(): # zrobimy na ładne kiedyś
    plt.figure()
    plt.plot(MSE1_list, label='MSE na przykładach uczących')
    plt.xlabel('Numer iteracji')
    plt.ylabel('Błąd MSE')
    plt.title('Błąd MSE dla warstwy 1')
    plt.legend()
    plt.show()

    # rysowanie wykresów błędu MSE dla warstwy 2
    plt.figure()
    plt.plot(MSE2_list, label='MSE na przykładach uczących')
    plt.plot(MSE2_total, label='MSE na całym zbiorze uczącym')
    plt.xlabel('Numer iteracji')
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
    # plt.ylim(0, 20)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    n = 5000
    e = 0.05  # error ten do tego wcześniejszego kończenia, nie wiem jaki nam wystarczy
    MSE1_list, MSE2_list, MSE2_total = [[], [], []]
    '''
    # To przykład z otyłością, też działa taki malutki
    P = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                  [.55, .5, .65, .50, .75, .75, 0.9, .70, .72, .68, 0.74, 1.20]])
    T = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1])
    '''
    Y_before_list = []
    Y_after_list = []
    #W1before, W2before = init2(2, 3, 1)
    W = init2(2, 5, 1)  # Duża liczba chyba pozwala uniknąć najgorszych przypadków, ale nie pomaga jakoś bardzo

    print("W1 size =", len(W[0]))
    print("W2 size =", len(W[1]))
    for column in range(P.shape[1]):
        Y2 = sim2(W[0], W[1], P[:, column])
        Y_before_list.append(Y2[1])

    W_after = [0, 0]
    # W_after[0], W_after[1] = train2(W[0], W[1], P, T, n)
    # W_after[0], W_after[1], MSE1_list, MSE2_list, MSE1_total, MSE2_total = train2(W[0], W[1], P, T, n)
    W_after[0], W_after[1], CE_error = trainCE(W[0], W[1], P, T, n)

    for column in range(P.shape[1]):
        Y2 = sim2(W_after[0], W_after[1], P[:, column])
        Y_after_list.append(Y2[1])

    plot_CE()


    ninety = []
    bad_output = 0
    good_output = 0

    for i in range(P.shape[1]):
        X = P[:, i]
        R = T[i]
        Y1, Y2 = sim2(W_after[0], W_after[1], X)

        ninety.append( 1 if Y2 > 0.5 else 0)
        if ninety[i] != R:
            bad_output += 1
            print("Jest źle: ")
            print(f"Input: {X}, Predicted Output: {Y2}, Real Output {R}")
        else:
            good_output += 1

    print("Bad: ", bad_output)
    print("Good: ", good_output)
    print("It went good {}% of the time".format(int(good_output/(good_output+bad_output)*100)))


