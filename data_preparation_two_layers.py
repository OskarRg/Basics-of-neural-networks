import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Wczytujemy plik csv do obiektu DataFrame
df = pd.read_csv("Salary.csv", sep=',')

# Usuwamy wiersze z brakującymi danymi
df = df.dropna()
print(df)
y = df["Salary"]
X = df[["Education Level", "Years of Experience", "Country"]]

dick = {}
num = 1
X.Country = X.Country.map( { "USA": 1, "UK": 2, "Canada": 3, "Australia":4, "China": 5, } )
y = y.values
X = X.values
X = X.T

# To jest skalowanie na wart od 0-1 ale nie jest potrzebne
#X = (X - np.min(X)) / (np.max(X) - np.min(X))
#y = (y - np.min(y)) / (np.max(y) - np.min(y))
y = np.where(y >= 55000, 1, 0)
# Sprawdzać będziemy, czy pracownik zarabia więcej niż 55k$
print("Kształt y:", y.shape)
print("Kształt X:", X.shape)

''' 
# Tym można patrzeć na korelacje, fajne można popatrzeć jak bardzo wyjście zależy od wejścia.
# Obliczamy wartości chi-kwadrat i p-wartości dla każdej kolumny X i y
# Używamy funkcji chi2_contingency z biblioteki scipy, która automatycznie tworzy tabelę kontyngencji i oblicza statystykę testu
# Używamy metody transpose, aby zamienić kolumny na wiersze
chi2s = []
pvals = []
for col in X.transpose():
    chi2, pval, _, _ = chi2_contingency(np.c_[col, y])
    chi2s.append(chi2)
    pvals.append(pval)

# Wyświetlamy wartości chi-kwadrat i p-wartości
print("Wartości chi-kwadrat i p-wartości dla każdej kolumny X i y:")
for i, (chi2, pval) in enumerate(zip(chi2s, pvals)):
    print(f"Kolumna {i}: chi-kwadrat = {chi2:.2f}, p-wartość = {pval:.2f}")'''