import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split


# Wczytujemy plik csv do obiektu DataFrame
df = pd.read_csv("Salary.csv", sep=',')

# Usuwamy wiersze z brakującymi danymi
df = df.dropna()
print(df)
y = df["Salary"]
X = df[["Education Level", "Years of Experience"]]

# Podział danych na treningowe i na te co użyjemy do sprawdzenia i testowania w proporcjach 70 15 15
X_train, X_unused, y_train, y_unused = train_test_split(X, y, test_size=0.3, random_state=42)


dick = {}
num = 1
# X.Country = X.Country.map( { "USA": 1, "UK": 2, "Canada": 3, "Australia":4, "China": 5, } )
# X_train.Gender = X_train.Gender.map( { "Male": 1, "Female": 2 } )

y = y_train.values
X = X_train.values
X = X.T

# To jest skalowanie na wart od 0-1 ale nie jest potrzebne
#X = (X - np.min(X)) / (np.max(X) - np.min(X))
#y = (y - np.min(y)) / (np.max(y) - np.min(y))
y = np.where(y >= 55000, 1, 0)
# Sprawdzać będziemy, czy pracownik zarabia więcej niż 55k$
print("Kształt y:", y.shape)
print("Kształt X:", X.shape)
