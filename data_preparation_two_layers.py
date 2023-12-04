import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
# Wczytujemy plik csv do obiektu DataFrame
'''
df = pd.read_csv("Salary.csv", sep=',')

# Usuwamy wiersze z brakującymi danymi
df = df.dropna()
print(df)
y = df["Salary"]
X = df[["Education Level", "Years of Experience", "Age"]]

# Podział danych na treningowe i na te co użyjemy do sprawdzenia i testowania w proporcjach 70 15 15
X_train, X_unused, y_train, y_unused = train_test_split(X, y, test_size=0.3, random_state=42)


dick = {}
num = 1
# X_train.Country = X_train.Country.map( { "USA": 1, "UK": 2, "Canada": 3, "Australia":4, "China": 5, } )
# X_train.Gender = X_train.Gender.map( { "Male": 1, "Female": 0 } )

y = y_train.values
X = X_train.values
X = X.T
y = np.where(y >= 55000, 1, 0)

# Sprawdzać będziemy, czy pracownik zarabia więcej niż 55k$
print("Kształt y:", y.shape)
print("Kształt X:", X.shape)
'''
# ----------------------------------- IRYS DATA ----------------------------------------------

df = pd.read_csv("iris.data", sep=',')
df = df.dropna()

y = df["classi"]

X = df[["sepal length", "sepal width", "petal length", "petal width"]]

# Podział danych na treningowe i na te, co użyjemy do sprawdzenia i testowania w proporcjach 70 15 15
X_train, X_unused, y_train, y_unused = train_test_split(X, y, test_size=0.3, random_state=42)

# Zakoduj etykiety klas w postaci one-hot encoding
y_train_encoded = to_categorical(y_train.map({"Iris-setosa": 0, "Iris-virginica": 1, "Iris-versicolor": 2}))

# Przekształć dane do postaci, która może być użyta w treningu modelu
y = y_train_encoded
X = X_train.values.T

print("Kształt y:", y.shape)
print("Kształt X:", X.shape)
