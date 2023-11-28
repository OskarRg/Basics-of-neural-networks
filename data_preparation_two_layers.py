# Tutaj będą dane robione, ale dla 2 warstwowej sieci, więc pewnie mniej niż 14 wejść użyjemy np 4, 5.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
selected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = heart_disease.data.features[selected_features]
y = heart_disease.data.targets

# Połącz X i y w jedną ramkę danych
data = pd.concat([X, y], axis=1)

# Usuń wiersze zawierające NaN
data_cleaned = data.dropna()

# Wybierz 5 najważniejszych cech
selected_features_5 = ['age', 'restecg', 'trestbps', 'chol', 'fbs']

selected_features_5 = ['chol', 'fbs']

# Utwórz DataFrame z wybranymi cechami i zmienną docelową
data_5 = data_cleaned[selected_features_5 + ['num']]

# Przekształć DataFrame do macierzy numpy
X_5 = data_5[selected_features_5].values
y_5 = data_5['num'].values.flatten()

# Zamień kategorie 'num' na 0 i 1
y_binary_5 = np.where(y_5 > 0, 1, 0)

# Transponuj macierz cech, aby pasowała do twojego pierwotnego podejścia
P_5 = X_5.T

# Utwórz wektor targetów T
T_5 = y_binary_5
print("T")
print(T_5)
print("P")
print(P_5)
print(P_5.shape)
