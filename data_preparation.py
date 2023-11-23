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

# Przekształć DataFrame do macierzy numpy
X_cleaned = data_cleaned[selected_features].values
y_cleaned = data_cleaned['num'].values.flatten()  # 'y' powinno być wektorem jednowymiarowym

# Zamień kategorie 'num' na 0 i 1
y_binary_cleaned = np.where(y_cleaned > 0, 1, 0)

# Transponuj macierz cech, aby pasowała do twojego pierwotnego podejścia
P_cleaned = X_cleaned.T

# Utwórz wektor targetów T
T_cleaned = y_binary_cleaned
print("T")
print(T_cleaned)
print("P")
print(P_cleaned)
print(P_cleaned.shape)

'''
# metadata
print(heart_disease.metadata)
# variable information
print(heart_disease.variables)
print(heart_disease)
'''