import pandas as pd
from sklearn.utils import shuffle


def to_categorical(y):
    unique_classes = list(set(y))
    class_mapping = {cls: i for i, cls in enumerate(unique_classes)}

    encoded_y = []
    for label in y:
        encoded_label = [0] * len(unique_classes)
        encoded_label[class_mapping[label]] = 1
        encoded_y.append(encoded_label)

    return encoded_y, class_mapping


# ----------------------------------- IRYS DATA ----------------------------------------------
df = pd.read_csv("iris.data", sep=',')
df = df.dropna()

y = df["classi"]

X = df[["sepal length", "sepal width", "petal length", "petal width"]]

X_train, y_train = shuffle(X, y, random_state=56)
y_train_encoded, class_mapping = to_categorical(
    y_train.map({"Iris-setosa": 0, "Iris-virginica": 1, "Iris-versicolor": 2}))

y = y_train_encoded
X = X_train.values.T
