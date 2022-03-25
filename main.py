import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from knn import KNN


def load_data(filename):
    """
    Function to load txt file into a dict
        :param str filename : string containing name of the file
        :return dict data_dict : dictionary containing array of decision attributes and target array
    """
    df_iris_train = pd.read_csv(filename, header=None, sep='\s+', decimal=',')

    data_array = df_iris_train.iloc[:, :-1].to_numpy()
    target_array = df_iris_train.iloc[:, -1].to_numpy()
    data_dict = {
        "data": data_array,
        "target": target_array
    }
    return data_dict


def accuracy(y_true, y_pred):
    """
    Function to compute the accuracy of prediction
        :param str y_true : string containing name of the file
        :param str y_pred : string containing name of the file
        :return float : number of correctly predicted target divided by nuber of all targets
    """
    return np.sum(y_true == y_pred) / len(y_true)


def plot_k_accuracy_correlation():
    """
    Function to plot how the classification accuracy changes with k
    """
    scores = []
    for a in range(1, 100, 2):
        clf = KNN(k=a)
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        scores.append(accuracy(y_test, predictions))

    sns.set_style("darkgrid")
    fig = plt.figure()
    plt.xlabel("K value")
    plt.ylabel("Accuracy")
    plt.title("Correlation between classification accuracy and K value")
    plt.plot(range(1, 100, 2), scores)
    plt.show()
    fig.savefig('plot.png', dpi=fig.dpi)


iris_train = load_data('iris_training.txt')
x_train = iris_train["data"]
y_train = pd.Series(iris_train["target"], dtype="category").cat.codes.values

classes = dict(zip(pd.unique(pd.Series(y_train, dtype="category").cat.codes.values), pd.unique(iris_train['target'])))

iris_test = load_data('iris_test.txt')
x_test = iris_test["data"]
y_test = pd.Series(iris_test["target"], dtype="category").cat.codes.values


plot_k_accuracy_correlation()

print(f"Input the parameter k: ")
k = int(input())

clf = KNN(k=k)
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
score = accuracy(y_test, predictions)
print(f"Number of correctly classified samples: {score * len(y_test)}")

print(f"Accuracy: {score}")

# Loop will run infinitely till user enters 'quit'
while True:
    user_input = input('Enter a sample to classify : ')
    if user_input == 'quit':
        break
    lst = list(map(float, user_input.strip().split()))[:len(x_test[0])]
    print(lst)
    prediction = clf.get_single_prediction(lst)
    print(classes.get(prediction))





