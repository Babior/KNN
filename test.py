import pandas as pd
import numpy as np
from sklearn import datasets


df_iris_train = pd.read_csv('iris_training.txt', header=None, sep='\s+', decimal=',')
df_iris_test = pd.read_csv('iris_test.txt', header=None, sep='\s+', decimal=',')

# print(df_iris_train.head())
# print(df_iris_train.info())
# print(df_iris_train.describe())
#
# print(df_iris_test.head())
# print(df_iris_test.info())
# print(df_iris_test.describe())

# data_array = df_iris_train.iloc[:, :-1].to_numpy()
# #print(data_array)
#
# target_array = df_iris_train.iloc[:, -1].to_numpy()
# #print(target_array)
#
#
# iris_train = {
#   "data": data_array,
#   "target": target_array
# }
#
#
# pd.unique(iris_train['target'])
#
#
# dictionary = dict(zip(pd.unique(iris_train['target']), pd.unique(pd.Series(target_array, dtype="category").cat.codes.values)))
#
# print(dictionary)


num_list = [1,2,3,4,5]
num_list.remove(2)
print(num_list)