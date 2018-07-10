import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

df = pd.read_csv('data.csv') # read data set using pandas
print(df.info()) # Overview of dataset
df = df.drop(['Date'],axis=1) # Drop Date feature
df = df.dropna(inplace=False)  # Remove all nan entries.

df = df.drop(['Adj Close','Volume'],axis=1) # Drop Adj close and volume feature

df_train = df[:1059]    # 60% training data and 40% testing data
df_test = df[1059:]
scaler = MinMaxScaler() # For normalizing dataset

X_train=df_train.drop(['Close'],axis=1).as_matrix()
Y_train=df_train['Close'].as_matrix()

X_test=df_test.drop(['Close'],axis=1).as_matrix()
Y_test=df_test['Close'].as_matrix()

np.save('Xtrain',X_train)
np.save('Ytrain',Y_train)
np.save('Xtest',X_test)
np.save('Ytest',Y_test)


print(np.size(Y_train,0))

#X_train = scaler.fit_transform(df_train.drop(['Close'],axis=1).as_matrix())
#y_train = scaler.fit_transform(df_train['Close'].as_matrix())

#X_test = scaler.fit_transform(df_test.drop(['Close'],axis=1).as_matrix())
#y_test = scaler.fit_transform(df_test['Close'].as_matrix())

#print(X_train)
