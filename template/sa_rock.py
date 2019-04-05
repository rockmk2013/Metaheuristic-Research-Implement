import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

# 資料集是以dictionary的形式存在
cancer = load_breast_cancer()
cancer.keys()

df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_feat.info()
df_feat.head()

# Data Preprocessing
X = df_feat.iloc[:, ].values
y = cancer['target']

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# check
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# training ANN
# %env KERAS_BACKEND = tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD,adam

# Initialising the ANN
classifier = Sequential()

## Construct ANN ##
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=16, init='uniform', activation='relu', input_dim=30))
# Adding dropout to prevent overfitting
classifier.add(Dropout(p=0.1))
# Adding the second hidden layer
classifier.add(Dense(output_dim=16, init='uniform', activation='relu'))
# Adding dropout to prevent overfitting
classifier.add(Dropout(p=0.1))
# Adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Model Summary
classifier.summary()

# fit ANN
classifier.fit(X_train,y_train,batch_size=100,epochs=100)

