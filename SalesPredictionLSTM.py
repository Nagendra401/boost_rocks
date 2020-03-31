
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

################################ load data ######################
train_data = pd.read_csv("C:\\Users\\NAGENDRA_CHAPALA\\Documents\\7400618_810833986\\DataScienceCourse_TechEssential\\course\\DATA_SCIENCE_AUTHORITY\\Skillenz_Hackthon\\Training-Data-Sets.csv")

# normalize features -
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(train_data.values)
series_train_data = pd.DataFrame(scaled)

#create a label set
label_df = pd.DataFrame(index = series_train_data.index, columns = ['EQ'])
label_df['EQ'] = np.log(train_data['EQ'])
label_df.shape

######################## Divide data into train and test data ########################
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
x_train , x_test , y_train , y_test = train_test_split(series_train_data, label_df, test_size = 0.20, random_state = 2)
x_train.shape
x_test.shape
y_train.shape
y_test.shape

X_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
X_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

# Define the LSTM model
model = Sequential()
model.add(LSTM(4, batch_input_shape=(1, X_train.shape[0], X_train.shape[1]), stateful=True))
model.add(Dropout(0.5))
#model.add(LSTM(256))
#model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("linear"))
model.compile(loss="mse", optimizer="adam")
model.summary()

start = time.time()
model.fit(x_train, y_train, batch_size=512, nb_epoch=3, validation_split=0.1)
print("> Compilation Time : ", time.time() - start)


# Doing a prediction on all the test data at once
preds = model.predict(x_test)

preds = scaler.inverse_transform(preds)
actuals = scaler.inverse_transform(y_test)
mean_squared_error(actuals,preds)