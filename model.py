import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings("ignore")

def check_stock_name(stock_name):
    stocks_folder = 'Stocks'
    stock_file = f"{stock_name.lower()}.us.txt"
    file_path = os.path.join(stocks_folder, stock_file)
    
    if os.path.isfile(file_path):
        return True
    else:
        return False

def stock_predict(stock_name, ephochs, no_unrollings, days):
    # load the data
    file_path = os.path.join('Stocks', f'{stock_name.lower()}.us.txt')
    df = pd.read_csv(file_path, delimiter=',', usecols=['Date', 'Open', 'High', 'Low', 'Close'], parse_dates=['Date'])
    # Split the data into training and testing datasets
    train_size = int(len(df) * 0.8)
    train_data = df[:train_size]
    test_data = df[train_size:] 
    # scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(train_data['Close'].values.reshape(-1, 1))

    
    x_train, y_train = [], []  # x is data feature and y is label, next stock price
    for i in range(days, len(scaled_data) - no_unrollings):
        x_train.append(scaled_data[i - days : i, 0])
        y_train.append(scaled_data[i + no_unrollings, 0])
    assert len(x_train) == len(y_train)
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) 
    real_prices = test_data['Close']
    total_data = pd.concat((train_data['Close'], test_data['Close']), axis=0)
    model_test = total_data[len(total_data) - len(test_data) - no_unrollings:].values
    model_test = model_test.reshape(-1, 1)
    model_test = scaler.transform(model_test)
    x_test = []
    for i in range(days, len(model_test)):
        x_test.append(model_test[i-days : i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = test_data['Close'].values[:len(x_test)]
    #build LSTM model 
    model = Sequential()
    # The 1st layer
    model.add(LSTM(units = 50, return_sequences= True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2)) # Add dropout to increase efficiency and avoid overfitting

    # The 2nd layer
    model.add(LSTM(units = 50, return_sequences= True))
    model.add(Dropout(0.2))

    # The 3rd layer
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(units= 1))

    # Compile the model
    model.compile(optimizer = 'adam', loss = 'mean_squared_error',  metrics=['mean_squared_error', 'mae'])

    # train the model 
    checkpointer = ModelCheckpoint(filepath= 'best_weight.hdf5', verbose= 2, save_best_only= True)
    model.fit(x_train, y_train, epochs= ephochs, batch_size = 32, validation_data=(x_test, y_test), callbacks= [checkpointer] )

    #Prediction
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    
    # Plot the graph
    save_dir = 'Stocks/save/'
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(16, 8))
    plt.plot(df.loc[train_size:, 'Date'], real_prices, color='red', label=f'Real {stock_name} Stock Price')
    plt.plot(df.loc[train_size:train_size+len(predicted_prices)-1, 'Date'], predicted_prices, color='blue', label=f'Predicted {stock_name} Stock Price')
    plt.title(f'{stock_name} Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'{stock_name} Price')
    plt.legend()
    plt.xticks(rotation=90)
    plt.savefig(save_dir + stock_name + '.png')
    plt.clf()
    plt.close()

    real_data = np.array([model_test[len(model_test) + 1 - no_unrollings:len(model_test + 1), 0]])
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    predicted_p = model.predict(real_data)
    predicted_p = scaler.inverse_transform(predicted_p)
    return predicted_p[0][0]






