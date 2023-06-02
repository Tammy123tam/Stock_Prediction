# Stock-Prediction

## Introduction
Welcome to our Stock Market Prediction Website! This platform utilizes a Recurrent Neural Network with LSTM to forecast US stock market prices. It allows users to input a stock name and prediction parameters to obtain the predicted price for the next day and visualize the stock price prediction graph.

## Technical Overview
- The stock prediction model is implemented using Python and TensorFlow. It utilizes historical stock price data to train a Long Short-Term Memory (LSTM) model. The LSTM model is designed with three LSTM layers, each followed by a dropout layer to prevent overfitting. The model is trained using the Adam optimizer and mean squared error loss function.

- The Streamlit framework is used to create a user-friendly web interface. Users can enter the stock name, number of epochs, number of unrollings, and number of days for prediction. The application validates the input, performs the stock prediction, and displays the predicted price and the stock price prediction graph.

## How to Install
1. Clone the repository: 
git clone 
2. Navigate to the project directory: 
cd <projetc directory>
3. Install the required dependencies:
pip install -r requirements.txt
  
## Usage
1. Run the Streamlit app:  
streamlit run main.py  
2. Access the application through the provided URL in your browser.
3. Enter the stock name (in capital letters), the number of epochs, number of unrollings, and number of days for prediction.
4. Click the "Predict" button to perform the stock prediction.
5. The predicted price for the next day will be displayed, along with the stock price prediction graph (if available).

Note: Make sure to have the necessary stock data files in the "Stocks" directory and ensure the file naming follows the convention "<stock_name>.us.txt".
  
  
  
  
  
  
  
