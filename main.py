from model import check_stock_name, stock_predict
import streamlit as st
import os

# Set the page title
st.title('Stock Market Prediction')
st.write("Welcome to our Stock Price Prediction Website! This platform utilizes a Recurrent Neural Network with LSTM to forecast US stock market prices.")

# Input
stock_name = st.text_input('Enter the stock name (You should write in captial)')
epochs = st.number_input('Enter the number of epochs (number of passes of the entire training dataset. Higher epochs can improve accuracy)', min_value=1, step=1)
unrollings = st.number_input('Enter the number of unrollings (number of days you want to predict ahead of time)', min_value=1, step=1)
days = st.number_input('Enter the number of days used for prediction', min_value=1, step=1)

# Button to trigger prediction
if st.button('Predict'):
    # Check if the stock name is valid
    try: 
        if stock_name and not check_stock_name(stock_name):
            st.error('Invalid stock name. Please enter a valid stock name.')
    except: 
        pass 

    try: 
        if unrollings >= days:
            st.error('Days should be larger than unrollings')
    except: 
        pass 
    # Calculate
    predicted_price = stock_predict(stock_name, epochs, unrollings, days)
    if predicted_price is not None:
        # Display the predicted price
        st.subheader('Predicted Price')
        st.write(f"The predicted price for the next day is: ${predicted_price:.2f}")
        
        # Display the graph
        image_path = os.path.join('Stocks/save/', f'{stock_name.lower()}.png')
        if os.path.isfile(image_path):
            st.subheader('Stock Price Prediction Graph')
            st.image(image_path)
        else:
            st.warning('Graph not available.')

# I was trying to use flask to build UI demo but I was desperate with fixing bugs and it was so close to deadline so I use streamlit as alternatives :<<
# from flask import Flask, render_template, request
# app = Flask(__name__)

# @app.route('/')
# def form():
#     return render_template('form.html')

# @app.route('/data', methods=['POST'])
# def request_stock(): 
#     name = request.form['stock_name'].upper()
#     epchochs = int(request.form['Epochs'])
#     ahead = int(request.form['Ahead'])
#     days = int(request.form['Days'])
#     if not check_stock_name(name):
#         return render_template('stock_result.html', error_message='Invalid stock name')
#     if ahead >= days: 
#         return render_template('stock_result.html', error_message='Days should be bigger than ahead')
#     predicted_prices = stock_predict(name, epchochs, ahead, days)
#     return render_template('stock_result.html', stock_detail = [name, epchochs, ahead, days], stock_price = predicted_prices)

# if __name__ == "__main__":
#     app.run(debug = True)

