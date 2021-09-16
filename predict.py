import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import base64
import os
from app_functions import *
from streamlit_echarts import st_echarts

st.sidebar.subheader('Stock Dataset')
status, df = file_upload('Please upload a stock price dataset')

st.title('Stock Price Prediction')
st.subheader('A simple stock Prediction App')

if not status:
    st.write('Please use the sidebar to update your data !')



if status:
    
    with st.expander('View dataset'):
        st.write(df)
    extract_features_from_date(df)
    
    feature_cols = st.multiselect('Please select columns to plot',list(df),key='plot_col')
    if feature_cols:
        if 'Date' in feature_cols:
            feature_cols.remove('Date')
            st.write('Ignoring "Date" column')
        if len(feature_cols) ==0:
            st.write('Please atleast one column')
        else:
            st.line_chart(df[feature_cols],use_container_width=True)

    # option = {
    #     "xAxis": {
    #         "type": "category",
    #         "data": df.index.astype(str).values.tolist(),
    #     },
    #     "yAxis": {"type": "value"},
    #     "series": [{"data": df["Close"].astype(str).values.tolist(), "type": "scatter"}],
    #     "lineStyle": [{ "color": '#5470C6', "width": '4', "type": 'dashed'}]
    #     }
    # st_echarts(
    #     options=option, height="400px",
    # )

if status == True:
    col_names = list(df)

    st.title('Training')
    st.subheader('Parameters')
    col1, col2, col3 = st.columns(3)

    with col1:
        label_col = st.selectbox('Please select column to predict',col_names)
    with col2:
        test_size_ratio = st.number_input('Please enter test size',0.01,0.99,0.25,0.05)
    with col3:
        period = int(st.number_input('Please enter lookback Period',1,10,4,1))
    l_col1, l_col2 = st.columns(2)
    with l_col1: layer_1 = st.slider('Layer 1 Nodes', min_value=1, max_value=100, value=50, step=1)
    with l_col2: layer_2 = st.slider('Layer 2 Nodes', min_value=1, max_value=100, value=50, step=1)

    with st.expander('Advanced Parameters'):
        col4_1, col4_2 = st.columns(2)
        with col4_1:
            optimizer = st.selectbox('Solver',['Adam','Adagrad','RMSprop','Nadam'])
        with col4_2:
            loss = st.selectbox('Loss Function',['mean_squared_error','mean_absolute_error'])
        col4_3, col4_4 = st.columns(2)
        with col4_3:
            epochs = int(st.number_input('Training epoch',1,100,10,5))
        with col4_4:
            batchsize = int(st.number_input('Batch Size',1,100,10,5))

    if label_col == 'Date':
        st.write('Can not apply model on Date column')
        st.stop()
    
    data=df.sort_index(ascending=True,axis=0)

    predict_df=data[[label_col]]

    dataset = predict_df.values
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    test_size = int(len(dataset) * test_size_ratio)
    train_size = len(dataset) - test_size

    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    trainX, trainY = create_dataset(train, period)
    testX, testY = create_dataset(test, period)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))



    lstm_model=Sequential()
    lstm_model.add(LSTM(units=layer_1,return_sequences=True,input_shape=(1, period)))
    lstm_model.add(LSTM(units=layer_2))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss=loss, optimizer=optimizer)
    lstm_model.fit(trainX, trainY, epochs=epochs, batch_size=batchsize, verbose=2)

    # make predictions
    trainPredict = lstm_model.predict(trainX)
    testPredict = lstm_model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[period:len(trainPredict)+period, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(period*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    fig = plt.figure()
    # plt.plot(scaler.inverse_transform(dataset))
    main_df = pd.DataFrame(scaler.inverse_transform(dataset),columns=[label_col])
    main_df['train']=trainPredictPlot
    main_df['test'] = testPredictPlot
    st.line_chart(main_df)


