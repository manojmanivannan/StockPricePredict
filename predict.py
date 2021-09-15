import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
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

if not status:
    st.write('Please use the sidebar to update your data !')



if status:
    st.subheader('Preview of the dataset')
    # df = extract_features_from_date(df)
    st.write(df.head())
    feature_cols = st.multiselect('Please select columns to plot',list(df),key='plot_col')
    if feature_cols:
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
    col2, col3 = st.columns(2)

    with col2:
        label_col = st.selectbox('Please select column to predict',col_names)
    with col3:
        test_size = st.number_input('Please enter test size',0.01,0.99,0.25,0.05)



    data=df.sort_index(ascending=True,axis=0)
    new_dataset=data[['Date',label_col]]

    scaler=MinMaxScaler(feature_range=(0,1))
    final_dataset=new_dataset.values
    st.write(final_dataset.shape)
    train_data=final_dataset[0:200,:]
    valid_data=final_dataset[200:,:]

    # new_dataset.index=new_dataset.Date
    new_dataset.drop("Date",axis=1,inplace=True)
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(new_dataset)

    x_train_data,y_train_data=[],[]

    for i in range(60,len(train_data)):
        x_train_data.append(scaled_data[i-5:i,0])
        y_train_data.append(scaled_data[i,0])
        
    x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

    x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

    l_col1, l_col2 = st.columns(2)
    with l_col1: layer_1 = st.slider('Layer 1 Nodes', min_value=1, max_value=100, value=50, step=1)
    with l_col2: layer_2 = st.slider('Layer 2 Nodes', min_value=1, max_value=100, value=50, step=1)

    lstm_model=Sequential()
    lstm_model.add(LSTM(units=layer_1,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
    lstm_model.add(LSTM(units=layer_1))
    lstm_model.add(Dense(1))

    inputs_data=new_dataset[len(new_dataset)-len(valid_data)-5:].values
    inputs_data=inputs_data.reshape(-1,1)
    inputs_data=scaler.transform(inputs_data)

    lstm_model.compile(loss='mean_squared_error',optimizer='adam')
    lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

    X_test=[]
    for i in range(5,inputs_data.shape[0]):
        X_test.append(inputs_data[i-5:i,0])
    X_test=np.array(X_test)

    st.write(inputs_data)

    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    predicted_closing_price=lstm_model.predict(X_test)
    predicted_closing_price=scaler.inverse_transform(predicted_closing_price)

    lstm_model.save("saved_model.h5")


    train_data=new_dataset[:200]
    valid_data=new_dataset[200:]
    valid_data['Predictions']=predicted_closing_price

    fig = plt.figure()
    plt.plot(train_data["Close"])
    plt.plot(valid_data[['Close',"Predictions"]])
    st.pyplot(fig)