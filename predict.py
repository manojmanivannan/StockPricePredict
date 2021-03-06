import streamlit as st
import pandas as pd
import numpy as np
import datetime, time
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.utils.vis_utils import plot_model
from AppFunctions.stockFunc import *
from AppFunctions.information import information
from streamlit_echarts import st_echarts
import cufflinks as cf

st.set_page_config(
    page_title="Stock Price Prediction", 
    page_icon='https://cdn-icons-png.flaticon.com/512/2422/2422796.png')

st.sidebar.subheader('Stock Dataset')
status, df, file_name = file_upload('Please upload a stock price dataset')
st.sidebar.markdown(information['profile'],unsafe_allow_html=True)

st.title('Stock Price Prediction')
st.subheader('Using Keras Long-Short Term Memory (LSTM) Neural Network')
st.text("")

if not status:
    st.write('Please use the sidebar to upload your dataset !')
    with st.expander('Download from yahoo finance !'):
        company = st.selectbox('Select company',['AAPL','TSLA','BTC-USD'], index=1)
        st_dt, ed_dt = st.columns(2)
        with st_dt: start_date = st.date_input('From', datetime.date(2021,1,1))
        with ed_dt: end_date = st.date_input('To', datetime.date(2021,9,1))

        period1 = int(time.mktime(start_date.timetuple()))
        period2 = int(time.mktime(end_date.timetuple()))
        interval = '1d' # 1d, 1m

        query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{company}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
        df_downloaded = pd.read_csv(query_string)
        st.write(df_downloaded)
        st.download_button('Download stock data', df_downloaded.to_csv(), file_name=f'{company}.csv')


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
            for each in feature_cols:
                df[each] = df[each].replace({'\$': '', ',': '','???':''}, regex=True).astype(float)

            fig_plot = df[feature_cols].iplot(asFigure=True)
            fig_plot.update_layout(plot_bgcolor='rgba(17,17,17,0)',paper_bgcolor ='rgba(10,10,10,0)', legend_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_plot)

            # st.line_chart(df[feature_cols],use_container_width=True)

if status == True:
    col_names = list(df)

    st.title('Training')
    st.subheader('Parameters')
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        label_col = st.selectbox('Column to predict',col_names)
    with col2:
        test_size_ratio = st.number_input('Test size',0.01,0.99,0.25,0.05)
    with col3:
        period = int(st.number_input('Lookback Period',1,10,4,1))
    with col4:
        no_layers = int(st.number_input('# of layers',2,10,2,1))

    if label_col == 'Date':
        st.write('Can\'t apply model on \'Date\' column. Select another column to proceed !')
        st.stop()
    

    l_col1, l_col2, *l_colx = st.columns(no_layers)

    with l_col1: layer_1 = st.slider('Layer 1 Nodes', min_value=1, max_value=100, value=10, step=1)
    with l_col2: layer_2 = st.slider('Layer 2 Nodes', min_value=1, max_value=100, value=10, step=1)
    layer_list = [period,layer_1, layer_2]
    layer_desc = ['Input','Layer 1', 'Layer 2']
    if no_layers>2:
        i=2
        for each in l_colx:
            with each: 
                globals()['layer_'+str(i+1)] = st.slider(f'Layer {i+1} Nodes', min_value=1, max_value=100, value=10, step=1)
            layer_list.append(globals()['layer_'+str(i+1)])
            layer_desc.append('Layer '+str(i+1))
            i+=1
    layer_list.append(1)
    layer_desc.append('Output')

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


    data=df.sort_index(ascending=True,axis=0)
    data[label_col] = data[label_col].replace({'\$': '', ',': '','???':''}, regex=True).astype(float)
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
    if no_layers == 2:
        lstm_model.add(LSTM(units=layer_2))
    else:
        lstm_model.add(LSTM(units=layer_2,return_sequences = True))
    if no_layers>2:
        i=2
        for each in l_colx:
            if i>=len(l_colx):
                lstm_model.add(LSTM(units=globals()['layer_'+str(i+1)]))
            else:
                lstm_model.add(LSTM(units=globals()['layer_'+str(i+1)],return_sequences = True))
            i+=1
    lstm_model.add(Dense(1))

    with st.expander('View model layer diagram'):

        plot_model(lstm_model,'data/image.png',show_shapes=True)
        st.image('data/image.png')

        fig = plt.figure(figsize=(12, 12))
        ax = fig.gca()
        ax.axis('off')
        draw_neural_net(ax, .1, .9, .1, .9, layer_list, layer_desc)
        st.pyplot(fig)

    lstm_model.compile(loss=loss, optimizer=optimizer)
    with st.spinner('Training Model..'):
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
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))


    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[period:len(trainPredict)+period, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(period*2)+1:len(dataset)-1, :] = testPredict

    result_df = pd.DataFrame(scaler.inverse_transform(dataset),columns=[label_col])
    result_df['Prediction on training set']=trainPredictPlot
    result_df['Prediction on training set'] = result_df['Prediction on training set'].shift(-2)

    result_df['Prediction on test set'] = testPredictPlot
    result_df['Prediction on test set'] = result_df['Prediction on test set'].shift(-2)

    result_df.index = data.index
    st.title('Result')
    result1, result2 = st.columns(2)
    with result1: st.write('Train Score: %.2f RMSE' % (trainScore))
    with result2: st.write('Test Score: %.2f RMSE' % (testScore))
    st.subheader('Plot')

    # st.line_chart(result_df)
    fig_result = result_df.iplot(asFigure=True)
    fig_result.update_layout(plot_bgcolor='rgba(17,17,17,0)',paper_bgcolor ='rgba(10,10,10,0)', legend_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_result)
    
    with st.expander('View result dataset'):
        st.write(result_df)
    st.download_button('Download result', result_df.to_csv(), file_name=f'{file_name}_prediction_results.csv')


