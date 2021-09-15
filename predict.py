import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
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
    df = extract_features_from_date(df)
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
    col1, col2, col3 = st.columns((3,3,2))

    with col1:
        feature_cols = st.multiselect('Please select features',col_names)
    with col2:
        label_col = st.selectbox('Please select label',col_names)
    with col3:
        test_size = st.number_input('Please enter test size',0.01,0.99,0.25,0.05)