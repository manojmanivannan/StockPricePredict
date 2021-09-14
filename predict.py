import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import base64
import os
from app_functions import *

st.sidebar.subheader('Stock Dataset')
status, df = file_upload('Please upload a stock price dataset')

st.title('Stock Price Prediction')

if not status:
    st.write('Please use the sidebar to update your data !')

if status:
    st.subheader('Preview of the dataset')
    df = extract_features_from_date(df)

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