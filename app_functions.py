from os import pread
import streamlit as st
import pandas as pd
import numpy as np

# fix random seed for reproducibility
np.random.seed(7)

def file_upload(name):
    uploaded_file = st.sidebar.file_uploader('%s' % (name),key='%s' % (name),accept_multiple_files=False)
    content = False
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            content = True
            return content, uploaded_df
        except:
            try:
                uploaded_df = pd.read_excel(uploaded_file)
                content = True
                return content, uploaded_df
            except:
                st.error('Please ensure file is .csv or .xlsx format and/or reupload file')
                return content, None
    else:
        return content, None

def extract_features_from_date(df):
    
    df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
    df.index=df['Date']
    # df.drop('Date',axis=1,inplace=True)
    df = df.sort_index(axis=0)
    return df

def create_dataset(dataset, look_back=1):
    '''
    For the features (x), we will always append the last 50 prices, 
    and for the label (y), we will append the next price. 
    Then we will use numpy to convert it into an array.
    '''
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def create_period_shift(df,period=4):
    tmp_df = df.copy()
    col_name = list(df)
    for i in range(period):
        new_col = (col_name[0]+'_'+str(i+1))
        tmp_df[new_col] = tmp_df[col_name].shift(-(i+1))
    
    return tmp_df.drop(col_name,axis=1).dropna()