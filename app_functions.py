import streamlit as st
import pandas as pd
import numpy as np

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

def create_dataset(df):
    '''
    For the features (x), we will always append the last 50 prices, 
    and for the label (y), we will append the next price. 
    Then we will use numpy to convert it into an array.
    '''
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y