import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
features = []
uploaded_file = st.file_uploader("Chọn file:")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    st.write("Chose input features: ")
    ncols = len(df.columns[:-1])
    cols = st.columns(ncols)
    for i, c in enumerate(cols):
        if c.checkbox(df.columns[i]):
            features.append(df.columns[i])
    X = df[features]
    y = df[df.columns[-1]]
    st.write("Output: ", df.columns[-1])
    split = st.number_input("Nhập hệ số chia train và test: ", min_value = 0.1, max_value = 1.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split)
    if st.checkbox("KFold: "):
        number = st.number_input("Chọn hệ số k: ", min_value =2, format="%d")
    if st.button("Run"):
        st.write('Waiting')
        

    
    
    
    
