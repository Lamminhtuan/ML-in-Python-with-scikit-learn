import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
features = []
uploaded_file = st.file_uploader("Chọn file:")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    st.write("Chose input features:")
    for i in df.columns[:-1]:
        if st.checkbox(i):
            features.append(i)
    X = df[features]
    y = df[df.columns[-1]]
    st.write("Output: ", df.columns[-1])
    split = st.number_input("Nhập hệ số chia train và test: ", min_value = 0.01, max_value = 1.0, format=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split)
    if st.checkbox("KFold: "):
        number = st.number_input("Chọn hệ số k: ")
    if st.button("Run"):
        st.write("Waiting")
        

    
    
    
    
