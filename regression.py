import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
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
    for i in features:
            if pd.to_numeric(X[i], errors='coerce').notnull().all() == False:
                one_hot = pd.get_dummies(X[i])
                X = X.drop(i, axis=1)
                X = X.join(one_hot)
    split = st.number_input("Nhập hệ số chia train và test: ", min_value = 0.1, max_value = 1.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    usekfold = st.checkbox("KFold: ")
    if usekfold:
        number = st.number_input("Chọn hệ số k: ", min_value =2, format="%d")
    if st.button("Run"):
        
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        st.write('Mean squared error: ', mean_squared_error(y_test, y_pred))

            

    
    
    
    
