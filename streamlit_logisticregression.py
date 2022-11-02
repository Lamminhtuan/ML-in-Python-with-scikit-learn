import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
features = []
st.write('**Lâm Minh Tuấn - 20520843 - Logistic Regression**')
uploaded_file = st.file_uploader("Chose file:")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    df = df.dropna()
    st.write("Chose input features: ")
    ncols = len(df.columns[:-1])
    cols = st.columns(ncols)
    for i, c in enumerate(cols):
        if c.checkbox(df.columns[i]):
            features.append(df.columns[i])
    isnumber = features.copy()
    X = df[features]
    y = df[df.columns[-1]]
    st.write("Output: ", df.columns[-1])
    #one hot encoding for categorial features
    for i in features:
            if pd.to_numeric(X[i], errors='coerce').notnull().all() == False:
                isnumber.remove(i)
                one_hot = pd.get_dummies(X[i])
                X = X.drop(i, axis=1)
                X = X.join(one_hot)
    left, right = st.columns(2)
    with left:
        st.write('##')
        st.write('Enter train test split ratio: ')
    with right:
        tr_size = st.number_input('', min_value = 0.1, max_value = 1.0, value = 0.8)
        t_size = 1 - tr_size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = t_size, random_state = 42)
    #standarize data on non categorial columns
    ct = ColumnTransformer([('scale', StandardScaler(), isnumber)], remainder = 'passthrough')
    scaler = ct.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    usekfold = st.checkbox("KFold: ")
    left, right = st.columns(2)
    with left:
        st.write('##')
        st.write('Enter k for KFold cross-validator: ')
    with right:
        if usekfold:
            k = st.number_input('', min_value =2, format="%d")
    left, right = st.columns(2)
    with left:
        f1 = st.checkbox('F1 Score')
    with right:
        log = st.checkbox('Log Loss')
    if st.button("Run"):
        if usekfold:
            
            f1_train_list = []
            f1_test_list = []
            log_train_list = []
            log_test_list = []
            kf = KFold(k)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                reg = LinearRegression()
                reg.fit(X_train, y_train)
                y_pred_train = reg.predict(X_train)
                y_pred_test = reg.predict(X_test)
                f1_train_list.append(mean_squared_error(y_train, y_pred_train))
                f1_test_list.append(mean_squared_error(y_test, y_pred_test))
                log_train_list.append(mean_absolute_error(y_train, y_pred_train))
                log_test_list.append(mean_absolute_error(y_test, y_pred_test))
            fig_mse = plt.figure()
            n = np.arange(len(mse_test_list))
            plt.bar(n - 0.2, mse_train_list, color='r', width=0.4, label="MSE_Train")
            plt.bar(n + 0.2, mse_test_list, color='g', width=0.4, label="MSE_Test")
            plt.ylabel('MSE')
            plt.xlabel('Subsets')
            plt.title('Mean squared error')
            plt.xticks(n)
            plt.legend()
            st.pyplot(fig_mse)

            fig_mae = plt.figure()
            n_ = np.arange(len(mae_test_list))
            
            plt.bar(n_ - 0.2, mae_train_list, color='r', width=0.4, label="MAE_Train")
            plt.bar(n_ + 0.2, mae_test_list, color='g', width=0.4, label="MAE_Test")
            plt.ylabel('MAE')
            plt.xlabel('Subsets')
            plt.title('Mean absolute error')
            plt.xticks(n_)
            plt.legend()
            
            st.pyplot(fig_mae)
        else:
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            y_pred_train = reg.predict(X_train)
            y_pred_test = reg.predict(X_test)
            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_test =  mean_squared_error(y_test, y_pred_test)
            mae_train = mean_absolute_error(y_train, y_pred_train)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            st.write('Mean squared error on train dataset: ', mse_train)
            st.write('Mean squared error on test dataset: ', mse_test)
            st.write('Mean absolute error on train dataset: ', mae_train)
            st.write('Mean absolute error on test dataset: ', mae_test)
            fig_mse = plt.figure() 
            plt.bar('mse_train', mse_train, color='r')
            plt.bar('mse_test', mse_test, color='g')   
            plt.ylabel('MSE')
            plt.xlabel('Train and test datasets')
            plt.title('Mean squared error')
            st.pyplot(fig_mse)
            fig_mae = plt.figure() 
            plt.bar('mae_train', mae_train, color='r')
            plt.bar('mae_test', mae_test, color='g')   
            plt.ylabel('MAE')
            plt.xlabel('Train and test datasets')
            plt.title('Mean absolute error')
            st.pyplot(fig_mae)