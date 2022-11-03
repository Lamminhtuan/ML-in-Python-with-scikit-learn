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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
features = []
st.markdown('**Lâm Minh Tuấn - 20520843 - CS116.N11 - Linear Regression**')
uploaded_file = st.file_uploader("Chose file:")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    df = df.dropna()
    st.write("Chose input features: ")
    nrows = int(np.ceil(len(df.columns[:-1]) / 4))
    rows = [st.columns(4) for _ in range(nrows)]
    cols = [column for row in rows for column in row]
    left, right = st.columns(2)
    with left:
        selectall = st.button('Select All')
    with right:
        delselect = st.button('Deselect All')
    for i, col in enumerate(cols):
        if i >= len(df.columns[:-1]):
            col.empty()
        else:
            if col.checkbox(df.columns[i]):
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
    if st.button("Run"):
        if usekfold:
            mse_train_list = []
            mse_test_list = []
            mae_train_list = []
            mae_test_list = []
            kf = KFold(k)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                reg = LinearRegression()
                reg.fit(X_train, y_train)
                y_pred_train = reg.predict(X_train)
                y_pred_test = reg.predict(X_test)
                mse_train_list.append(mean_squared_error(y_train, y_pred_train))
                mse_test_list.append(mean_squared_error(y_test, y_pred_test))
                mae_train_list.append(mean_absolute_error(y_train, y_pred_train))
                mae_test_list.append(mean_absolute_error(y_test, y_pred_test))
            mse_avg_train = sum(mse_train_list) / len(mse_train_list)
            mse_avg_test = sum(mse_test_list) / len(mse_test_list)
            mae_avg_train = sum(mae_train_list) / len(mae_train_list)
            mae_avg_test = sum(mae_test_list) / len(mae_test_list)
            fig_mse = plt.figure()
            n = np.arange(len(mse_test_list))
            plt.bar(n - 0.2, mse_train_list, color='r', width=0.4, label="MSE_Train")
            plt.bar(n + 0.2, mse_test_list, color='g', width=0.4, label="MSE_Test")
            plt.ylabel('MSE')
            plt.xlabel('Folds')
            plt.title('Mean squared error')
            plt.xticks(n)
            plt.legend()
            st.pyplot(fig_mse)
            fig_mae = plt.figure()
            n_ = np.arange(len(mae_test_list))
            plt.bar(n_ - 0.2, mae_train_list, color='r', width=0.4, label="MAE_Train")
            plt.bar(n_ + 0.2, mae_test_list, color='g', width=0.4, label="MAE_Test")
            plt.ylabel('MAE')
            plt.xlabel('Folds')
            plt.title('Mean absolute error')
            plt.xticks(n_)
            plt.legend()
            st.pyplot(fig_mae)
            fig_mse_avg = plt.figure()
            plt.bar('mse_avg_train', mse_avg_train, color='r')
            plt.bar('mse_avg_test', mse_avg_test, color='g')   
            plt.ylabel('MSE')
            plt.xlabel('Train and test datasets')
            plt.title('Average Mean squared error of Folds')
            st.pyplot(fig_mse_avg)
            fig_mae_avg = plt.figure()
            plt.bar('mae_avg_train', mae_avg_train, color='r')
            plt.bar('mae_avg_test', mae_avg_test, color='g')   
            plt.ylabel('MAE')
            plt.xlabel('Train and test datasets')
            plt.title('Average Mean absolute error of Folds')
            st.pyplot(fig_mae_avg)
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