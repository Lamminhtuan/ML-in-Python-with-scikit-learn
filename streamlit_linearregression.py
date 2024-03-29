import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
features = []
def check():
    for i in range(len(df.columns[:-1])):
        st.session_state[str(i)] = True
    return
def uncheck():
    for i in range(len(df.columns[:-1])):
        st.session_state[str(i)] = False
    return
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
        selectall = st.button('Select All', on_click=check)
    with right:
        delselect = st.button('Deselect All', on_click=uncheck)
    for i, col in enumerate(cols):
        if i >= len(df.columns[:-1]):
            col.empty()
        else:
            if col.checkbox(df.columns[i], key=str(i)):
                features.append(df.columns[i])
    X = df[features]
    y = df[df.columns[-1]]
    needstandarize = features.copy()
    for i in features:
        #Standarize the data on number columns and not the order column
        if pd.to_numeric(X[i], errors='coerce').notnull().all() == False or X[i].is_monotonic_increasing == True:
            needstandarize.remove(i)
    st.write("Output: ", df.columns[-1])
    #one hot encoding for categorial features
    if features:
        X = pd.get_dummies(X)
    left, right = st.columns(2)
    with left:
        st.write('##')
        st.write('Enter train test split ratio: ')
    with right:
        tr_size = st.number_input('', min_value = 0.1, max_value = 1.0, value = 0.8)
        t_size = 1 - tr_size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = t_size, random_state = 42)
    if needstandarize:
        ct = StandardScaler()
        ct.fit(X_train[needstandarize])
        X_train[needstandarize] = ct.transform(X_train[needstandarize])
        X_test[needstandarize] = ct.transform(X_test[needstandarize])
    usekfold = st.checkbox("K-Fold cross-validation")
    if usekfold:
        left, right = st.columns(2)
        with left:
            st.write('##')
            st.write('Enter k for K-Fold cross-validation: ')
        with right:
            k = st.number_input('', min_value =2, format="%d")
    left, right = st.columns(2)
    with left:
        btn_mse = st.checkbox('MSE')
    with right:
        btn_mae = st.checkbox('MAE')
    if st.button("Run"):
        if features:
            if usekfold:
                mse_train_list = []
                mse_test_list = []
                mae_train_list = []
                mae_test_list = []
                kf = KFold(k)
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    #Standarize the data
                    if needstandarize:
                        ct = StandardScaler()
                        ct.fit(X_train[needstandarize])
                        X_train[needstandarize] = ct.transform(X_train[needstandarize])
                        X_test[needstandarize] = ct.transform(X_test[needstandarize])
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
                if btn_mse:
                    fig_mse = plt.figure()
                    n = np.arange(len(mse_test_list))
                    plt.bar(n - 0.2, mse_train_list, color='r', width=0.4, label="MSE on Training Set")
                    plt.bar(n + 0.2, mse_test_list, color='g', width=0.4, label="MSE on Test Set")
                    plt.ylabel('MSE')
                    plt.xlabel('Folds')
                    plt.title('Mean squared error of Folds')
                    plt.xticks(n)
                    plt.legend()
                    st.pyplot(fig_mse)
                    fig_mse_avg = plt.figure()
                    plt.bar('Average MSE on Training Set', mse_avg_train, color='r')
                    plt.bar('Average MSE on Test Set', mse_avg_test, color='g')   
                    plt.ylabel('MSE')
                    plt.xlabel('Training and Test Datasets')
                    plt.title('Average Mean squared error of Folds')
                    st.pyplot(fig_mse_avg)
                if btn_mae:
                    fig_mae = plt.figure()
                    n_ = np.arange(len(mae_test_list))
                    plt.bar(n_ - 0.2, mae_train_list, color='r', width=0.4, label="MAE on Training Set")
                    plt.bar(n_ + 0.2, mae_test_list, color='g', width=0.4, label="MAE on Test Set")
                    plt.ylabel('MAE')
                    plt.xlabel('Folds')
                    plt.title('Mean absolute error of Folds')
                    plt.xticks(n_)
                    plt.legend()
                    st.pyplot(fig_mae)
                    fig_mae_avg = plt.figure()
                    plt.bar('Average MAE on Training Set', mae_avg_train, color='r')
                    plt.bar('Average MAE on Test Set', mae_avg_test, color='g')   
                    plt.ylabel('MAE')
                    plt.xlabel('Training and Test Datasets')
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
                if btn_mse:
                    st.write('Mean squared error on Training Set: ', mse_train)
                    st.write('Mean squared error on Test Set: ', mse_test)
                    fig_mse = plt.figure() 
                    plt.bar('MSE on Training Set', mse_train, color='r')
                    plt.bar('MSE on Test Set', mse_test, color='g')   
                    plt.ylabel('MSE')
                    plt.xlabel('Train and Test Datasets')
                    plt.title('Mean squared error')
                    st.pyplot(fig_mse)
                if btn_mae:
                    st.write('Mean absolute error on Training Set: ', mae_train)
                    st.write('Mean absolute error on Test Set: ', mae_test)
                    fig_mae = plt.figure() 
                    plt.bar('MAE on Training Set', mae_train, color='r')
                    plt.bar('MAE on Test Set', mae_test, color='g')   
                    plt.ylabel('MAE')
                    plt.xlabel('Training and Test Datasets')
                    plt.title('Mean absolute error')
                    st.pyplot(fig_mae)
        else:
            st.write('No feature selected!')