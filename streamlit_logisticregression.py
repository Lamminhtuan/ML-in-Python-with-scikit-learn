import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import log_loss
features = []
#Function to check all features checkboxes
def check():
    for i in range(len(df.columns[:-1])):
        st.session_state[str(i)] = True
    return
#Function to uncheck all features checkboxes
def uncheck():
    for i in range(len(df.columns[:-1])):
        st.session_state[str(i)] = False
    return
st.markdown('**Lâm Minh Tuấn - 20520843 - CS116.N11 - Logistic Regression**')
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
    #Standarize the data
    if needstandarize:
        ct = StandardScaler()
        ct.fit(X_train[needstandarize])
        X_train[needstandarize] = ct.transform(X_train[needstandarize])
        X_test[needstandarize] = ct.transform(X_test[needstandarize])
    usekfold = st.checkbox("KFold: ")
    if usekfold:
        left, right = st.columns(2)
        with left:
            st.write('##')
            st.write('Enter k for KFold cross-validation: ')
        with right:
            k = st.number_input('', min_value =2, format="%d")
    col_1, col_2, col_3, col_4 = st.columns(4)
    with col_1:
        btn_pre = st.checkbox('Precision')
    with col_2:
        btn_re = st.checkbox('Recall')
    with col_3:
        btn_f1 = st.checkbox('F1')
    with col_4:
        btn_log = st.checkbox('Log Loss')
    if st.button("Run"):
        if usekfold:
            pre_train_list = []
            pre_test_list = []
            re_train_list = []
            re_test_list = []
            f1_train_list = []
            f1_test_list = []
            log_train_list = []
            log_test_list = []
            kf = KFold(k)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                reg = LogisticRegression()
                reg.fit(X_train, y_train)
                y_pred_train = reg.predict(X_train)
                y_pred_test = reg.predict(X_test)
                pre_train_list.append(precision_score(y_train, y_pred_train))
                pre_test_list.append(precision_score(y_test, y_pred_test))
                re_train_list.append(recall_score(y_train, y_pred_train))
                re_test_list.append(recall_score(y_test, y_pred_test))
                f1_train_list.append(f1_score(y_train, y_pred_train))
                f1_test_list.append(f1_score(y_test, y_pred_test))
                log_train_list.append(log_loss(y_train, y_pred_train))
                log_test_list.append(log_loss(y_test, y_pred_test))
            pre_avg_train = sum(pre_train_list) / len(pre_train_list)
            pre_avg_test = sum(pre_test_list) / len(pre_test_list)
            re_avg_train = sum(re_train_list) / len(re_train_list)
            re_avg_test = sum(re_test_list) / len(re_test_list)
            f1_avg_train = sum(f1_train_list) / len(f1_train_list)
            f1_avg_test = sum(f1_test_list) / len(f1_test_list)
            log_avg_train = sum(log_train_list) / len(log_train_list)
            log_avg_test = sum(log_test_list) / len(log_test_list)
            n = np.arange(len(log_test_list))
            if btn_pre:
                fig_pre = plt.figure()
                plt.bar(n - 0.2, pre_train_list, color='r', width=0.4, label="Precision on Train")
                plt.bar(n + 0.2, pre_test_list, color='g', width=0.4, label="Precision on Test")
                plt.ylabel('Precision')
                plt.xlabel('Folds')
                plt.title('Precision of Folds')
                plt.xticks(n)
                plt.legend()
                st.pyplot(fig_pre)
                fig_pre_avg = plt.figure()
                plt.bar('Average Precision on Train', pre_avg_train, color='r')
                plt.bar('Average Precision on Test', pre_avg_test, color='g')   
                plt.ylabel('Precision')
                plt.xlabel('Train and test datasets')
                plt.title('Average Precision of Folds')
                st.pyplot(fig_pre_avg)
            if btn_re:
                fig_re = plt.figure()
                plt.bar(n - 0.2, re_train_list, color='r', width=0.4, label="Recall on Train")
                plt.bar(n + 0.2, re_test_list, color='g', width=0.4, label="Recall on Test")
                plt.ylabel('Recall')
                plt.xlabel('Folds')
                plt.title('Recall of Folds')
                plt.xticks(n)
                plt.legend()
                st.pyplot(fig_re)
                fig_re_avg = plt.figure()
                plt.bar('Average Recall on Train', re_avg_train, color='r')
                plt.bar('Average Recall on Test', re_avg_test, color='g')   
                plt.ylabel('Recall')
                plt.xlabel('Train and test datasets')
                plt.title('Average Recall of Folds')
                st.pyplot(fig_re_avg)
            if btn_f1:
                fig_f1 = plt.figure()
                plt.bar(n - 0.2, f1_train_list, color='r', width=0.4, label="F1 Score on Train")
                plt.bar(n + 0.2, f1_test_list, color='g', width=0.4, label="F1 Score on Test")
                plt.ylabel('F1 Score')
                plt.xlabel('Folds')
                plt.title('F1 Score of Folds')
                plt.xticks(n)
                plt.legend()
                st.pyplot(fig_f1)
                fig_f1_avg = plt.figure()
                plt.bar('Average F1 Score on Train', f1_avg_train, color='r')
                plt.bar('Average F1 Score on Test', f1_avg_test, color='g')   
                plt.ylabel('F1 Score')
                plt.xlabel('Train and test datasets')
                plt.title('Average F1 Score of Folds')
                st.pyplot(fig_f1_avg)
            if btn_log:
                fig_log = plt.figure()
                plt.bar(n - 0.2, f1_train_list, color='r', width=0.4, label="Log Loss on Train")
                plt.bar(n + 0.2, f1_test_list, color='g', width=0.4, label="Log Loss on Test")
                plt.ylabel('Log Loss')
                plt.xlabel('Folds')
                plt.title('Log Loss of Folds')
                plt.xticks(n)
                plt.legend()
                st.pyplot(fig_log)
                fig_log_avg = plt.figure()
                plt.bar('Average Log Loss on Train', log_avg_train, color='r')
                plt.bar('Average Log Loss on Test', log_avg_test, color='g')   
                plt.ylabel('Log Loss')
                plt.xlabel('Train and test datasets')
                plt.title('Average Log Loss of Folds')
                st.pyplot(fig_log_avg)
        else:
            reg = LogisticRegression()
            reg.fit(X_train, y_train)
            y_pred_train = reg.predict(X_train)
            y_pred_test = reg.predict(X_test)
            pre_train = precision_score(y_train, y_pred_train)
            pre_test = precision_score(y_test, y_pred_test)
            re_train = recall_score(y_train, y_pred_train)
            re_test = recall_score(y_test, y_pred_test)
            f1_train = f1_score(y_train, y_pred_train)
            f1_test = f1_score(y_test, y_pred_test)
            log_train = log_loss(y_train, y_pred_train)
            log_test = log_loss(y_test, y_pred_test)
            if btn_pre:
                st.write('Precision on train dataset: ', pre_train)
                st.write('Precision on test dataset: ', pre_test)
                fig_pre = plt.figure() 
                plt.bar('Precision on Train', pre_train, color='r')
                plt.bar('Precision on Test', pre_test, color='g')   
                plt.ylabel('Precision')
                plt.xlabel('Train and test datasets')
                plt.title('Precision')
                st.pyplot(fig_pre)
            if btn_re:
                st.write('Recall on train dataset: ', re_train)
                st.write('Recall on test dataset: ', re_test)
                fig_re = plt.figure() 
                plt.bar('Recall on Train', re_train, color='r')
                plt.bar('Recall on Test', re_test, color='g')   
                plt.ylabel('Recall')
                plt.xlabel('Train and test datasets')
                plt.title('Recall')
                st.pyplot(fig_re) 
            if btn_f1:
                st.write('F1 score on train dataset: ', f1_train)
                st.write('F1 score on test dataset: ', f1_test)
                fig_f1 = plt.figure() 
                plt.bar('F1 Score on Train', f1_train, color='r')
                plt.bar('F1 Score on Test', f1_test, color='g')   
                plt.ylabel('F1 Score')
                plt.xlabel('Train and test datasets')
                plt.title('F1 Score')
                st.pyplot(fig_f1)
            if btn_log:
                st.write('Log Loss on train dataset: ', log_train)
                st.write('Log Loss on test dataset: ', log_test)
                fig_log = plt.figure() 
                plt.bar('Log Loss on Train', log_train, color='r')
                plt.bar('Log Loss on Test', log_test, color='g')   
                plt.ylabel('Log Loss')
                plt.xlabel('Train and test datasets')
                plt.title('Log Loss')
                st.pyplot(fig_log)