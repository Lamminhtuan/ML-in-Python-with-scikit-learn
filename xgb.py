import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.datasets import load_wine
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import log_loss
import xgboost as xgb
features = []
def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])
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
st.markdown('**Lâm Minh Tuấn - 20520843 - CS116.N11 - XGBoosting**')
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
    labels = np.unique(y)
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
    usekfold = st.checkbox('K-Fold cross validation')
    if usekfold:
        left, right = st.columns(2)
        with left:
            st.write('##')
            st.write('Enter k for K-Fold cross-validation: ')
        with right:
            k = st.number_input('', min_value =2, format="%d")
    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        btn_pre = st.checkbox('Precision')
    with col_2:
        btn_re = st.checkbox('Recall')
    with col_3:
        btn_f1 = st.checkbox('F1')
    
    if st.button("Run"):
        if features:
            kf = KFold(4)
            f1_score_fold_log = []
            f1_score_fold_svm = []
            f1_score_fold_tree = []
            f1_score_fold_xgb = []
            for train_index, test_index in kf.split(X):
                
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                #Standarize the data
                if needstandarize:
                    ct = StandardScaler()
                    ct.fit(X_train[needstandarize])
                    X_train[needstandarize] = ct.transform(X_train[needstandarize])
                    X_test[needstandarize] = ct.transform(X_test[needstandarize])
                
                reg_log = LogisticRegression()
                reg_svm = svm.SVC() 
                reg_tree = DecisionTreeClassifier()
                reg_xgb = xgb.XGBClassifier()
                reg_log.fit(X_train, y_train)
                reg_svm.fit(X_train, y_train)
                reg_tree.fit(X_train, y_train)
                reg_xgb.fit(X_train, y_train)
                y_pred_log = reg_log.predict(X_test)
                y_pred_svm = reg_svm.predict(X_test)
                y_pred_tree = reg_tree.predict(X_test)
                y_pred_xgb = reg_xgb.predict(X_test)
                f1_score_fold_log.append(f1_score(y_test, y_pred_log , average="micro"))
                f1_score_fold_svm.append(f1_score(y_test, y_pred_svm , average="micro"))
                f1_score_fold_tree.append(f1_score(y_test, y_pred_tree , average="micro"))
                f1_score_fold_xgb.append(f1_score(y_test, y_pred_xgb , average="micro"))
            f1_score_avg_plot_log = sum(f1_score_fold_log) / len(f1_score_fold_log)
            f1_score_avg_plot_svm = sum(f1_score_fold_svm) / len(f1_score_fold_svm)
            f1_score_avg_plot_tree = sum(f1_score_fold_tree) / len(f1_score_fold_tree)
            f1_score_avg_plot_xgb = sum(f1_score_fold_xgb) / len(f1_score_fold_xgb)

            fig_max = plt.figure()
            data = {'Logistic Regression':f1_score_avg_plot_log, 'SVM': f1_score_avg_plot_svm, 'Decision Tree': f1_score_avg_plot_tree, 'XGB':f1_score_avg_plot_xgb}
            courses = list(data.keys())
            values = list(data.values())
            addlabels(courses, values)
            plt.bar(courses, values, color = 'maroon', width = 0.4)
            plt.ylabel('F1 Score (Micro)')
            plt.xlabel('Models')
            
            plt.title('F1 Score (Micro) of Logistic Regression, SVM, Decision Tree, XGB (4 Folds)')
            
            st.pyplot(fig_max)
            
            if usekfold:
                pre_log_train_list = []
                pre_log_test_list = []
                re_log_train_list = []
                re_log_test_list = []
                f1_log_train_list = []
                f1_log_test_list = []
                

                pre_svm_train_list = []
                pre_svm_test_list = []
                re_svm_train_list = []
                re_svm_test_list = []
                f1_svm_train_list = []
                f1_svm_test_list = []

                pre_tree_train_list = []
                pre_tree_test_list = []
                re_tree_train_list = []
                re_tree_test_list = []
                f1_tree_train_list = []
                f1_tree_test_list = []

                pre_xgb_train_list = []
                pre_xgb_test_list = []
                re_xgb_train_list = []
                re_xgb_test_list = []
                f1_xgb_train_list = []
                f1_xgb_test_list = []
                kf = KFold(k)
                f1_score_fold_log = []
                f1_score_fold_svm = []
                f1_score_fold_tree = []
                f1_score_fold_xgb = []
                for train_index, test_index in kf.split(X):
                    
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    #Standarize the data
                    if needstandarize:
                        ct = StandardScaler()
                        ct.fit(X_train[needstandarize])
                        X_train[needstandarize] = ct.transform(X_train[needstandarize])
                        X_test[needstandarize] = ct.transform(X_test[needstandarize])
                    
                  
                    reg_log = LogisticRegression()
                    reg_svm = svm.SVC() 
                    reg_tree = DecisionTreeClassifier()
                    reg_xgb = xgb.XGBClassifier()
                    reg_log.fit(X_train, y_train)
                    reg_svm.fit(X_train, y_train)
                    reg_tree.fit(X_train, y_train)
                    reg_xgb.fit(X_train, y_train)
                    y_pred_log_train = reg_log.predict(X_train)
                    y_pred_svm_train = reg_svm.predict(X_train)
                    y_pred_tree_train = reg_tree.predict(X_train)
                    y_pred_xgb_train = reg_xgb.predict(X_train)
                    y_pred_log_test = reg_log.predict(X_test)
                    y_pred_svm_test = reg_svm.predict(X_test)
                    y_pred_tree_test = reg_tree.predict(X_test)
                    y_pred_xgb_test = reg_xgb.predict(X_test)
         
                    pre_log_train_list.append(precision_score(y_train, y_pred_log_train, average="micro"))
                    pre_log_test_list.append(precision_score(y_test, y_pred_log_test, average="micro"))
                    re_log_train_list.append(recall_score(y_train, y_pred_log_train, average="micro"))
                    re_log_test_list.append(recall_score(y_test, y_pred_log_test, average="micro"))
                    f1_log_train_list.append(f1_score(y_train, y_pred_log_train, average="micro"))
                    f1_log_test_list.append(f1_score(y_test, y_pred_log_test, average="micro"))

                    pre_svm_train_list.append(precision_score(y_train, y_pred_svm_train, average="micro"))
                    pre_svm_test_list.append(precision_score(y_test, y_pred_svm_test, average="micro"))
                    re_svm_train_list.append(recall_score(y_train, y_pred_svm_train, average="micro"))
                    re_svm_test_list.append(recall_score(y_test, y_pred_svm_test, average="micro"))
                    f1_svm_train_list.append(f1_score(y_train, y_pred_svm_train, average="micro"))
                    f1_svm_test_list.append(f1_score(y_test, y_pred_svm_test, average="micro"))

                    pre_tree_train_list.append(precision_score(y_train, y_pred_tree_train, average="micro"))
                    pre_tree_test_list.append(precision_score(y_test, y_pred_tree_test, average="micro"))
                    re_tree_train_list.append(recall_score(y_train, y_pred_tree_train, average="micro"))
                    re_tree_test_list.append(recall_score(y_test, y_pred_tree_test, average="micro"))
                    f1_tree_train_list.append(f1_score(y_train, y_pred_tree_train, average="micro"))
                    f1_tree_test_list.append(f1_score(y_test, y_pred_tree_test, average="micro"))

                    pre_xgb_train_list.append(precision_score(y_train, y_pred_xgb_train, average="micro"))
                    pre_xgb_test_list.append(precision_score(y_test, y_pred_xgb_test, average="micro"))
                    re_xgb_train_list.append(recall_score(y_train, y_pred_xgb_train, average="micro"))
                    re_xgb_test_list.append(recall_score(y_test, y_pred_xgb_test, average="micro"))
                    f1_xgb_train_list.append(f1_score(y_train, y_pred_xgb_train, average="micro"))
                    f1_xgb_test_list.append(f1_score(y_test, y_pred_xgb_test, average="micro"))
                tongsoluong = len(pre_log_train_list)
                
                pre_log_avg_test = sum(pre_log_test_list) / tongsoluong
                
                re_log_avg_test = sum(re_log_test_list) / tongsoluong
                
                f1_log_avg_test = sum(f1_log_test_list) / tongsoluong

                pre_svm_avg_test = sum(pre_svm_test_list) / tongsoluong
                re_svm_avg_test = sum(re_svm_test_list) / tongsoluong
                f1_svm_avg_test = sum(f1_svm_test_list) / tongsoluong

                pre_tree_avg_test = sum(pre_tree_test_list) / tongsoluong
                re_tree_avg_test = sum(re_tree_test_list) / tongsoluong
                f1_tree_avg_test = sum(f1_tree_test_list) / tongsoluong

                
                pre_xgb_avg_test = sum(pre_xgb_test_list) / tongsoluong
                re_xgb_avg_test = sum(re_xgb_test_list) / tongsoluong
                f1_xgb_avg_test = sum(f1_xgb_test_list) / tongsoluong
                n = np.arange(tongsoluong)
             
                if btn_pre:
                    
                    fig = plt.figure()
                    data = {'Logistic Regression':pre_log_avg_test, 'SVM': pre_svm_avg_test, 'Decision Tree': pre_tree_avg_test, 'XGB':pre_xgb_avg_test}
                    courses = list(data.keys())
                    values = list(data.values())
                    addlabels(courses, values)
                    plt.bar(courses, values, color = 'maroon', width = 0.4)
                    plt.ylabel('Precision (Micro)')
                    plt.xlabel('Models')
                    
                    plt.title('Average Precision (Micro) of Logistic Regression, SVM, Decision Tree, XGB of Folds')
                    
                    st.pyplot(fig)
                if btn_re:
                    fig = plt.figure()
                    data = {'Logistic Regression':re_log_avg_test, 'SVM': re_svm_avg_test, 'Decision Tree': re_tree_avg_test, 'XGB':re_xgb_avg_test}
                    courses = list(data.keys())
                    values = list(data.values())
                    addlabels(courses, values)
                    plt.bar(courses, values, color = 'maroon', width = 0.4)
                    plt.ylabel('Recall (Micro)')
                    plt.xlabel('Models')
                    
                    plt.title('Average Recall (Micro) of Logistic Regression, SVM, Decision Tree, XGB of Folds')
                    
                    st.pyplot(fig)
                if btn_f1:
                    fig = plt.figure()
                    data = {'Logistic Regression':f1_log_avg_test, 'SVM': f1_svm_avg_test, 'Decision Tree': f1_tree_avg_test, 'XGB':f1_xgb_avg_test}
                    courses = list(data.keys())
                    values = list(data.values())
                    addlabels(courses, values)
                    plt.bar(courses, values, color = 'maroon', width = 0.4)
                    plt.ylabel('F1 Score (Micro)')
                    plt.xlabel('Models')
                    
                    plt.title('Average F1 Score (Micro) of Logistic Regression, SVM, Decision Tree, XGB of Folds')
                    
                    st.pyplot(fig)
            else:
                
                reg_log = LogisticRegression()
                reg_svm = svm.SVC() 
                reg_tree = DecisionTreeClassifier()
                reg_xgb = xgb.XGBClassifier()
                reg_log.fit(X_train, y_train)
                reg_svm.fit(X_train, y_train)
                reg_tree.fit(X_train, y_train)
                reg_xgb.fit(X_train, y_train)
                y_pred_log = reg_log.predict(X_test)
                y_pred_svm = reg_svm.predict(X_test)
                y_pred_tree = reg_tree.predict(X_test)
                y_pred_xgb = reg_xgb.predict(X_test)
                pre_log_test = precision_score(y_test, y_pred_log, average="micro")
                
                re_log_test = recall_score(y_test, y_pred_log, average="micro")
                
                f1_log_test = f1_score(y_test, y_pred_log, average="micro")
                pre_svm_test = precision_score(y_test, y_pred_svm, average="micro")
                
                re_svm_test = recall_score(y_test, y_pred_svm, average="micro")
                
                f1_svm_test = f1_score(y_test, y_pred_svm, average="micro")
                pre_tree_test = precision_score(y_test, y_pred_tree, average="micro")
                
                re_tree_test = recall_score(y_test, y_pred_tree, average="micro")
                
                f1_tree_test = f1_score(y_test, y_pred_tree, average="micro")
                pre_xgb_test = precision_score(y_test, y_pred_xgb, average="micro")
                
                re_xgb_test = recall_score(y_test, y_pred_xgb, average="micro")
                
                f1_xgb_test = f1_score(y_test, y_pred_xgb, average="micro")
       
                if btn_pre:
                    fig = plt.figure()
                    data = {'Logistic Regression':pre_log_test, 'SVM': pre_svm_test, 'Decision Tree': pre_tree_test, 'XGB':pre_xgb_test}
                    courses = list(data.keys())
                    values = list(data.values())
                    addlabels(courses, values)
                    plt.bar(courses, values, color = 'maroon', width = 0.4)
                    plt.ylabel('Precision (Micro)')
                    plt.xlabel('Models')
                    
                    plt.title('Precision (Micro) of Logistic Regression, SVM, Decision Tree, XGB')
                    
                    st.pyplot(fig)
                if btn_re:
                    fig = plt.figure()
                    data = {'Logistic Regression':re_log_test, 'SVM': re_svm_test, 'Decision Tree': re_tree_test, 'XGB':re_xgb_test}
                    courses = list(data.keys())
                    values = list(data.values())
                    addlabels(courses, values)
                    plt.bar(courses, values, color = 'maroon', width = 0.4)
                    plt.ylabel('Recall (Micro)')
                    plt.xlabel('Models')
                    
                    plt.title('Recall (Micro) of Logistic Regression, SVM, Decision Tree, XGB')
                    
                    st.pyplot(fig)
                if btn_f1:
                    fig = plt.figure()
                    data = {'Logistic Regression':f1_log_test, 'SVM': f1_svm_test, 'Decision Tree': f1_tree_test, 'XGB':f1_xgb_test}
                    courses = list(data.keys())
                    values = list(data.values())
                    addlabels(courses, values)
                    plt.bar(courses, values, color = 'maroon', width = 0.4)
                    plt.ylabel('F1 Score (Micro)')
                    plt.xlabel('Models')
                    
                    plt.title('F1 Score (Micro) of Logistic Regression, SVM, Decision Tree, XGB')
                    
                    st.pyplot(fig)

        else:
            st.write('No feature selected!')