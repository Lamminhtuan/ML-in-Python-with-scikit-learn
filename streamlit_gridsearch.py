import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import GridSearchCV
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
st.markdown('**Lâm Minh Tuấn - 20520843 - CS116.N11 - GridSearch**')
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
            if usekfold:
                pre_test_list_gs = []
                pre_test_list = []
              
                re_test_list = []
                re_test_list_gs = []
                f1_test_list = []
                f1_test_list_gs = []
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
                    
                  
                    parameters = {'C': [0.1, 1, 10, 100, 1000],
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                    'kernel': ['rbf']}
                    svc = svm.SVC() 
                    
                    clf = GridSearchCV(svc, parameters, refit = True, verbose = 3)
                    clf.fit(X_train, y_train)
                
                    svc.fit(X_train, y_train)
                    
                    y_pred_test = svc.predict(X_test)
                
                    
                    y_pred_test_gs = clf.predict(X_test)
         
                    
                    pre_test_list.append(precision_score(y_test, y_pred_test, average="micro"))
                    pre_test_list_gs.append(precision_score(y_test, y_pred_test_gs, average="micro"))
                    re_test_list.append(recall_score(y_test, y_pred_test, average="micro"))
                    re_test_list_gs.append(recall_score(y_test, y_pred_test_gs, average="micro"))
                    f1_test_list.append(f1_score(y_test, y_pred_test, average="micro"))
                    f1_test_list_gs.append(f1_score(y_test, y_pred_test_gs, average="micro"))
                tongsoluong = len(pre_test_list_gs)
                pre_avg_test = sum(pre_test_list) / tongsoluong
                pre_avg_test_gs = sum(pre_test_list_gs) / tongsoluong
                re_avg_test = sum(re_test_list) / tongsoluong
                re_avg_test_gs = sum(re_test_list_gs) / tongsoluong
                f1_avg_test = sum(f1_test_list) / tongsoluong
                f1_avg_test_gs = sum(f1_test_list_gs) / tongsoluong
                n = np.arange(tongsoluong)
             
                if btn_pre:
                    
                    fig_pre_avg = plt.figure()
                    
                    data = {'Without GridSearchCV':pre_avg_test, 'With GridSearchCV': pre_avg_test_gs}
                    courses = list(data.keys())
                    values = list(data.values())
                    addlabels(courses, values)
                    plt.bar(courses, values, color = 'maroon', width = 0.8) 
                    plt.ylabel('Precision (Micro)')
                    plt.xlabel('With and without GridSearchCV')
                    plt.title('Average Precision (Micro) of Folds')
                    st.pyplot(fig_pre_avg)
                if btn_re:
                    fig_re_avg = plt.figure()
                    data = {'Without GridSearchCV':re_avg_test, 'With GridSearchCV': re_avg_test_gs}
                    courses = list(data.keys())
                    values = list(data.values())
                    addlabels(courses, values)
                    plt.bar(courses, values, color = 'maroon', width = 0.8)  
                    plt.ylabel('Recall (Micro)')
                    plt.xlabel('With and without GridSearchCV')
                    plt.title('Average Recall (Micro) of Folds')
                    st.pyplot(fig_re_avg)
                if btn_f1:
                    fig_f1_avg = plt.figure()
                    data = {'Without GridSearchCV':f1_avg_test, 'With GridSearchCV': f1_avg_test_gs}
                    courses = list(data.keys())
                    values = list(data.values())
                    addlabels(courses, values)
                    plt.bar(courses, values, color = 'maroon', width = 0.8)  
                    plt.ylabel('F1 Score (Micro)')
                    plt.xlabel('With and without GridSearchCV')
                    plt.title('Average F1 Score (Micro) of Folds')
                    st.pyplot(fig_f1_avg)
       
            else:
                parameters = {'C': [0.1, 1, 10, 100, 1000],
                 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}
                svc = svm.SVC() 
                
                clf = GridSearchCV(svc, parameters, refit = True, verbose = 3)
                clf.fit(X_train, y_train)
             
                svc.fit(X_train, y_train)
                
                y_pred_test = svc.predict(X_test)
               
                
                y_pred_test_gs = clf.predict(X_test)
                if btn_pre:
                    
                    pre_test = precision_score(y_test, y_pred_test, average="micro")
                    
                    pre_test_gs = precision_score(y_test, y_pred_test_gs, average="micro")
                    st.write('Precision (Micro) on Test Set without GridSearchCV: ', pre_test)
                    st.write('Precision (Micro) on Test Set with GridSearchCV: ', pre_test_gs)
                    fig_pre = plt.figure() 
                    data = {'Without GridSearchCV':pre_test, 'With GridSearchCV': pre_test_gs}
                    courses = list(data.keys())
                    values = list(data.values())
                    addlabels(courses, values)
                    plt.bar(courses, values, color = 'maroon', width = 0.8) 
                    
                    plt.ylabel('Precision (Micro)')
                    plt.xlabel('With and without GridSearchCV')
                    plt.title('Precision (Micro)')
                    st.pyplot(fig_pre)
                if btn_re:
                    
                    re_test = recall_score(y_test, y_pred_test, average="micro")
                    
                    re_test_gs = recall_score(y_test, y_pred_test_gs, average="micro")
                    st.write('Recall (Micro) on Test Set without GridSearchCV: ', re_test)
                    st.write('Recall (Micro) on Test Set with GridSearchCV: ', re_test_gs)
                    fig_re = plt.figure() 
                    data = {'Without GridSearchCV':re_test, 'With GridSearchCV': re_test_gs}
                    courses = list(data.keys())
                    values = list(data.values())
                    addlabels(courses, values)  
                    plt.bar(courses, values, color = 'maroon', width = 0.8) 
                    plt.ylabel('Recall (Micro)')
                    plt.xlabel('With and without GridSearchCV')
                    plt.title('Recall (Micro)')
                    st.pyplot(fig_re)
                if btn_f1:
                    
                    f1_test = f1_score(y_test, y_pred_test, average="micro")
                    
                    f1_test_gs = f1_score(y_test, y_pred_test_gs, average="micro")
                    st.write('F1 Score (Micro) on Test Set without GridSearchCV: ', f1_test)
                    st.write('F1 Score (Micro) on Test Set with GridSearchCV: ', f1_test_gs)
                    fig_f1 = plt.figure() 
                    data = {'Without GridSearchCV':f1_test, 'With GridSearchCV': f1_test_gs}
                    courses = list(data.keys())
                    values = list(data.values())
                    addlabels(courses, values)   
                    plt.bar(courses, values, color = 'maroon', width = 0.8) 
                    plt.ylabel('F1 Score (Micro)')
                    plt.xlabel('With and without GridSearchCV')
                    plt.title('F1 Score (Micro)')
                    st.pyplot(fig_f1)

        else:
            st.write('No feature selected!')