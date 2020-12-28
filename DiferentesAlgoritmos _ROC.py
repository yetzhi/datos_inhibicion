
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix
import warnings  
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import seaborn as sns

warnings.simplefilter('ignore')

df = pd.read_csv("g2-r7_attr_resample.csv")
X = df.drop(['clase'], axis=1)
y = df['clase']

scaler = StandardScaler()  
scaler.fit(X)  
X = scaler.transform(X) 


param_AD = {'max_depth':[2,4,7,10,12,15],
          'max_features' : [2,6,10,12,15]}


#param_AD = {'max_depth':[2,4,7,10,12,15]}

param_SVM = { 'C': [0.1,1, 10, 100, 250], 'gamma': [1,0.1,0.01,0.001], 'kernel': ['rbf', 'poly', 'sigmoid','linear']}

param_LR = { 'C': [0.1,1, 10, 100, 250], 'penalty':['l1','l2']}


param_RF = { 
    'n_estimators': [80,100,150,200],
    'max_features' : [2,6,10,12,15,20]
}

param_GBC = { 
    'learning_rate' :[ 0.01, 0.1, 0.2, 0.3, 0.5, 0.7],
    'n_estimators': [40,80,100,150,200,300,350]
}

param_DTC = {
	'max_features' : [1,2,4,6,8,10,18,20],
	'criterion': ['gini', 'entropy']
}




accuracy_RF=[]
precision_RF=[]
recall_RF=[]

accuracy_NB=[]
precision_NB=[]
recall_NB=[]

accuracy_SVM=[]
precision_SVM=[]
recall_SVM=[]

accuracy_AD=[]
precision_AD=[]
recall_AD=[]

accuracy_GBC=[]
precision_GBC=[]
recall_GBC=[]

accuracy_LR=[]
precision_LR=[]
recall_LR=[]

accuracy_DTC=[]
precision_DTC=[]
recall_DTC=[]


#####***********
fpr_acum_RF=[]
tpr_acum_RF=[]
threshold_acum_RF=[]
roc_auc_acum_RF = []
base_fpr_RF = np.linspace(0, 1, 101)

fpr_acum_NB=[]
tpr_acum_NB=[]
threshold_acum_NB=[]
roc_auc_acum_NB = []
base_fpr_NB = np.linspace(0, 1, 101)

fpr_acum_SVM=[]
tpr_acum_SVM=[]
threshold_acum_SVM=[]
roc_auc_acum_SVM = []
base_fpr_SVM = np.linspace(0, 1, 101)

fpr_acum_LR=[]
tpr_acum_LR=[]
threshold_acum_LR=[]
roc_auc_acum_LR = []
base_fpr_LR = np.linspace(0, 1, 101)

fpr_acum_GBC=[]
tpr_acum_GBC=[]
threshold_acum_GBC=[]
roc_auc_acum_GBC = []
base_fpr_GBC = np.linspace(0, 1, 101)

fpr_acum_DTC=[]
tpr_acum_DTC=[]
threshold_acum_DTC=[]
roc_auc_acum_DTC = []
base_fpr_DTC = np.linspace(0, 1, 101)



for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=i, stratify=y)
    
    RF = RandomForestClassifier(random_state=0)
    CV_RF = GridSearchCV(estimator=RF, param_grid=param_RF, cv= 5)
    CV_RF.fit(X_train,y_train)
    modeloRF=CV_RF.best_estimator_
    print("Mejores Parametros Experimento Random Forest",i)
    print(CV_RF.best_params_)
    y_pred=modeloRF.predict(X_test)

    probs = modeloRF.predict_proba(X_test)
    preds = probs[:,1]
    fpr_RF, tpr_RF, threshold_RF = metrics.roc_curve(y_test, preds)
    roc_auc_RF = metrics.auc(fpr_RF, tpr_RF)
    tpr_RF = interp(base_fpr_RF, fpr_RF, tpr_RF)
    tpr_RF[0] = 0.0
    fpr_acum_RF.append(fpr_RF)
    tpr_acum_RF.append(tpr_RF)
    threshold_acum_RF.append(threshold_RF)
    roc_auc_acum_RF.append(roc_auc_RF)

    accuracy_RF.append(metrics.accuracy_score(y_test,y_pred))
    precision_RF.append(metrics.precision_score(y_test,y_pred,average='macro'))
    recall_RF.append(metrics.recall_score(y_test,y_pred,average='macro'))
    #para imprimir y guardar la mejor matriz
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
    plt.figure(figsize = (10,7))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.savefig('MC_RF_'+str(i)+'.png')
    print(confusion_matrix(y_test,y_pred))
    
    NB = GaussianNB()
    NB = NB.fit(X_train,y_train)
    y_pred=NB.predict(X_test)
    probs = NB.predict_proba(X_test)
    preds = probs[:,1]
    fpr_NB, tpr_NB, threshold_NB = metrics.roc_curve(y_test, preds)
    roc_auc_NB = metrics.auc(fpr_NB, tpr_NB)
    tpr_NB = interp(base_fpr_NB, fpr_NB, tpr_NB)
    tpr_NB[0] = 0.0
    fpr_acum_NB.append(fpr_NB)
    tpr_acum_NB.append(tpr_NB)
    threshold_acum_NB.append(threshold_NB)
    roc_auc_acum_NB.append(roc_auc_NB)
    accuracy_NB.append(metrics.accuracy_score(y_test,y_pred))
    precision_NB.append(metrics.precision_score(y_test,y_pred,average='macro'))
    recall_NB.append(metrics.recall_score(y_test,y_pred,average='macro'))
    #para imprimir y guardar la mejor matriz
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
    plt.figure(figsize = (10,7))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.savefig('MC_NB_'+str(i)+'.png')
    print(confusion_matrix(y_test,y_pred))

    LR = LogisticRegression(random_state=0)
    CV_LR = GridSearchCV(estimator=LR, param_grid=param_LR, cv= 5)
    CV_LR.fit(X_train,y_train)
    modeloLR=CV_LR.best_estimator_
    print("Mejores Parametros Experimento Random Forest",i)
    print(CV_LR.best_params_)
    y_pred=modeloLR.predict(X_test)
    probs = modeloLR.predict_proba(X_test)
    preds = probs[:,1]
    fpr_LR, tpr_LR, threshold_LR = metrics.roc_curve(y_test, preds)
    roc_auc_LR = metrics.auc(fpr_LR, tpr_LR)
    tpr_LR = interp(base_fpr_LR, fpr_LR, tpr_LR)
    tpr_RF[0] = 0.0
    fpr_acum_LR.append(fpr_LR)
    tpr_acum_LR.append(tpr_LR)
    threshold_acum_LR.append(threshold_LR)
    roc_auc_acum_LR.append(roc_auc_LR)
    accuracy_LR.append(metrics.accuracy_score(y_test,y_pred))
    precision_LR.append(metrics.precision_score(y_test,y_pred,average='macro'))
    recall_LR.append(metrics.recall_score(y_test,y_pred,average='macro'))
    #para imprimir y guardar la mejor matriz
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
    plt.figure(figsize = (10,7))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.savefig('MC_LR_'+str(i)+'.png')
    print(confusion_matrix(y_test,y_pred))

    svm = SVC(probability=True) 
    CV_svm = GridSearchCV(estimator=svm, param_grid=param_SVM, cv= 5)
    CV_svm.fit(X_train,y_train)
    modelosvm=CV_svm.best_estimator_
    print("Mejores Parametros Experimento SVM",i)
    print(CV_svm.best_params_)
    y_pred=modelosvm.predict(X_test)
    probs = modelosvm.predict_proba(X_test)
    preds = probs[:,1]
    fpr_SVM, tpr_SVM, threshold_SVM = metrics.roc_curve(y_test, preds)
    roc_auc_SVM = metrics.auc(fpr_SVM, tpr_SVM)
    tpr_SVM = interp(base_fpr_SVM, fpr_SVM, tpr_SVM)
    tpr_SVM[0] = 0.0
    fpr_acum_SVM.append(fpr_SVM)
    tpr_acum_SVM.append(tpr_SVM)
    threshold_acum_SVM.append(threshold_SVM)
    roc_auc_acum_SVM.append(roc_auc_SVM)
    accuracy_SVM.append(metrics.accuracy_score(y_test,y_pred))
    precision_SVM.append(metrics.precision_score(y_test,y_pred,average='macro'))
    recall_SVM.append(metrics.recall_score(y_test,y_pred,average='macro'))
    #para imprimir y guardar la mejor matriz
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
    plt.figure(figsize = (10,7))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.savefig('MC_SVM_'+str(i)+'.png')
    print(confusion_matrix(y_test,y_pred))

    GBC = GradientBoostingClassifier(random_state=0)
    CV_GBC = GridSearchCV(estimator=GBC, param_grid=param_GBC, cv= 5)
    CV_GBC.fit(X_train,y_train)
    modeloGBC=CV_GBC.best_estimator_
    print("Mejores Parametros Experimento GBC",i)
    print(CV_GBC.best_params_)
    y_pred=modeloGBC.predict(X_test)
    probs = modeloGBC.predict_proba(X_test)
    preds = probs[:,1]
    fpr_GBC, tpr_GBC, threshold_GBC = metrics.roc_curve(y_test, preds)
    roc_auc_GBC = metrics.auc(fpr_GBC, tpr_GBC)
    tpr_GBC = interp(base_fpr_GBC, fpr_GBC, tpr_GBC)
    tpr_GBC[0] = 0.0
    fpr_acum_GBC.append(fpr_GBC)
    tpr_acum_GBC.append(tpr_GBC)
    threshold_acum_GBC.append(threshold_GBC)
    roc_auc_acum_GBC.append(roc_auc_GBC)
    accuracy_GBC.append(metrics.accuracy_score(y_test,y_pred))
    precision_GBC.append(metrics.precision_score(y_test,y_pred,average='macro'))
    recall_GBC.append(metrics.recall_score(y_test,y_pred,average='macro'))
    #para imprimir y guardar la mejor matriz
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
    plt.figure(figsize = (10,7))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.savefig('MC_GBG_'+str(i)+'.png')
    print(confusion_matrix(y_test,y_pred))
    
    DTC = DecisionTreeClassifier(class_weight='balanced',random_state=0)
    CV_DTC = GridSearchCV(estimator=DTC, param_grid=param_DTC, cv= 5)
    CV_DTC.fit(X_train,y_train)
    modeloDTC=CV_DTC.best_estimator_
    print("Mejores Parametros Experimento DTC",i)
    print(CV_DTC.best_params_)
    y_pred=modeloDTC.predict(X_test)
    probs = modeloDTC.predict_proba(X_test)
    preds = probs[:,1]
    fpr_DTC, tpr_DTC, threshold_DTC = metrics.roc_curve(y_test, preds)
    roc_auc_DTC = metrics.auc(fpr_DTC, tpr_DTC)
    tpr_DTC = interp(base_fpr_DTC, fpr_DTC, tpr_DTC)
    tpr_DTC[0] = 0.0
    fpr_acum_DTC.append(fpr_DTC)
    tpr_acum_DTC.append(tpr_DTC)
    threshold_acum_DTC.append(threshold_DTC)
    roc_auc_acum_DTC.append(roc_auc_DTC)
    accuracy_DTC.append(metrics.accuracy_score(y_test,y_pred))
    precision_DTC.append(metrics.precision_score(y_test,y_pred,average='binary',pos_label = 1))
    recall_DTC.append(metrics.recall_score(y_test,y_pred,average='binary',pos_label = 1))
    #para imprimir y guardar la mejor matriz
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
    plt.figure(figsize = (10,7))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.savefig('MC_DTC_'+str(i)+'.png')
    print(confusion_matrix(y_test,y_pred))
   
    

  
print("Promedio Accuracy Random Forest :",accuracy_RF)
print("Promedio Precision Random Forest :",precision_RF)
print("Promedio Recall Random Forest:",recall_RF)
print("************************")
print("Promedio Accuracy LR :",accuracy_LR)
print("Promedio Precision LR :",precision_LR)
print("Promedio Recall LR :",recall_LR)
print("************************")
print("Promedio Accuracy NB :",accuracy_NB)
print("Promedio Precision NB :",precision_NB)
print("Promedio Recall NB :",recall_NB)
print("************************")
print("Promedio Accuracy SVM:",accuracy_SVM)
print("Promedio Precision SVM :",precision_SVM)
print("Promedio Recall SVM:",recall_SVM)
print("************************")
print("Promedio Accuracy GBC :",accuracy_GBC)
print("Promedio Precision GBC :",precision_GBC)
print("Promedio Recall GBC:",recall_GBC)
print("************************")
print("Promedio Accuracy GBC :",accuracy_DTC)
print("Promedio Precision GBC :",precision_DTC)
print("Promedio Recall GBC:",recall_DTC)


tpr_acum_RF = np.array(tpr_acum_RF)
tpr_acum_NB = np.array(tpr_acum_NB)
tpr_acum_SVM = np.array(tpr_acum_SVM)
tpr_acum_LR = np.array(tpr_acum_LR)
tpr_acum_GBC = np.array(tpr_acum_GBC)
tpr_acum_DTC = np.array(tpr_acum_DTC)

mean_tprs_RF = tpr_acum_RF.mean(axis=0)
mean_tprs_NB = tpr_acum_NB.mean(axis=0)
mean_tprs_SVM = tpr_acum_SVM.mean(axis=0)
mean_tprs_LR = tpr_acum_LR.mean(axis=0)
mean_tprs_GBC = tpr_acum_GBC.mean(axis=0)
mean_tprs_DTC = tpr_acum_DTC.mean(axis=0)


mean_auc_RF = metrics.auc(base_fpr_RF, mean_tprs_RF)
mean_auc_NB = metrics.auc(base_fpr_NB, mean_tprs_NB)
mean_auc_SVM = metrics.auc(base_fpr_SVM, mean_tprs_SVM)
mean_auc_LR = metrics.auc(base_fpr_LR, mean_tprs_LR)
mean_auc_GBC = metrics.auc(base_fpr_GBC, mean_tprs_GBC)
mean_auc_DTC = metrics.auc(base_fpr_DTC, mean_tprs_DTC)


desv_RF=np.std(roc_auc_acum_RF)
desv_NB=np.std(roc_auc_acum_NB)
desv_SVM=np.std(roc_auc_acum_SVM)
desv_LR=np.std(roc_auc_acum_LR)
desv_GBC=np.std(roc_auc_acum_GBC)
desv_DTC=np.std(roc_auc_acum_DTC)

plt.figure(figsize=(12,8))
plt.rcParams['font.size'] = 14

plt.plot(base_fpr_RF,mean_tprs_RF,label=r'Mean ROC_Random Forest (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_RF,desv_RF),lw=2, alpha=.8,color='red')
plt.plot(base_fpr_NB,mean_tprs_NB,label=r'Mean ROC_Naive Bayes (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_NB,desv_NB),lw=2, alpha=.8,color='yellow')
plt.plot(base_fpr_LR,mean_tprs_LR,label=r'Mean ROC_Regresion Logistica (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_LR,desv_LR),lw=2, alpha=.8,color='green')
plt.plot(base_fpr_SVM,mean_tprs_SVM,label=r'Mean ROC_SVM (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_SVM,desv_SVM),lw=2, alpha=.8,color='blue')
plt.plot(base_fpr_GBC,mean_tprs_GBC,label=r'Mean ROC_GBM (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_GBC,desv_GBC ),lw=2, alpha=.8,color='purple')
plt.plot(base_fpr_DTC,mean_tprs_DTC,label=r'Mean ROC_DTC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_DTC,desv_DTC ),lw=2, alpha=.8,color='pink')
plt.plot([0, 1], [0, 1], color='navy',linestyle='--',label = 'Chance')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.savefig('ROC_prueba.png')
#plt.show()
 


#https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py

        


