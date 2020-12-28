

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

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
#esta libreria es nueva
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.utils import resample
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

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

param_XGB = {
	'max_depth' : [2,4,6,8,10,12,18],
	'gamma': [1,0.1,0.01,2,6],
}

accuracy_RF=[]
precision_RF=[]
recall_RF=[]
f1_score_RF=[]
y_real=[]
preds_RF=[]
prs_RF = []
aucs_RF = []
mean_recall_RF = np.linspace(0, 1, 100)

accuracy_NB=[]
precision_NB=[]
recall_NB=[]
f1_score_NB=[]
preds_NB=[]
prs_NB = []
aucs_NB = []
mean_recall_NB = np.linspace(0, 1, 100)

accuracy_SVM=[]
precision_SVM=[]
recall_SVM=[]
f1_score_SVM=[]
preds_SVM=[]
prs_SVM = []
aucs_SVM = []
mean_recall_SVM = np.linspace(0, 1, 100)

accuracy_GBC=[]
precision_GBC=[]
recall_GBC=[]
f1_score_GBC=[]
preds_GBC=[]
prs_GBC = []
aucs_GBC = []
mean_recall_GBC = np.linspace(0, 1, 100)

accuracy_LR=[]
precision_LR=[]
recall_LR=[]
f1_score_LR=[]
preds_LR=[]
prs_LR = []
aucs_LR = []
mean_recall_LR = np.linspace(0, 1, 100)

accuracy_DTC=[]
precision_DTC=[]
recall_DTC=[]
f1_score_DTC=[]
preds_DTC=[]
prs_DTC = []
aucs_DTC = []
mean_recall_DTC = np.linspace(0, 1, 100)


for i in range(6):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=i, stratify=y)
    #df = pd.concat([x_train,Y_train], axis=1)
    #df_max = df[df.clase==0]
    #df_min = df[df.clase==1]
    #df_min_balance= resample(df_min,replace=True,n_samples=35,random_state=0)
    #df= pd.concat([df_max,df_min_balance])
    #X_train = df.drop(['clase'], axis=1)
    #y_train = df['clase']
    if (i > 1):
     RF = RandomForestClassifier(class_weight='balanced',random_state=0)
     CV_RF = GridSearchCV(estimator=RF, param_grid=param_RF, cv= 5)
     CV_RF.fit(X_train,y_train)
     modeloRF=CV_RF.best_estimator_
     print("Mejores Parametros Experimento Random Forest",i)
     print(CV_RF.best_params_)
     y_pred=modeloRF.predict(X_test)
     probs = modeloRF.predict_proba(X_test)
     #y_real.append(y_test)
     preds_RF =probs[:,1]
     precision_RF,recall_RF,thresholds=precision_recall_curve(y_test,preds_RF)
     prs_RF.append(interp(mean_recall_RF, precision_RF, recall_RF))
     pr_auc_RF = auc(recall_RF, precision_RF)
     aucs_RF.append(pr_auc_RF)
     
     NB = GaussianNB()
     NB = NB.fit(X_train,y_train)
     y_pred=NB.predict(X_test)
     probs = NB.predict_proba(X_test)
     preds_NB = probs[:,1]
     precision_NB, recall_NB, _ =precision_recall_curve(y_test,preds_NB)
     prs_NB.append(interp(mean_recall_NB, precision_NB, recall_NB))
     pr_auc_NB = auc(recall_NB, precision_NB)
     aucs_NB.append(pr_auc_NB)
     
     LR = LogisticRegression(class_weight='balanced',random_state=0)
     CV_LR = GridSearchCV(estimator=LR, param_grid=param_LR, cv= 5)
     CV_LR.fit(X_train,y_train)
     modeloLR=CV_LR.best_estimator_
     print("Mejores Parametros Experimento Random Forest",i)
     print(CV_LR.best_params_)
     y_pred=modeloLR.predict(X_test)
     probs = modeloLR.predict_proba(X_test)
     preds_LR = probs[:,1]
     precision_LR, recall_LR, _ =precision_recall_curve(y_test,preds_LR)
     prs_LR.append(interp(mean_recall_LR, precision_LR, recall_LR))
     pr_auc_LR = auc(recall_LR, precision_LR)
     aucs_LR.append(pr_auc_LR)
     

     svm = SVC(class_weight='balanced',probability=True) 
     CV_svm = GridSearchCV(estimator=svm, param_grid=param_SVM, cv= 5)
     CV_svm.fit(X_train,y_train)
     modelosvm=CV_svm.best_estimator_
     print("Mejores Parametros Experimento SVM",i)
     print(CV_svm.best_params_)
     y_pred=modelosvm.predict(X_test)
     probs = modelosvm.predict_proba(X_test)
     preds_SVM = probs[:,1]
     precision_SVM, recall_SVM, _ =precision_recall_curve(y_test,preds_SVM)
     prs_SVM.append(interp(mean_recall_SVM, precision_SVM, recall_SVM))
     pr_auc_SVM = auc(recall_SVM, precision_SVM)
     aucs_SVM.append(pr_auc_SVM)
     

     GBC = GradientBoostingClassifier(random_state=0)
     CV_GBC = GridSearchCV(estimator=GBC, param_grid=param_GBC, cv= 5)
     CV_GBC.fit(X_train,y_train)
     modeloGBC=CV_GBC.best_estimator_
     print("Mejores Parametros Experimento GBC",i)
     print(CV_GBC.best_params_)
     y_pred=modeloGBC.predict(X_test)
     probs = modeloGBC.predict_proba(X_test)
     preds_GBC = probs[:,1]
     precision_GBC, recall_GBC, _ =precision_recall_curve(y_test,preds_GBC)
     prs_GBC.append(interp(mean_recall_GBC, precision_GBC, recall_GBC))
     pr_auc_GBC = auc(recall_GBC, precision_GBC)
     aucs_GBC.append(pr_auc_GBC)
     
    
     DTC = DecisionTreeClassifier(class_weight='balanced',random_state=0)
     CV_DTC = GridSearchCV(estimator=DTC, param_grid=param_DTC, cv= 5)
     CV_DTC.fit(X_train,y_train)
     modeloDTC=CV_DTC.best_estimator_
     print("Mejores Parametros Experimento DTC",i)
     print(CV_DTC.best_params_)
     y_pred=modeloDTC.predict(X_test)
     probs = modeloDTC.predict_proba(X_test)
     preds_DTC= probs[:,1]
     precision_DTC, recall_DTC, _ =precision_recall_curve(y_test,preds_DTC)
     prs_DTC.append(interp(mean_recall_DTC, precision_DTC, recall_DTC))
     pr_auc_DTC = auc(recall_DTC, precision_DTC)
     aucs_DTC.append(pr_auc_DTC)
     
     

mean_precision_RF = np.mean(prs_RF, axis=0)
mean_auc_RF = auc(mean_recall_RF, mean_precision_RF)
std_auc_RF = np.std(aucs_RF)

mean_precision_NB = np.mean(prs_NB, axis=0)
mean_auc_NB = auc(mean_recall_NB, mean_precision_RF)
std_auc_NB = np.std(aucs_NB)

mean_precision_LR = np.mean(prs_LR, axis=0)
mean_auc_LR = auc(mean_recall_LR, mean_precision_RF)
std_auc_LR = np.std(aucs_LR)

mean_precision_SVM = np.mean(prs_SVM, axis=0)
mean_auc_SVM = auc(mean_recall_SVM, mean_precision_RF)
std_auc_SVM = np.std(aucs_SVM)

mean_precision_GBC = np.mean(prs_GBC, axis=0)
mean_auc_GBC = auc(mean_recall_GBC, mean_precision_GBC)
std_auc_GBC = np.std(aucs_GBC)

mean_precision_DTC = np.mean(prs_DTC, axis=0)
mean_auc_DTC = auc(mean_recall_DTC, mean_precision_DTC)
std_auc_DTC = np.std(aucs_DTC)

plt.plot(mean_precision_RF, mean_recall_RF,'b', label = 'Random Forest,area = %0.2f $\pm$ %0.2f'% (mean_auc_RF, std_auc_RF),lw=1,color='red')
plt.plot(mean_precision_NB, mean_recall_NB,'b', label = 'Naive Bayes, area = %0.2f $\pm$ %0.2f' % (mean_auc_NB, std_auc_NB) ,lw=1,color='yellow')
plt.plot(mean_precision_SVM, mean_recall_SVM,'b', label = 'SVM, area= %0.2f $\pm$ %0.2f' % (mean_auc_SVM, std_auc_SVM),lw=1,color='blue')
plt.plot(mean_precision_GBC, mean_recall_GBC,'b', label = 'GBM, area = %0.2f $\pm$ %0.2f' % (mean_auc_GBC, std_auc_GBC),lw=1,color='purple')
plt.plot(mean_precision_LR, mean_recall_LR,'b', label = 'Regresion Logistica,area = %0.2f $\pm$ %0.2f' % (mean_auc_LR, std_auc_LR),lw=1,color='green')
plt.plot(mean_precision_DTC, mean_recall_DTC,'b', label = 'DTC,area = %0.2f $\pm$ %0.2f' % (mean_auc_DTC, std_auc_DTC),lw=1,color='pink')


no_skill = len(y[y==1]) / len(y)
#no_skill = 0.5
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill',lw=1,color='black')
#plt.legend(loc='center')
plt.legend(loc='lower left')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim(0.5,1)
plt.title('Precision-Recall Curves')
plt.savefig('PS_M5_prueba_todo_promedio.png')
