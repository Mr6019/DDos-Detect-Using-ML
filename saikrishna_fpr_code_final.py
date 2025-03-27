# Attaching Libraries

import warnings
warnings.filterwarnings("ignore")
import numpy as np,pandas as pd,seaborn,datetime,os
import matplotlib.pyplot as vsap
import matplotlib as mpvs
import plotly.express as pxids

from sklearn.pipeline import make_pipeline
from sklearn import preprocessing, utils,metrics,pipeline,feature_selection, model_selection, decomposition
from sklearn import tree, linear_model,svm, ensemble, neural_network

import visualkeras
from PIL import ImageFont
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.utils import to_categorical
import os

## Data Reading

def ReadAPA():
    apa=pd.read_csv("APA-DDoS-Dataset.csv")
    print("Total Records of Data: {}".format(apa.shape[0]))
    print("Total Features of Data: {}".format(apa.shape[1]))
    return apa

APADT=ReadAPA()
APADT.head()

## Data Cleaning

def CleanAPA(apa):
    print(apa.info())
    print("Missing Values (Before Cleaning): ",sum(apa.isna().sum()))
    if sum(apa.isna().sum())>0:
        apa=apa.dropna()
    print("Missing Values (After Cleaning): ",sum(apa.isna().sum()))
    return apa
APADT=CleanAPA(APADT)

## Data Visualization

pd.crosstab(APADT['ip.src'],APADT['Label']).plot(kind='barh',
                                                       title="Network Traffic Types by Source IP", figsize=(8,3))
vsap.grid()
vsap.show()

apaft=['frame.len','Packets','Bytes']
apanmft=['Frame Length','Packets','Bytes Transacted']
for c in range(len(apaft)):
    APADT.groupby('Label').mean('{}'.format(apaft[c]))['{}'.format(apaft[c])].plot(
        kind='bar',figsize=(5,3), color=['g','c','m'], title="{} for Network Traffics".format(apanmft[c]))
    vsap.ylabel('{}'.format(apaft[c]))
    vsap.xlabel('Traffic Type')
    vsap.grid()
    vsap.show()

## Feature Engineering

### Feature Encoding

def EncodeAttr(dt):
    apa=dt.copy()
    apa=apa.drop(['frame.time','Label'],axis=1)
    lbapa=dt['Label']
    apa_cts=apa.dtypes[apa.dtypes=='object'].index.tolist()
    print("Detected Object Type Features: \n", *apa_cts, sep="\n")
    if len(apa_cts)==0:
        apa['Label']=lbapa
        return apa
    else:
        for c in range(len(apa_cts)):
            apa[apa_cts[c]]=apa[apa_cts[c]].replace(apa[apa_cts[c]].unique(),[i+1 for i in range(len(apa[apa_cts[c]].unique()))])
        apa['Label']=lbapa
        return apa
Enc_APADT=EncodeAttr(APADT)
Enc_APADT.head()

### Outlier Treatement

def OutlierChecking(apa,n,col,tx):
    arrapa=np.array(apa)
    pcapa = decomposition.PCA(n_components=n)
    pcapa.fit(arrapa)
    pcapacm=["Comp-{}".format(i+1) for i in range(len(pcapa.explained_variance_ratio_.tolist()))]
    vsap.figure(figsize=(4,2))
    vsap.title("{} Normalization\nMaximum Variance: {}".format(tx,round(max(pcapa.explained_variance_ratio_),5)),fontsize=18)
    vsap.bar(pcapacm,pcapa.explained_variance_ratio_.tolist(),width=0.5,color=col)
    vsap.xlabel("PCA",fontsize=14)
    vsap.ylabel("Variance",fontsize=14)
    vsap.grid()
    vsap.show()
    return pcapa.explained_variance_ratio_

def DataScaling(apa):
    ppc = preprocessing.MinMaxScaler()
    nrmapa=ppc.fit_transform(apa)
    return nrmapa

outapa=[]
Enc_APADT = Enc_APADT.replace([np.inf, -np.inf], np.finfo('float32').max)
outapa.append(OutlierChecking(Enc_APADT.drop('Label',axis=1),2,"#FF4500","Before"))

fledt=[]
apapca_flag=[]
for rp in outapa:
    for r in rp:
        if r>0.7:
            apapca_flag.append(True)
if len(apapca_flag)==1 and True in apapca_flag:
    APA_Norm=DataScaling(Enc_APADT.drop('Label',axis=1))
APA_Norm=pd.DataFrame(APA_Norm,columns=Enc_APADT.drop('Label',axis=1).columns.tolist())
APA_Norm['Label']=Enc_APADT['Label']

OutlierChecking(APA_Norm.drop('Label',axis=1),2,"#12E193","After")
APA_Norm.head()

### Hybrid Feature Selection By Combining RFE and Chi-Sq

def FetRFE(apa):
    Xapa=apa.drop([apa.columns.tolist()[-1]],axis=1)
    Yapa=apa[apa.columns.tolist()[-1]]
    Yapa=Yapa.replace(Yapa.unique(),[x for x in range(len(Yapa.unique()))])
    rfe_apa = feature_selection.RFE(estimator=linear_model.LogisticRegression(),n_features_to_select = int(len(apa.columns)*0.7), step = 0.7)
    rfe_trnd=rfe_apa.fit(Xapa,Yapa)
    print("RFE Selected Features: \n",*Xapa.columns[rfe_trnd.get_support()],sep="\n")
    return Xapa.columns[rfe_trnd.get_support()]

def FetChisq(apa):
    Xapa=apa.drop([apa.columns.tolist()[-1]],axis=1)
    Yapa=apa[apa.columns.tolist()[-1]]
    Yapa=Yapa.replace(Yapa.unique(),[x for x in range(len(Yapa.unique()))])
    chi_apa=feature_selection.SelectKBest(feature_selection.chi2, k=int(len(apa.columns)*0.7))
    chi_apa.fit(Xapa, Yapa)
    print("Chi-Squared Selected Features: \n",*Xapa.columns[chi_apa.get_support()],sep="\n")
    return Xapa.columns[chi_apa.get_support()]

ApaFets=[]
ftc=FetChisq(APA_Norm)
ftr=FetRFE(APA_Norm)
for x in ftc:
    if x in ftr:
        ApaFets.append(x)

print("\nSelected Features Using Hybrid Technique: \n")
print(*ApaFets, sep="\n")

## Data Preparation

### Preparing Predictor and Target Data


Predictor=APA_Norm[ApaFets]
Predictor.head()

Target=APA_Norm['Label']
print(Target.value_counts())

### Data Split

def SegData(PredDt, TgrDt):
    TrnDDSX,TstDDSX,TrnDDSy,TstDDSy=model_selection.train_test_split(PredDt,TgrDt, test_size=0.25, random_state=0)
    print("Test Class Distribution: ",TstDDSy.value_counts(),"\n")
    print("Training Class Distribution: ",TrnDDSy.value_counts())
    return TrnDDSX,TstDDSX,TrnDDSy,TstDDSy
TrnDDSX,TstDDSX,TrnDDSy,TstDDSy=SegData(Predictor, Target)

## Assigning Algorithms

ClfDDS=[
        ensemble.RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=11,min_weight_fraction_leaf=0.45,max_features='log2'),
        linear_model.LogisticRegression(tol=0.06, C=0.01,max_iter=2,solver='liblinear',fit_intercept=False),
        neural_network.MLPClassifier(hidden_layer_sizes=(2,1,), learning_rate_init=0.001, power_t=0.7,max_fun=3,max_iter=4),
        make_pipeline(preprocessing.StandardScaler(),svm.SVC(C=0.2, kernel='sigmoid', degree=3,tol=0.01,max_iter=20))
]
Clfs=[
      "Random Forest",
      "Logistic Regression",
      "MLP Classifier",
      "SVC"
]

ClfDDS[0]

ClfDDS[1]

ClfDDS[2]

ClfDDS[3]

## DDoS Detection

def ConfMatVs(yAct,yPreds,ModelDDS):
    CLSS=np.unique(np.array(yAct))
    ids_cnf=pd.crosstab(yAct,yPreds,rownames=['True'], colnames=['Predicted'], margins=True)
    vsap.figure(figsize=(5,3))
    vsap.title("{}".format(ModelDDS), fontsize=16,color="m")
    seaborn.heatmap(ids_cnf.iloc[:len(CLSS),:len(CLSS)],fmt="d",annot=True,cmap="plasma")
    vsap.show()
    return ids_cnf

DDSResultData=[[],[],[],[],[],[]]

for i in range(len(ClfDDS)):
    print("                         {} ".format(Clfs[i]))
    PrsClf = ClfDDS[i]
    StartTime_1 = datetime.datetime.now()
    PrsClf.fit(TrnDDSX, TrnDDSy)
    StartTime_2 = datetime.datetime.now()
    TimeDifference = StartTime_2 - StartTime_1
    TimeTrnSec=TimeDifference.total_seconds()
    print("Training Time: {} Seconds".format(round(TimeTrnSec,3)))
    StartTime_3 = datetime.datetime.now()
    DDSPredTst=PrsClf.predict(TstDDSX)
    StartTime_4 = datetime.datetime.now()
    TimeDifference = StartTime_4 - StartTime_3
    TimeTstSec=TimeDifference.total_seconds()
    print("Training Time: {} Seconds".format(round(TimeTstSec,3)))
    AccDDSTst=metrics.accuracy_score(TstDDSy,DDSPredTst)
    print("Train Accuracy: ",round(PrsClf.score(TrnDDSX,TrnDDSy),5)*100)
    print("Test Accuracy: ",round(PrsClf.score(TstDDSX,TstDDSy),5)*100)
    DDSResultData[0].append(round(AccDDSTst,5)*100)
    DDSResultData[1].append(round(metrics.precision_score(TstDDSy,DDSPredTst,average="weighted"),5)*100)
    DDSResultData[2].append(round(metrics.recall_score(TstDDSy,DDSPredTst,average="weighted"),5)*100)
    DDSResultData[3].append(round(metrics.f1_score(TstDDSy,DDSPredTst,average="weighted"),5)*100)
    DDSResultData[4].append(round(TimeTstSec,3))
    DDSResultData[5].append(round(PrsClf.score(TrnDDSX,TrnDDSy),5)*100)
    ConfMatVs(TstDDSy,DDSPredTst,Clfs[i])
    print("_________________ Classification report for __________________")
    print("_________________ {} ______________".format(Clfs[i]))
    print(metrics.classification_report(TstDDSy,DDSPredTst))
DDoSDf=pd.DataFrame({
    "Models":Clfs,
    "Train_Accuracy":DDSResultData[5],
    "Test_Accuracy":DDSResultData[0],
    "Precision":DDSResultData[1],
    "Recall":DDSResultData[2],
    "F1-Score":DDSResultData[3],
    "Prediction_Time":DDSResultData[4]
})
DDoSDf=DDoSDf.sort_values(by="Test_Accuracy",ascending=False)
DDoSDf


for i in DDoSDf.columns.tolist()[2:]:
    figres = pxids.bar(DDoSDf, y=i, x="Models",
             text=i,color="Models",title="Comparison of {}".format(i),height=400,width=600)
    figres.show()

## CNN

def ResultVisualizer(ffanhs,md,strt,en,gprhknd,yax,eps):
    reshs=ffanhs.history
    fithst=pd.DataFrame({
        "Iteration":[i+1 for i in range(eps)],
        "Loss_Train":reshs['loss'],
        "Loss_Valid":reshs['val_loss'],
        "Accuracy_Train":reshs['accuracy'],
        "Accuracy_Valid":reshs['val_accuracy']
    })
    clscl=["#FF00FF","#0002FF"]
    fithst.iloc[:,strt:en].plot(kind=gprhknd,figsize=(6,3),color=clscl)
    vsap.title("{} Comparison ({})".format(yax,md),fontsize=17,color="m")
    vsap.xlabel("Epochs",fontsize=15,color="m")
    vsap.ylabel("{}".format(yax),fontsize=15,color="m")
    vsap.grid()
    vsap.show()

TrUn=TrnDDSy.unique()
TstUn=TstDDSy.unique()

print(TrnDDSy.unique())
print(TstDDSy.unique())
clss=TrnDDSy.unique().tolist()
TrnDDSyUN=TrnDDSy.replace(TrnDDSy.unique(),[x for x in range(len(TrnDDSy.unique()))])
TstDDSyUN=TstDDSy.replace(TstDDSy.unique(),[2,0,1])
print(TrnDDSyUN.unique())
print(TstDDSyUN.unique())

scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(TrnDDSX)
x_test = scaler.transform(TstDDSX)

# Reshape data to fit into the Conv1D layer
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
DOSCNN = Sequential([
    Conv1D(1024, 1, activation='relu', input_shape=(14, 1)),
    MaxPooling1D(2),
    Conv1D(512, 1, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='sigmoid')
])

DOSCNN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
DOSCNN.summary()
TrnTimeStrt = datetime.datetime.now()
DOSCNN_Hist=DOSCNN.fit(x_train,TrnDDSyUN, epochs=10, validation_data=(x_test,TstDDSyUN))
TrnTimeEnd = datetime.datetime.now()
TrnTm = TrnTimeEnd - TrnTimeStrt
TrnTimeSecCNN=TrnTm.total_seconds()

ResultVisualizer(DOSCNN_Hist,"CNN",3,5,'line','Accuracy',10)
ResultVisualizer(DOSCNN_Hist,"CNN",1,3,'line','Loss',10)

pred_train=DOSCNN.predict(x_train)
trnpred = pred_train.argmax(axis=1)
dftrn=pd.DataFrame({
        "Actual":TrnDDSyUN,
        "Predicted":trnpred
})
acctrcnn=metrics.accuracy_score(TrnDDSyUN,trnpred)
AcrFinCNNTr=round(acctrcnn,4)*100
print("Training Accuracy for CNN: {}%".format(AcrFinCNNTr))

TstTimeStrt = datetime.datetime.now()
pred_test=DOSCNN.predict(x_test)
TstTimeEnd = datetime.datetime.now()
PrdTime = TstTimeEnd - TstTimeStrt
TstTimeSecCNN=PrdTime.total_seconds()
tstpred = pred_test.argmax(axis=1)
dftest=pd.DataFrame({
        "Actual":TstDDSyUN,
        "Predicted":tstpred
})
cnfcnn=pd.crosstab(TstDDSyUN,tstpred,rownames=['Actual'], colnames=['Detected'], margins=True)
cnfcnncls=cnfcnn.iloc[:3,:3]
print("--------\n{}\n-----------".format(cnfcnncls))
vsap.figure(figsize=(6,4))
vsap.title("Confusion Matrix: {}".format("CNN"), fontsize=16,color="m")
seaborn.heatmap(cnfcnncls,fmt="d",annot=True,cmap="flag",xticklabels=clss, yticklabels=clss)
vsap.show()

acr_cnn_test=metrics.accuracy_score(TstDDSyUN,tstpred)
prec_cnn_test=metrics.precision_score(TstDDSyUN,tstpred,average='weighted')
recl_cnn_test=metrics.recall_score(TstDDSyUN,tstpred,average='weighted')
f1s_cnn_test=metrics.f1_score(TstDDSyUN,tstpred,average='weighted')
AcrFinCNN=round(acr_cnn_test,4)*100
PrcFinCNN=round(prec_cnn_test,4)*100
RclFinCNN=round(recl_cnn_test,4)*100
FsFinCNN=round(f1s_cnn_test,4)*100
print("Accuracy: {}%\nPrecision: {}%\nRecall: {}%\nF1-Score: {}%".format(AcrFinCNN,PrcFinCNN,RclFinCNN,FsFinCNN))

tensorflow.keras.utils.plot_model(
    DOSCNN,
    to_file="CNNModel.png",
    rankdir="TB",
    dpi=65
)

DDoSDfFinal=pd.DataFrame({
    "Models":Clfs+['CNN'],
    "Train_Accuracy":DDSResultData[5]+[AcrFinCNNTr],
    "Test_Accuracy":DDSResultData[0]+[AcrFinCNN],
    "Precision":DDSResultData[1]+[PrcFinCNN],
    "Recall":DDSResultData[2]+[RclFinCNN],
    "F1-Score":DDSResultData[3]+[FsFinCNN],
    "Prediction_Time":DDSResultData[4]+[TstTimeSecCNN]
})
DDoSDfFinal=DDoSDfFinal.sort_values(by='Test_Accuracy',ascending=False).reset_index(drop=True)
DDoSDfFinal

bc_colrs=["#66CDAA","#FFA600","#1F45FC","#FFCBA4","#F67280","#6A0DAD"]
for rh in DDoSDfFinal.columns.tolist()[2:]:
    DDoSDfFinal=DDoSDfFinal.sort_values(by=rh,ascending=True)
    vsap.figure(figsize=(6,4))
    vsap.title("Comparison of {}".format(rh),fontsize=18,color="#8B008B")
    vsap.barh(DDoSDfFinal['Models'],DDoSDfFinal[rh],color=bc_colrs)
    vsap.ylabel("Models",fontsize=16,color="#8B008B")
    vsap.xlabel("{}".format(rh),fontsize=16,color="#8B008B")
    for sr, val in enumerate(DDoSDfFinal["{}".format(rh)]):
        vsap.text(val, sr, str(val))
    vsap.grid()
    vsap.show()

