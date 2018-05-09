import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
import array
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import cross_val_score


dfo = pd.read_csv('./data/xaxis/train.csv')
dfo_test = pd.read_csv('./data/xaxis/test.csv')
print(dfo.columns)

#print(dfo.describe())
dfo.groupby('response').size()


#Look for duplicated rows and drop them
print(dfo.columns)

#Some websites are NaN so dropping these
#df.dropna(inplace=True)
dfo['domain'].fillna('Unknown',inplace=True)
df = dfo

#Get Some info about df
Nsig=len(df[df['response']==1].index)
Nbkg=len(df[df['response']==0].index)
ratio=Nsig/(Nbkg)
print(ratio)

for name in df.columns:
    print(name,len(df[name].unique()))

#Function to create dictionary between domain and weekly-views for that domain
def getdict(data):
    data.loc[data['user_hour']>0,'user_hour']=1

    grouped = data.groupby(['domain','user_day'],as_index=False).agg({'user_hour':np.sum})
    grouped = grouped.groupby(['domain'],as_index=False).agg({'user_hour':np.sum})
    grouped['weekviews'] = grouped['user_hour']
    
    mydict=grouped.set_index(['domain']).to_dict()['weekviews']
    return mydict

def dummies(data,name):
 
    myarr_o = data[name].unique()
    myarr = []
    for i in range(len(myarr_o)): 
        myarr.append(name+"-"+str(myarr_o[i]))
    
    print(myarr)
    data[myarr]=pd.get_dummies(data[name])
    data.drop(name,axis=1,inplace=True)
    data = data.drop(myarr[0],axis=1)
    print('Dropped Column:',myarr[0], 'from:',name)
    
    return(data)


#Perform various steps to transform the dat
def replace(data,mydict):
    #Create a new category domctr (Domain CTR)
    data['domctr']=0.0
    data['domctr'] = data['domain'].replace(mydict)

    data.loc[(data.user_day >=0) & (data.user_day<5),'user_day']=0
    data.loc[(data.user_day >5),'user_day']=1
    return(data)
    
def scale(data):
    
    X_scaled= preprocessing.scale(data['os_extended'])
    data['os_extended']=X_scaled
    return(data)


print(df['user_hour'].unique())
dict = getdict(df)
df = replace(df,dict)

cat_vars=['carrier','size','device_type','position','region','supply_type']
dropcols=['publisher','placement','browser','language','domain','device_model']
for i in range(0,len(cat_vars)):
    df = dummies(df,cat_vars[i])

for i in range(0,len(dropcols)):
    df.drop(dropcols[i],axis=1,inplace=True)   

#Check if anything wierd happened during dictionary translation
print(df['domctr'].isnull().any().any())
df['domctr'].dropna(inplace=True)
df['domctr']=np.float32(df['domctr'])
df.columns

dict1 = getdict(dfo_test)
dfo_test = replace(dfo_test,dict1)
for i in range(0,len(cat_vars)):
    dfo_test = dummies(dfo_test,cat_vars[i])

for i in range(0,len(dropcols)):
    dfo_test.drop(dropcols[i],axis=1,inplace=True)   


dfo_test.dropna(inplace=True)
#dfo_test.isnull().any()
print(dfo_test.isnull().any().any())
dfo_test.dropna(inplace=True)
dfo_test['domctr']=np.float32(dfo_test['domctr'])

#Created my own test Sample
dfx = df
Y = dfx['response']
X = dfx.drop('response',1)

df_train, df_test, dfy_train, y_test = train_test_split(X,Y, test_size=0.20, random_state=0)

#Create the features data set
T=dfy_train
F=df_train

# A simple logistic Regression
ind = len(F.columns)
print(ind)
F.insert(ind,'intercept',1)
logit_mod = sm.Logit(T, F[F.columns])
results = logit_mod.fit()
results.summary()


F = F.drop('intercept',1)



#Decision Tree Classifieer
y = T
X = F



#X.replace([np.inf, -np.inf], np.nan)
#print(np.array(X['domctr'][1:500]))
#DecisionTreeClaissifer
dt = DecisionTreeClassifier(min_samples_split=5, random_state=99)
dt.fit(X, y)
scores = cross_val_score(dt, X, y).mean()
print(scores)



#Logistic Regression
from sklearn.pipeline import make_pipeline
rt = RandomTreesEmbedding(max_depth=5, n_estimators=10,
    random_state=0)

rt_lm = LogisticRegression()
pipeline = make_pipeline(rt, rt_lm)
pipeline.fit(X,y)
scores = cross_val_score(rt_lm, X, y).mean()
print(scores)




#Random rf Classifier
rf = RandomForestClassifier(min_samples_split=3)
rf.fit(X,y)
scores = cross_val_score(rf, X, y).mean()
print(scores)

print(X.columns)
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the rf
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


X_test=df_test
#Feature Importance
features=X_test.columns
print(features)
print(dt.feature_importances_)


#Predict and get the confusion Matrix
X_test = df_test
preds = dt.predict(X_test, check_input=True)
print(confusion_matrix(y_test, preds))

#RandomForest
preds3 = rf.predict(X_test)
print(confusion_matrix(y_test,preds3))

#Logistcregression
preds4 = pipeline.predict(X_test)
print(confusion_matrix(y_test,preds4))



#Plot ROC curve and Precision-Recall
from sklearn.metrics import roc_curve

y_pred_rf = rf.predict_proba(X_test)[:, 1]
y_pred_lg = pipeline.predict_proba(X_test)[:, 1]
y_pred_dt = dt.predict_proba(X_test)[:, 1]

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
fpr_lg, tpr_lg, _ = roc_curve(y_test, y_pred_lg)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt)

# The random forest model by itself
plt.figure(1)
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.plot(fpr_lg, tpr_lg, label='LR')
plt.plot(fpr_dt, tpr_dt, label='DT')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

plt.figure(2)
plt.xlim(0.1, 0.3)
plt.ylim(0.6, 0.9)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.plot(fpr_lg, tpr_lg, label='LR')
plt.plot(fpr_dt, tpr_dt, label='DT')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()

y_score=y_pred_rf
average_precision = average_precision_score(y_test, y_score)
precision, recall, _ = precision_recall_curve(y_test, y_score)
plt.step(recall, precision, color='b', alpha=0.2,where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()



print('Precision:',precision_score(y_test, preds3))
print('Recall:',recall_score(y_test, preds3))
print('Accuracy:',accuracy_score(y_test,preds3))



def write(data):
    final = data
    final['prob']=0.0
    final['prob']=y_pred_axis
    final.head()
    final.to_csv('test_prob.csv',index=False)
#final = pd.concat([dfo_test,yarr],axis=1)



#plt.hist(y_pred_grd)
X_test2 = dfo_test
y_pred_axis = rf.predict_proba(X_test2)[:,1]



plt.hist(y_pred_axis,alpha=0.3,normed=1)
plt.hist(y_pred_rf,color='r',alpha=0.2,normed=1)
plt.xlabel('Predicted Response')
plt.title('Predicted Response R:My Test Sample, B: Axis Test Sample')
plt.show()



write(dfo_test)




check = pd.read_csv('./test_prob.csv')
check.head()