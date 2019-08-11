
import numpy as np
import pandas as pd
import random
#from data import Data
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier


###-----construct a workable data set-----###
def import_x_data(p):
    train_rs = []
    #train_label = []
    path = p
    with open(path, 'r+') as f:
        for l in f:
            if l.strip() == "":
                continue
            vec = [0 for _ in range(219)]
            tokens = l.split(' ')
            #labels = tokens[0]
            #train_label.append(labels)
            for pair in tokens[1:]:
                t = pair.split(':')
                idx = int(t[0])
                value = int(t[1])
                vec[idx-1]=value
            train_rs.append(vec)
    x = np.asarray(train_rs)
    #y = np.asarray(train_label)
    return x


x_train0 = import_x_data("./training00.data")
x_train1 = import_x_data("./training01.data")
x_train2 = import_x_data("./training02.data")
x_train3 = import_x_data("./training03.data")
x_train4 = import_x_data("./training04.data")

print(x_train0)

#-----create train and testing set X-----#
train_set1 = np.append(x_train0, x_train1, axis = 0)
train_set2 = np.append(train_set1, x_train2, axis = 0)
train_set0 = np.append(train_set2, x_train3, axis = 0)
train_set3 = np.append(train_set1, x_train3, axis = 0)
train_set4 = np.append(x_train0, x_train2, axis = 0)
train_set5 = np.append(train_set4, x_train3, axis = 0)
train_set6 = np.append(x_train1, x_train2, axis = 0)
train_set7 = np.append(train_set6, x_train3, axis = 0)
#print(train_set1)

xtrain_k = np.append(train_set0, x_train4, axis = 0)
xfold4 = np.append(train_set2, x_train3, axis = 0)
xfold3 = np.append(train_set2, x_train4, axis = 0)
xfold2 = np.append(train_set3, x_train4, axis = 0)
xfold1 = np.append(train_set5, x_train4, axis = 0)
xfold0 = np.append(train_set7, x_train4, axis = 0)

print(xfold4.shape)


#-----extract labels from the data-----#
def import_y_data(p):
    train_label = []
    path = p
    with open(path, 'r+') as f:
        for l in f:
            if l.strip() == "":
                continue
            vec = [0 for _ in range(17)]
            tokens = l.split(' ')
            labels = int(tokens[0])
            train_label.append(labels)
    y = np.asarray(train_label)
    return y


y_train0 = import_y_data("./training00.data")
y_train1 = import_y_data("./training01.data")
y_train2 = import_y_data("./training02.data")
y_train3 = import_y_data("./training03.data")
y_train4 = import_y_data("./training04.data")


#-----create training and testing set Y-----#
train1 = np.append(y_train0, y_train1, axis = 0)
train2 = np.append(train1, y_train2, axis = 0)
train0 = np.append(train2, y_train3, axis = 0)
train3 = np.append(train1, y_train3, axis = 0)
train4 = np.append(y_train0, y_train2, axis = 0)
train5 = np.append(train4, y_train3, axis = 0)
train6 = np.append(y_train1, y_train2, axis = 0)
train7 = np.append(train6, y_train3, axis = 0)

yfold4 = np.append(train2, y_train3, axis = 0)
yfold3 = np.append(train2, y_train4, axis = 0)
yfold2 = np.append(train3, y_train4, axis = 0)
yfold1 = np.append(train5, y_train4, axis = 0)
yfold0 = np.append(train7, y_train4, axis = 0)
ytrain_k = np.append(train0, y_train4, axis = 0)

#print(ytrain_k.shape)
print(yfold4.shape)


###----SUPPORT VECTOR MACHINE-----###


            
def fitt(X,Y,gamma,c):
    epoch = 20
    w = np.ones(219)
    b = 1
    for t in range(1, epoch):
        #I need to shuffle data
        gt = gamma/(1+epoch)
        for i, x in enumerate(X):
            if (Y[i]*np.dot(X[i], w)+b) <= 1:
                w = ((1-gt)*w)+(gt*(c*(Y[i]*X[i])))*100000
                #print(w[0:3])
            else:
                w = (1-gt)*w
                #print(w[0:5])
    return w

def predict(X,Y,w):
    tp = 0
    fp = 0
    fn = 0
    b = 1
    #print(len(X))
    for i, x in enumerate(X):
        classification = np.sign(np.dot(X[i],w)+b)
        if classification > 0 and Y[i] > 0:
            tp += 1
        elif classification > 0 and Y[i] < 0:
            fp += 1
        if classification < 0 and Y[i] > 0:
            fn += 1
        else:
            pass
            
    
    return tp,fp,fn

#-----test for F1 statistic-----#

def F1(TP,FP,FN):
    if TP+FP == 0:
        p = 1
    else:
        p = TP/(TP+FP)
    if TP+FN == 0:
        r = 1
    else:
        r = TP/(TP+FN)
    f1 = 2*(p*r)/(p+r)
    return f1

def p(TP,FP):
    if TP+FP ==0:
        p = 1
    else:
        p = TP/(TP+FP)
    return p

def r(TP,FN):
    if TP+FN == 0:
        r = 1
    else:
        r = TP/(TP+FN)
    return r

#-----average f1 score with: gamma=0.1, c=0.01-----#

t4 = fitt(xfold4,yfold4,0.1,0.01)
p_tst4 = predict(x_train4,y_train4,t4)
p_tra4 = predict(xfold4,yfold4,t4)
tf4 = F1(p_tst4[0],p_tst4[1],p_tst4[2])
trf4 = F1(p_tra4[0],p_tra4[1],p_tra4[2])
print(t4)
print(p_tst4)
print(p_tra4)
print(tf4)
print(trf4)
t3 = fitt(xfold3,yfold3,0.1,0.01)
p_tst3 = predict(x_train3,y_train3,t3)
p_tra3 = predict(xfold3,yfold3,t3)
tf3 = F1(p_tst3[0],p_tst3[1],p_tst3[2])
trf3 = F1(p_tra3[0],p_tra3[1],p_tra3[0])
print(t3)
print(p_tst3)
print(p_tra3)
print(tf3)
print(trf3)
t2 = fitt(xfold2,yfold2,0.1,0.01)
p_tst2 = predict(x_train2,y_train2,t2)
p_tra2 = predict(xfold2,yfold2,t2)
tf2 = F1(p_tst2[0],p_tst2[1],p_tst2[2])
trf2 = F1(p_tra2[0],p_tra2[1],p_tra2[2])
print(t2)
print(p_tst2)
print(p_tra2)
print(tf2)
print(trf2)
t1 = fitt(xfold1,yfold1,0.1,0.01)
p_tst1 = predict(x_train1,y_train1,t1)
p_tra1 = predict(xfold1,yfold1,t1)
tf1 = F1(p_tst1[0],p_tst1[1],p_tst1[2])
trf1 = F1(p_tra1[0],p_tra1[1],p_tra1[2])
print(t1)
print(p_tst1)
print(p_tra1)
print(tf1)
print(trf1)
t0 = fitt(xfold0,yfold0,0.1,0.01)
p_tst0 = predict(x_train0,y_train0,t0)
p_tra0 = predict(xfold0,yfold0,t0)
tf0 = F1(p_tst0[0],p_tst0[1],p_tst0[2])
trf0 = F1(p_tra0[0],p_tra0[1],p_tra0[2])
print(t0)
print(p_tst0)
print(p_tra0)
print(tf0)
print(trf0)
print('TESTavgf1(gamma=0.1,c=0.01):',(tf4+tf3+tf2+tf1+tf0)/5)  #avg TEST F1#
print('TRAINavgf1(gamma=0.1,c=0.01):',(trf4+trf3+trf2+trf1+trf0)/5) #avg TRAIN F1#
### ###

#-----average f1 score with:: gamma=0.01, c=1-----#

t4 = fitt(xfold4,yfold4,0.01,1)
p_tst4 = predict(x_train4,y_train4,t4)
p_tra4 = predict(xfold4,yfold4,t4)
tf4 = F1(p_tst4[0],p_tst4[1],p_tst4[2])
trf4 = F1(p_tra4[0],p_tra4[1],p_tra4[2])

t3 = fitt(xfold3,yfold3,0.01,1)
p_tst3 = predict(x_train3,y_train3,t3)
p_tra3 = predict(xfold3,yfold3,t3)
tf3 = F1(p_tst3[0],p_tst3[1],p_tst3[2])
trf3 = F1(p_tra3[0],p_tra3[1],p_tra3[0])

t2 = fitt(xfold2,yfold2,0.01,1)
p_tst2 = predict(x_train2,y_train2,t2)
p_tra2 = predict(xfold2,yfold2,t2)
tf2 = F1(p_tst2[0],p_tst2[1],p_tst2[2])
trf2 = F1(p_tra2[0],p_tra2[1],p_tra2[2])

t1 = fitt(xfold1,yfold1,0.01,1)
p_tst1 = predict(x_train1,y_train1,t1)
p_tra1 = predict(xfold1,yfold1,t1)
tf1 = F1(p_tst1[0],p_tst1[1],p_tst1[2])
trf1 = F1(p_tra1[0],p_tra1[1],p_tra1[2])

t0 = fitt(xfold0,yfold0,0.01,1)
p_tst0 = predict(x_train0,y_train0,t0)
p_tra0 = predict(xfold0,yfold0,t0)
tf0 = F1(p_tst0[0],p_tst0[1],p_tst0[2])
trf0 = F1(p_tra0[0],p_tra0[1],p_tra0[2])
print('TESTavgf1(gamma=0.01,c=1):',(tf4+tf3+tf2+tf1+tf0)/5)  #avg TEST F1
print('TRAINavgf1(gamma=0.01,c=1):',(trf4+trf3+trf2+trf1+trf0)/5) #avg TRAIN F1
### ###

#-----average f1 score with: gamma=0.1 c=0.00001-----#

t4 = fitt(xfold4,yfold4,0.1,0.00001)
p_tst4 = predict(x_train4,y_train4,t4)
p_tra4 = predict(xfold4,yfold4,t4)
tf4 = F1(p_tst4[0],p_tst4[1],p_tst4[2])
trf4 = F1(p_tra4[0],p_tra4[1],p_tra4[2])
print(p_tst4)
t3 = fitt(xfold3,yfold3,0.1,0.00001)
p_tst3 = predict(x_train3,y_train3,t3)
p_tra3 = predict(xfold3,yfold3,t3)
tf3 = F1(p_tst3[0],p_tst3[1],p_tst3[2])
trf3 = F1(p_tra3[0],p_tra3[1],p_tra3[0])
print(p_tst3)
t2 = fitt(xfold2,yfold2,0.1,0.00001)
p_tst2 = predict(x_train2,y_train2,t2)
p_tra2 = predict(xfold2,yfold2,t2)
tf2 = F1(p_tst2[0],p_tst2[1],p_tst2[2])
trf2 = F1(p_tra2[0],p_tra2[1],p_tra2[2])
print(p_tst2)
t1 = fitt(xfold1,yfold1,0.1,0.00001)
p_tst1 = predict(x_train1,y_train1,t1)
p_tra1 = predict(xfold1,yfold1,t1)
tf1 = F1(p_tst1[0],p_tst1[1],p_tst1[2])
trf1 = F1(p_tra1[0],p_tra1[1],p_tra1[2])
print(p_tst1)
t0 = fitt(xfold0,yfold0,0.1,0.00001)
p_tst0 = predict(x_train0,y_train0,t0)
p_tra0 = predict(xfold0,yfold0,t0)
tf0 = F1(p_tst0[0],p_tst0[1],p_tst0[2])
trf0 = F1(p_tra0[0],p_tra0[1],p_tra0[2])
print(p_tst0)
print('TESTavgf1(gamma=0.1,c=0.00001):',(tf4+tf3+tf2+tf1+tf0)/5)  #avg TEST F1
print('TRAINavgf1(gamma=0.1,c=0.00001):',(trf4+trf3+trf2+trf1+trf0)/5) #avg TRAIN F1
### ###

#-----average f1 score with: gamma=0.001 c=10-----#

t4 = fitt(xfold4,yfold4,0.001,10)
p_tst4 = predict(x_train4,y_train4,t4)
p_tra4 = predict(xfold4,yfold4,t4)
tp4 = p(p_tst4[0],p_tst4[1])
tr4 = r(p_tst4[0],p_tst4[2])
trp4 = p(p_tra4[0],p_tra4[1])
trr4 = p(p_tra4[0],p_tra4[2])
tf4 = F1(p_tst4[0],p_tst4[1],p_tst4[2])
trf4 = F1(p_tra4[0],p_tra4[1],p_tra4[2])
print(p_tst4)
t3 = fitt(xfold3,yfold3,0.001,10)
p_tst3 = predict(x_train3,y_train3,t3)
p_tra3 = predict(xfold3,yfold3,t3)
tp3 = p(p_tst3[0],p_tst3[1])
tr3 = r(p_tst3[0],p_tst3[2])
trp3 = p(p_tra3[0],p_tra3[1])
trr3 = p(p_tra3[0],p_tra3[2])
tf3 = F1(p_tst3[0],p_tst3[1],p_tst3[2])
trf3 = F1(p_tra3[0],p_tra3[1],p_tra3[0])

print(p_tst3)
t2 = fitt(xfold2,yfold2,0.001,10)
p_tst2 = predict(x_train2,y_train2,t2)
p_tra2 = predict(xfold2,yfold2,t2)
tp2 = p(p_tst2[0],p_tst2[1])
tr2 = r(p_tst2[0],p_tst2[2])
trp2 = p(p_tra2[0],p_tra2[1])
trr2 = p(p_tra2[0],p_tra2[2])
tf2 = F1(p_tst2[0],p_tst2[1],p_tst2[2])
trf2 = F1(p_tra2[0],p_tra2[1],p_tra2[2])
print(p_tst2)
t1 = fitt(xfold1,yfold1,0.001,10)
p_tst1 = predict(x_train1,y_train1,t1)
p_tra1 = predict(xfold1,yfold1,t1)
tp1 = p(p_tst1[0],p_tst1[1])
tr1 = r(p_tst1[0],p_tst1[2])
trp1 = p(p_tra1[0],p_tra1[1])
trr1 = p(p_tra1[0],p_tra1[2])
tf1 = F1(p_tst1[0],p_tst1[1],p_tst1[2])
trf1 = F1(p_tra1[0],p_tra1[1],p_tra1[2])
print(p_tst1)
t0 = fitt(xfold0,yfold0,0.001,10)
p_tst0 = predict(x_train0,y_train0,t0)
p_tra0 = predict(xfold0,yfold0,t0)
tp0 = p(p_tst0[0],p_tst0[1])
tr0 = r(p_tst0[0],p_tst0[2])
trp0 = p(p_tra0[0],p_tra0[1])
trr0 = p(p_tra0[0],p_tra0[2])
tf0 = F1(p_tst0[0],p_tst0[1],p_tst0[2])
trf0 = F1(p_tra0[0],p_tra0[1],p_tra0[2])
print(p_tst0)
print('TESTavgp(gamma=0.001,c=10:',(tp4+tp3+tp2+tp1+tp0)/5)
print('TESTavgr(gamma=0.001,c=10:',(tr4+tr3+tr2+tr1+tr0)/5)
print('TRAINavgp(gamma=0.001,c=10:',(trp4+trp3+trp2+trp1+trp0)/5)
print('TRAINavgr(gamma=0.001,c=10:',(trr4+trr3+trr2+trr1+trr0)/5)
print('TESTavgf1(gamma=0.001,c=10):',(tf4+tf3+tf2+tf1+tf0)/5)  #avg TEST F1
print('TRAINavgf1(gamma=0.001,c=10):',(trf4+trf3+trf2+trf1+trf0)/5) #avg TRAIN F1


print('### after many iterations my f1 score converged to roughly 36%, I have contrasted my model with sklearn and found a big difference. Sklearn produced an accuracy of 82% I believe this indicates that there is a bug in my code but I have yet to find it. My optimal hyperparameters turned out to be gamma=0.001, c=10 for a 39% F1 avg. accuracy on training set')

#-----report accuracy using sklearn-----#
print('please wait...thinking..') 
model_svm = svm.SVC(C=1, gamma=0.01)
acc = cross_val_score(model_svm, xtrain_k, ytrain_k, cv=5)
print('sklearn acc',np.mean(acc))




###-----LOGISTIC REGRESSION-----###


def logit(X,Y,gamma,sigma,intercept=False):
    w = np.ones(219)
    b = 1
    epoch = 20
    for t in range(1, epoch):
        for i, x in enumerate(X):
            z = np.dot(X[i], w)
            s = 1/(1+np.exp(-z))
            gradient = np.dot(X[i], (s-Y[i]))+2*w/sigma
            #print(gradient)
            w = gamma*gradient
            
    return w

def estimate(X,Y,w):
    tp = 0
    fp = 0
    fn = 0
    #print(len(X))
    for i, x in enumerate(X):
        classification = np.sign(np.dot(X[i],w))
        if classification > 0 and Y[i] > 0:
            tp += 1
        elif classification > 0 and Y[i] < 0:
            fp += 1
        if classification < 0 and Y[i] > 0:
            fn += 1
        else:
            pass
            
    
    return tp,fp,fn

#-----average f1 score with gamma=0.1, sigma=1-----#        
l4 = logit(xfold4,yfold4,0.1,1)
tst_l4 = estimate(xfold4,yfold4,l4)
tra_l4 = estimate(x_train4,y_train4,l4)
tlf4 = F1(tst_l4[0],tst_l4[1],tst_l4[2])
trlf4 = F1(tra_l4[0],tra_l4[1],tra_l4[2])
print(l4)
print(tst_l4)
print(tra_l4)
print(tlf4)
print(trlf4)
l3 = logit(xfold3,yfold3,0.1,1)
tst_l3 = estimate(xfold3,yfold3,l3)
tra_l3 = estimate(x_train3,y_train3,l3)
tlf3 = F1(tst_l3[0],tst_l3[1],tst_l3[2])
trlf3 = F1(tra_l3[0],tra_l3[1],tra_l3[2])
print(l3)
print(tst_l3)
print(tra_l3)
print(tlf3)
print(trlf3)
l2 = logit(xfold2,yfold2,0.1,1)
tst_l2 = estimate(xfold2,yfold2,l2)
tra_l2 = estimate(x_train2,y_train2,l2)
tlf2 = F1(tst_l2[0],tst_l2[1],tst_l2[2])
trlf2 = F1(tra_l2[0],tra_l2[1],tra_l2[2])
print(l2)
print(tst_l2)
print(tra_l2)
print(tlf2)
print(trlf2)
l1 = logit(xfold1,yfold1,0.1,1)
tst_l1 = estimate(xfold1,yfold1,l1)
tra_l1 = estimate(x_train1,y_train1,l1)
tlf1 = F1(tst_l1[0],tst_l1[1],tst_l1[2])
trlf1 = F1(tra_l1[0],tra_l1[1],tra_l1[2])
print(l1)
print(tst_l1)
print(tra_l1)
print(tlf1)
print(trlf1)
l0 = logit(xfold0,yfold0,0.1,1)
tst_l0 = estimate(xfold0,yfold0,l0)
tra_l0 = estimate(x_train0,y_train0,l0)
tlf0 = F1(tst_l0[0],tst_l0[1],tst_l0[2])
trlf0 = F1(tra_l0[0],tra_l0[1],tra_l0[2])
print(l0)
print(tst_l0)
print(tra_l0)
print(tlf0)
print(trlf0)
print('TESTavgf1(gamma=0.1,c=1):',(tlf4+tlf3+tlf2+tlf1+tlf0)/5)  
print('TRAINavgf1(gamma=0.1,c=1):',(trlf4+trlf3+trlf2+trlf1+trlf0)/5)

#-----average f1 score with gamma=0.01, sigma=0.1-----#        
l4 = logit(xfold4,yfold4,0.01,0.1)
tst_l4 = estimate(xfold4,yfold4,l4)
tra_l4 = estimate(x_train4,y_train4,l4)
tlf4 = F1(tst_l4[0],tst_l4[1],tst_l4[2])
trlf4 = F1(tra_l4[0],tra_l4[1],tra_l4[2])

l3 = logit(xfold3,yfold3,0.01,0.1)
tst_l3 = estimate(xfold3,yfold3,l3)
tra_l3 = estimate(x_train3,y_train3,l3)
tlf3 = F1(tst_l3[0],tst_l3[1],tst_l3[2])
trlf3 = F1(tra_l3[0],tra_l3[1],tra_l3[2])

l2 = logit(xfold2,yfold2,0.01,0.1)
tst_l2 = estimate(xfold2,yfold2,l2)
tra_l2 = estimate(x_train2,y_train2,l2)
tlf2 = F1(tst_l2[0],tst_l2[1],tst_l2[2])
trlf2 = F1(tra_l2[0],tra_l2[1],tra_l2[2])

l1 = logit(xfold1,yfold1,0.01,0.1)
tst_l1 = estimate(xfold1,yfold1,l1)
tra_l1 = estimate(x_train1,y_train1,l1)
tlf1 = F1(tst_l1[0],tst_l1[1],tst_l1[2])
trlf1 = F1(tra_l1[0],tra_l1[1],tra_l1[2])

l0 = logit(xfold0,yfold0,0.01,0.1)
tst_l0 = estimate(xfold0,yfold0,l0)
tra_l0 = estimate(x_train0,y_train0,l0)
tlf0 = F1(tst_l0[0],tst_l0[1],tst_l0[2])
trlf0 = F1(tra_l0[0],tra_l0[1],tra_l0[2])
print('TESTavgf1(gamma=0.01,c=0.1):',(tlf4+tlf3+tlf2+tlf1+tlf0)/5)  
print('TRAINavgf1(gamma=0.01,c=0.1):',(trlf4+trlf3+trlf2+trlf1+trlf0)/5)

#-----average f1 score with gamma=0.001, sigma=0.01-----#
l4 = logit(xfold4,yfold4,0.001,0.01)
tst_l4 = estimate(xfold4,yfold4,l4)
tra_l4 = estimate(x_train4,y_train4,l4)
tlf4 = F1(tst_l4[0],tst_l4[1],tst_l4[2])
trlf4 = F1(tra_l4[0],tra_l4[1],tra_l4[2])

l3 = logit(xfold3,yfold3,0.001,0.01)
tst_l3 = estimate(xfold3,yfold3,l3)
tra_l3 = estimate(x_train3,y_train3,l3)
tlf3 = F1(tst_l3[0],tst_l3[1],tst_l3[2])
trlf3 = F1(tra_l3[0],tra_l3[1],tra_l3[2])

l2 = logit(xfold2,yfold2,0.001,0.01)
tst_l2 = estimate(xfold2,yfold2,l2)
tra_l2 = estimate(x_train2,y_train2,l2)
tlf2 = F1(tst_l2[0],tst_l2[1],tst_l2[2])
trlf2 = F1(tra_l2[0],tra_l2[1],tra_l2[2])

l1 = logit(xfold1,yfold1,0.001,0.01)
tst_l1 = estimate(xfold1,yfold1,l1)
tra_l1 = estimate(x_train1,y_train1,l1)
tlf1 = F1(tst_l1[0],tst_l1[1],tst_l1[2])
trlf1 = F1(tra_l1[0],tra_l1[1],tra_l1[2])

l0 = logit(xfold0,yfold0,0.001,0.01)
tst_l0 = estimate(xfold0,yfold0,l0)
tra_l0 = estimate(x_train0,y_train0,l0)
tlf0 = F1(tst_l0[0],tst_l0[1],tst_l0[2])
trlf0 = F1(tra_l0[0],tra_l0[1],tra_l0[2])
print('TESTavgf1(gamma=0.001,c=0.01):',(tlf4+tlf3+tlf2+tlf1+tlf0)/5)  
print('TRAINavgf1(gamma=0.001,c=0.01):',(trlf4+trlf3+trlf2+trlf1+trlf0)/5)

#-----average f1 score with gamma=0.001, sigma=0.001-----#
l4 = logit(xfold4,yfold4,1,1)
tst_l4 = estimate(xfold4,yfold4,l4)
tra_l4 = estimate(x_train4,y_train4,l4)
tp4 = p(tst_l4[0],tst_l4[1])
tr4 = r(tst_l4[0],tst_l4[2])
trp4 = p(tra_l4[0],tra_l4[1])
trr4 = p(tra_l4[0],tra_l4[2])
tlf4 = F1(tst_l4[0],tst_l4[1],tst_l4[2])
trlf4 = F1(tra_l4[0],tra_l4[1],tra_l4[2])

l3 = logit(xfold3,yfold3,1,1)
tst_l3 = estimate(xfold3,yfold3,l3)
tra_l3 = estimate(x_train3,y_train3,l3)
tp4 = p(tst_l3[0],tst_l3[1])
tr4 = r(tst_l3[0],tst_l3[2])
trp4 = p(tra_l3[0],tra_l3[1])
trr4 = p(tra_l3[0],tra_l3[2])
tlf3 = F1(tst_l3[0],tst_l3[1],tst_l3[2])
trlf3 = F1(tra_l3[0],tra_l3[1],tra_l3[2])

l2 = logit(xfold2,yfold2,1,1)
tst_l2 = estimate(xfold2,yfold2,l2)
tra_l2 = estimate(x_train2,y_train2,l2)
tp4 = p(tst_l2[0],tst_l2[1])
tr4 = r(tst_l2[0],tst_l2[2])
trp4 = p(tra_l2[0],tra_l2[1])
trr4 = p(tra_l2[0],tra_l2[2])
tlf2 = F1(tst_l2[0],tst_l2[1],tst_l2[2])
trlf2 = F1(tra_l2[0],tra_l2[1],tra_l2[2])

l1 = logit(xfold1,yfold1,1,1)
tst_l1 = estimate(xfold1,yfold1,l1)
tra_l1 = estimate(x_train1,y_train1,l1)
tp4 = p(tst_l1[0],tst_l1[1])
tr4 = r(tst_l1[0],tst_l1[2])
trp4 = p(tra_l1[0],tra_l1[1])
trr4 = p(tra_l1[0],tra_l1[2])
tlf1 = F1(tst_l1[0],tst_l1[1],tst_l1[2])
trlf1 = F1(tra_l1[0],tra_l1[1],tra_l1[2])

l0 = logit(xfold0,yfold0,1,1)
tst_l0 = estimate(xfold0,yfold0,l0)
tra_l0 = estimate(x_train0,y_train0,l0)
tp4 = p(tst_l0[0],tst_l0[1])
tr4 = r(tst_l0[0],tst_l0[2])
trp4 = p(tra_l0[0],tra_l0[1])
trr4 = p(tra_l0[0],tra_l0[2])
tlf0 = F1(tst_l0[0],tst_l0[1],tst_l0[2])
trlf0 = F1(tra_l0[0],tra_l0[1],tra_l0[2])
print('TESTavgp(gamma=1,c=1:',(tp4+tp3+tp2+tp1+tp0)/5)
print('TESTavgr(gamma=1,c=1:',(tr4+tr3+tr2+tr1+tr0)/5)
print('TRAINavgp(gamma=1,c=1:',(trp4+trp3+trp2+trp1+trp0)/5)
print('TRAINavgr(gamma=1,c=1:',(trr4+trr3+trr2+trr1+trr0)/5)
print('TESTavgf1(gamma=1,c=1):',(tlf4+tlf3+tlf2+tlf1+tlf0)/5)  
print('TRAINavgf1(gamma=1,c=1):',(trlf4+trlf3+trlf2+trlf1+trlf0)/5)


#-----sklearn logistic regression accuracy-----#
model_logit = LogisticRegression(penalty='l2',C=0.1,solver='lbfgs')
accl = cross_val_score(model_logit, xtrain_k, ytrain_k, cv=5)
print('sklearn accuracy:',np.mean(accl))



###-----Naive Bayes-----###

#-----combining data-----#
y_fold4 = yfold4.reshape(16000,1)
y_fold3 = yfold3.reshape(16000,1)
y_fold2 = yfold2.reshape(16000,1)
y_fold1 = yfold1.reshape(16000,1)
y_fold0 = yfold0.reshape(16000,1)
print(y_fold4.shape)
print(xfold4.shape)
fold4 = np.hstack((y_fold4,xfold4))
fold3 = np.hstack((y_fold3,xfold3))
fold2 = np.hstack((y_fold2,xfold2))
fold1 = np.hstack((y_fold1,xfold1))
fold0 = np.hstack((y_fold0,xfold0))
print(fold4)
print(fold4.shape)

#-----object oriented programming-----#
'''
t0 = Data(fpath='training00.data')
t1 = Data(fpath='training01.data')
t2 = Data(fpath='training02.data')
t3 = Data(fpath='training03.data')
t4 = Data(fpath='training04.data')
t = Data(fpath='test.liblinear')
tr = Data(fpath='train.liblinear')
t0._load_data(fpath="training00.data")
t0._set_attributes_info(index_column_dict,data)
t0.get_row_subset(attribute_name,attribute_value)
'''


def tran_x(X):
    return X.T

def p_label(X):
    pos = 0
    neg = 0
    for i in X[0]:
        if i == 1:
            pos += 1
        else:
            neg += 1
        
    print("Positive: %d" %  pos)
    print("Negative: %d" %  neg)
    ppos = pos/len(X)
    pneg = neg/len(X)
    
    return ppos, pneg

print(p_label(fold4))





            

def frequency(X,Y):
    pos = []
    neg = []
    m = 218
    for i in range(219):
        if Y[i] == 1:
            if X[i][m] == 1:
                pos[0][2*m+0]
            else:
               print(neg[1][2*m+1])

    return pos, neg

'''
                print(pos)
        elif Y[i] == -1:
            #print('####')
            neg = np.count_nonzero(X[i], axis=0)
        else:
            pass
    
    return pos,neg
        
print(frequency(xfold4,yfold4))
#frequency(xfold4,yfold4)
'''

model_bayes = GaussianNB(var_smoothing=1.5)
accb = cross_val_score(model_bayes, xtrain_k, ytrain_k, cv =5)
print('sklearn accuracy:', np.mean(accb))



###-----SVM Over Trees-----###

print('please wait....this will take a bit..')
model_rf = RandomForestClassifier(n_estimators=200, max_depth=10)
model_svm = svm.SVC(C=1, gamma=0.01, probability=True)
ensemble_svmot = VotingClassifier(estimators=[('dt',model_rf), ('svm',model_svm)],voting='soft')
accuracy = cross_val_score(ensemble_svmot, xtrain_k, ytrain_k, cv=5)
print('sklearn accuracy',np.mean(accuracy))

