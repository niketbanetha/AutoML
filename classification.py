# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 02:56:31 2020

@author: abc
"""


# to convert values of dataset into range 0 to 1
from sklearn.preprocessing import MinMaxScaler
# to split training dataset into train and test sub dataset
from sklearn.model_selection import train_test_split

#Feature Selection
from sklearn.feature_selection import RFECV


#Classification Algorithm
# 1. Logistic Regression
# 2. Random Forest Classification
# 3. Naive Bayes
# 4. Decision Tree Classification

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import   LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Evaluation Metrics
from sklearn.metrics import accuracy_score,f1_score

import pickle

class classification():
    def __init__(self):
        self.lm=LogisticRegression()
        self.rfg=RandomForestClassifier()
        self.dt=DecisionTreeClassifier()
        self.nb=GaussianNB()
        self.minmax=MinMaxScaler()
        self.features=[]
      
    
    #Feature Selection
    def features_selection(self,X,Y):
        dt=DecisionTreeClassifier()
        ref=RFECV(dt)
        f=ref.fit(X,Y)
        col=X.columns
        k=f.ranking_
        for i in range(len(k)):
            if k[i]==1:
                self.features.append(col[i])
        self.X=X[self.features]
        self.Y=Y
        return "Feature Selection Process Completed."

    #Normalization
    def data_normalization(self):
        self.X_norm=self.minmax.fit_transform(self.X)
        return "Data Normalisation Process Completed."
        
    #Splitting dataset into train and test
    def data_splitting(self): 
        self.train_x,self.test_x,self.train_y,self.test_y=train_test_split(self.X_norm,self.Y,random_state=0)
        return "Data Spliting into train and test completed"


    #Logistic Regression    
    def logistic_model(self):
        self.lm.fit(self.train_x,self.train_y)
        self.predlm=self.lm.predict(self.test_x)
        return "Linear Regression applied successfully"
    
    #Random Forest Classifier
    def random_forest(self):
        self.rfg.fit(self.train_x,self.train_y)
        self.predrfg=self.rfg.predict(self.test_x)
        return "Random Forest Classifier applied successfully"

    # Decision Tree Classifier
    def decision_model(self):
        self.dt.fit(self.train_x,self.train_y)
        self.preddt=self.dt.predict(self.test_x)
        return "Decision Tree  applied successfully"
    
    #Gauusian NB
    def nb_model(self):
         self.nb.fit(self.train_x,self.train_y)
         self.prednb=self.nb.predict(self.test_x)
         return "NB applied successfully"
        
    def prediction(self,model_name,dftest):
        scaler=MinMaxScaler()
        self.test=scaler.fit_transform(dftest[self.features])
        pred=None
        if model_name=="DecisionTreeClassifier":
            pred=self.dt.predict(self.test)
        elif model_name=="RandomForestClassifier":
            pred=self.rfg.predict(self.test)
        elif model_name=="LogisticRegression":
            pred=self.lm.predict(self.test)
        elif model_name=="NB":
            pred=self.nb.predict(self.test)
        
        dftest["Predicted Value"]=pred
        return dftest
        
    
    def compare_error(self):
        d={"DecisionTreeClassifier":[accuracy_score(self.test_y,self.preddt),f1_score(self.test_y,self.preddt)],
        "RandomForestClassifier":[accuracy_score(self.test_y,self.predrfg),f1_score(self.test_y,self.predrfg)],
        "LogisticRegression":[accuracy_score(self.test_y,self.predlm),f1_score(self.test_y,self.predlm)],
        "NB":[accuracy_score(self.test_y,self.prednb),f1_score(self.test_y,self.prednb)]
        }
        return d

    def best_model(self,model_name):
        model=None
        
        if model_name=="DecisionTreeClassifier":
            model=pickle.dumps(self.dt)
        elif model_name=="NB":
            model=pickle.dumps(self.nb)
        elif model_name=="RandomForestClassifier":
            model=pickle.dumps(self.rfg)
        elif model_name=="LogisticRegression":
            model=pickle.dumps(self.lm)
        

        return model










