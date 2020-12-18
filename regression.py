

# to convert values of dataset into range 0 to 1
from sklearn.preprocessing import MinMaxScaler
# to split training dataset into train and test sub dataset
from sklearn.model_selection import train_test_split

#Feature Selection
from sklearn.feature_selection import RFECV


#Regression Algorithm
# 1. Linear Regression
# 2. Random Forest Regressor
# 3. Polynomial Regression
# 4. Decision Tree Regressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

# Evaluation Metrics
from sklearn.metrics import mean_squared_error,r2_score

import pickle


class regression():
    def __init__(self):
        self.lm=LinearRegression()
        self.rfg=RandomForestRegressor()
        Input=[('polynomial',PolynomialFeatures(degree=2)),('modal',LinearRegression())]
        self.poly2=Pipeline(Input)
        self.dt=DecisionTreeRegressor()
        self.minmax=MinMaxScaler()
        self.features=[]

    #Feature Selection
    def features_selection(self,X,Y):
        dt=DecisionTreeRegressor()
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
        self.Y_norm=self.Y
        return "Data Normalisation Process Completed."

    #Splitting dataset into train and test
    def data_splitting(self):
        self.train_x,self.test_x,self.train_y,self.test_y=train_test_split(self.X_norm,self.Y_norm,random_state=0)
        return "Data Spliting into train and test completed"


    #Linear Regression
    def linear_model(self):
        self.lm.fit(self.train_x,self.train_y)
        self.predlm=self.lm.predict(self.test_x)
        return "Linear Regression applied successfully"
    
    #Random Forest Regressor
    def random_forest(self):
        self.rfg.fit(self.train_x,self.train_y)
        self.predrfg=self.rfg.predict(self.test_x)
        return "Random Forest Regressor applied successfully"
    
    # poly degree="2"
    def poly_degree2(self):
        self.poly2.fit(self.train_x,self.train_y)
        self.predd2=self.poly2.predict(self.test_x)
        return "Polynomial Regression(Degree=2) applied successfully"
    
    
    # Decision Tree Regressor
    def decision_model(self):
        self.dt.fit(self.train_x,self.train_y)
        self.preddt=self.dt.predict(self.test_x)
        return "Decision Tree Regressor applied successfully"
    
    
        
    def prediction(self,model_name,dftest):
        scaler=MinMaxScaler()
        self.test=scaler.fit_transform(dftest[self.features])
        pred=None
        if model_name=="DecisionTreeRegressor":
            pred=self.dt.predict(self.test)
        elif model_name=="Poly_Degree_2":
            pred=self.poly2.predict(self.test)
        elif model_name=="RandomForestRegressor":
            pred=self.rfg.predict(self.test)
        elif model_name=="LinearRegression":
            pred=self.lm.predict(self.test)
        
        
        dftest["Predicted Value"]=pred
        return dftest
        
    
    def compare_error(self):
        d={"DecisionTreeRegressor":[mean_squared_error(self.test_y,self.preddt),r2_score(self.test_y,self.preddt)],
        "Poly_Degree_2":[mean_squared_error(self.test_y,self.predd2),r2_score(self.test_y,self.predd2)],
        "RandomForestRegressor":[mean_squared_error(self.test_y,self.predrfg),r2_score(self.test_y,self.predrfg)],
        "LinearRegression":[mean_squared_error(self.test_y,self.predlm),r2_score(self.test_y,self.predlm)],
        }
        return d










