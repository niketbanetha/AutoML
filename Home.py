import streamlit as st
import pandas as pd
import base64
import mysql.connector
import graphviz
import regression
import classification
import warnings
warnings.filterwarnings("ignore")


st.set_option('deprecation.showfileUploaderEncoding', False)

#Mysql Connection
mydb=mysql.connector.connect(host="freedb.tech",
user="freedbtech_niketjain23",
password="yYS814@1",
database="freedbtech_MajorProject")
mycursor=mydb.cursor()

def checker(df):
    null_count=df.isnull().sum().sum()
    flag=0
    if null_count!=0:
        st.error("{} Null value present in dataset".format(null_count))
        flag=1
    c=0
    for i in df:
        if df[i].dtype=="object":
            c+=1
            
    if c!=0:
        st.error("{} column contain string data type".format(c))
        flag=1
    
    if flag!=0:
        return False
    return True
        
        



def best_algorithm(d,_):
    if _=="regression":
        asec=sorted(d.items(),key=lambda kv:kv[1][0])
        desc=sorted(asec,key=lambda kv:kv[1][1],reverse=True)
        return desc[0][0]
    else:
        desc=sorted(d.items(),key=lambda kv:kv[1][0],reverse=True)
        desc2=sorted(desc,key=lambda k:k[1][0],reverse=True)
        return desc2[0][0]


def regression_func(dftrain=None,dftest=None,x=None,y=None):
      
    #st.write("Regression Algorithm is applied to given dataset")  
    
    progress_bar=st.progress(0)
    
    r=regression.regression()
    
    result=r.features_selection(dftrain[x],dftrain[y])
    #st.write(result)
    progress_bar.progress(10)
    
    result=r.data_normalization()
    #st.write(result)
    progress_bar.progress(20)
    
    result=r.data_splitting()
    #st.write(result)
    progress_bar.progress(30)
    
    result=r.linear_model()
    #st.write(result)
    progress_bar.progress(45)
    
    result=r.poly_degree2()
    #st.write(result)
    progress_bar.progress(60)
    
    result=r.decision_model()
    #st.write(result)
    progress_bar.progress(75)
    
    result=r.random_forest()
    #st.write(result)
    progress_bar.progress(100)
    
    
    d=r.compare_error()
    s=best_algorithm(d,"regression")
    model=r.best_model(s)
    if dftest.empty==False:
        pred=r.prediction(s,dftest)
        return [d,pred,model]
    
    return [d,model]

def classification_func(dftrain=None,dftest=None,x=None,y=None):
      
    #st.write("Classification Algorithm is applied to given dataset")  
    
    progress_bar=st.progress(0)
    
    r=classification.classification()
    
    result=r.features_selection(dftrain[x],dftrain[y])
    #st.write(result)
    progress_bar.progress(10)
    
    result=r.data_normalization()
    #st.write(result)
    progress_bar.progress(20)
    
    result=r.data_splitting()
    #st.write(result)
    progress_bar.progress(30)
    
    result=r.logistic_model()
    #st.write(result)
    progress_bar.progress(45)
    
    result=r.nb_model()
    #st.write(result)
    progress_bar.progress(60)
    
    result=r.decision_model()
    #st.write(result)
    progress_bar.progress(75)
    
    result=r.random_forest()
    #st.write(result)
    progress_bar.progress(100)
    
    
    d=r.compare_error()
    s=best_algorithm(d,"classification")
    model=r.best_model(s)
    if dftest.empty==False:
        pred=r.prediction(s,dftest)
        return [d,pred,model]
    
    return [d,model]
    


# How frontend part work
def instruction_graph():
    graph = graphviz.Digraph()
    graph.edge("Home","Register") 
    graph.edge("Home","Login")
    graph.edge("Register","Enter email address and password")
    graph.edge("Enter email address and password","Login")
    graph.edge("Login","Select type of dataset")
    graph.edge("Select type of dataset","Upload training dataset")
    graph.edge("Upload training dataset","Select columns for which prediction hast to be done")
    graph.edge("Select columns for which prediction hast to be done","Upload test dataset[Optional]")
    graph.edge("Upload test dataset[Optional]","Click on start button")
    graph.edge("Click on start button","Download predicted file and pickle model of best algorithm")
    st.graphviz_chart(graph,width=500,height=200)
    
    
 #To check that whether username is already registered or not   
def search(username):
    global mycursor
    mycursor.execute('Select id from register where email_id="{}"'.format(username))
    res=mycursor.fetchall()
    if len(res)>=1:
        return True
    return False

# Register a new user
def register(username,password):
    k=search(username)
    if k==False:
        global mycursor,mydb
        mycursor.execute('Insert into register (email_id,password) values ("{}","{}")'.format(username,password))
        mydb.commit()
        return True
    else:
        return False

    
# Login 
def login(username,password):
    global mycursor
    mycursor.execute('Select password from register where email_id="{}"'.format(username))
    res=mycursor.fetchall()
    for i in res:
        if i[0]==password:
            return True
    return False

def download_model(model):
    b64 = base64.b64encode(model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="myfile.pkl">Download Trained Model .pkl File</a>'
    st.markdown(href, unsafe_allow_html=True)
    

# Show bar grpah where x is mse/r2score and y is algorithm name
def graph(d,_):
    algorithm=""
    if _=="regression":
        algorithm=best_algorithm(d,"regression")
        st.table(pd.DataFrame(d,index=["Mean Squared Error","R2 Score"]))
        
    else:
        algorithm=best_algorithm(d,"classification")
        st.table(pd.DataFrame(d,index=["Accuracy","F1 Score"]))
    
    st.markdown("<p> **Best Algorithm is <strong>{}</strong> </p>".format(algorithm),unsafe_allow_html=True)
        
    
    
    
# Portion that is visible to user after login is successfull
def home():
    
    g=st.selectbox(label="Type of Dataset",options=["Regression","Classification"])
    
    st.info("Option selected is {}".format(g))
    
    dftrain=upload("Trainig Dataset")
    
    colname=""
    
    if dftrain.empty==False:
        st.subheader("Uploaded Training Dataset Preview")
        st.dataframe(dftrain.head(),height=2000,width=1000)
        st.info("Dimension of dataset is {}".format(dftrain.shape))
        
        colname=st.selectbox("Name of colum for which prediction has to be done.",options=dftrain.columns)
    
    dftest=upload("Test Dataset [Optional]")
    
    if dftest.empty==False:
        st.subheader("Uploaded Test Dataset Preview")
        st.dataframe(dftest.head(),height=2000,width=1000)
        st.info("Dimension of dataset is {}".format(dftest.shape))
        
    
    button=st.button("Click to Start")
    
    if button:
        if dftrain.empty==True:
            st.error("Upload trainig dataset")
        if colname=="":
            st.error("Select column for which prediction has to be done")
        if checker(dftrain):
            if g=="Regression":
                if dftrain.empty==False and colname!="":
                    x=list(dftrain.columns)
                    x.remove(colname)
                    y=[colname]
                    d=regression_func(dftrain,dftest,x,y)
                    graph(d[0],"regression")
                    if len(d)==2:
                        model=d[1]
                        download_model(model)
                    else:
                        #graph(d[0],regression)
                        model=d[2]
                        download_model(model)
                        st.markdown(download(d[1]), unsafe_allow_html=True)
            else:
                    if dftrain.empty==False and colname!="":
                        x=list(dftrain.columns)
                        x.remove(colname)
                        y=[colname]
                        d=classification_func(dftrain,dftest,x,y)
                        graph(d[0],"classification")
                        if len(d)==2:
                            model=d[1]
                            download_model(model)
                        else:
                            model=d[2]
                            download_model(model)
                            st.markdown(download(d[1]), unsafe_allow_html=True)



        
                
            
            

# For uploading training and test dataset
def upload(s):
    file_bytes = st.file_uploader("{} (csv or excel file)".format(s),type=("csv","xlsx"))
    df=pd.DataFrame()
    if file_bytes!=None:
        try:
            df=pd.read_csv(file_bytes)
        except:
            df=pd.read_excel(file_bytes)
    return df

# for downoad of test dataset
def download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<center><a href="data:file/csv;base64,{b64}" download="predicted_value.csv"><strong>Download Predicted Output</strong></a></center>'

#Home Page
def intro():
    option=st.sidebar.selectbox("Choose page to navigate",options=["Home","Login","Register"])
    if option=="Home":
        st.markdown("<h1 style='text-align: center;'>Comparative Analysis of Regression and Classification algorithms over cloud</h1>", unsafe_allow_html=True)
        #st.markdown("<h3>Team Member</h3><ul><li>Naman Gupta</li><li>Niket Jain</li><li>Rajan Sainher</li><li>Aditya Malik</li></ul>", unsafe_allow_html=True)
        #st.markdown("<h4>Mentor Name: Amar Shukla</h4>",unsafe_allow_html=True)
        instruction_graph()
        
    elif option=="Login":
        username=st.sidebar.text_input("Enter email address")
        password=st.sidebar.text_input("Enter password",type="password")
        button=st.sidebar.checkbox("Login")
        
        if button:
            if username=="" or password=="":
                st.sidebar.error("Enter the details")
            else:
                if login(username,password):
                    home()
                else:
                    st.sidebar.error("Wrong email address/password")
    elif option=="Register":
        username=st.sidebar.text_input("Enter email address")
        password=st.sidebar.text_input("Enter password",type="password")
        button=st.sidebar.checkbox("Register")
        
        st.markdown("<h1 style='text-align: center;'>Comparative Analysis of Regression and Classification algorithms over cloud</h1>", unsafe_allow_html=True)
        instruction_graph()
        #st.markdown("<h3>Team Member</h3><ul><li>Naman Gupta</li><li>Niket Jain</li><li>Rajan Sainher</li><li>Aditya Malik</li></ul>", unsafe_allow_html=True)
        #st.markdown("<h4>Mentor Name: Amar Shukla</h4>",unsafe_allow_html=True)
        if button:
            if username=="" or password=="":
                st.sidebar.error("Enter the details")
            else:
                if register(username,password):
                    st.sidebar.info("Move to login page")
                else:
                    st.sidebar.error("Email id aleady registered")
        
                
                
                              
if __name__=="__main__":
    intro()
