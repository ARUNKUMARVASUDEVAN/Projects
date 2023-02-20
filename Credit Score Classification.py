#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default ='plotly_white'


# In[4]:


data=pd.read_csv(r"C:\Users\ARUN KUMAR V\My datascience projects\Credit-Score-Data\Credit Score Data\train.csv")


# In[5]:


print(data.head())


# In[6]:


print(data.info())


# In[7]:


print(data.isnull().sum())


# In[8]:


data['Credit_Score'].value_counts()


# In[11]:


fig=px.box(data,x='Occupation',
          color='Credit_Score',
          title='Credit Scores Based on Occupation',
          color_discrete_map={'Poor':'red',
                             'Standard':'yellow',
                             'Good':'green'})
fig.show()


# In[12]:


fig=px.box(data,
          x='Credit_Score',
          y='Annual_Income',
          color='Credit_Score',
          title='Credit Score Based on Annual Income',
          color_discrete_map={'Poor':'red',
                             'Standard':'yellow',
                             'Good':'green'})
fig.update_traces(quartilemethod='exclusive')
fig.show()


# In[13]:


fig=px.box(data,
          x='Credit_Score',
          y='Monthly_Inhand_Salary',
          color='Credit_Score',
          title='Credit Scores Based on monthly inhand salary',
          color_discrete_map={'Poor':'red',
                             'Standard':'yellow',
                             'Good':'Green'})
fig.update_traces(quartilemethod='exclusive')
fig.show()


# In[23]:


fig=px.box(data,
          x='Credit_Score',
          y='Num_Bank_Accounts',
          color='Credit_Score',
               
          title= 'Credit score based on Num of bank accounts',
          color_discrete_map={'Poor':'red',
                             'Standard':'yellow',
                             'Good':'Green'})
fig.update_traces(quartilemethod='exclusive')

fig.show()


# In[24]:


fig=px.box(data,
          x='Credit_Score',
          y='Num_Credit_Card',
          color='Credit_Score',
           title='Credit score based on Num of credit card',
          color_discrete_map={'Poor':'red',
                             'Standard':'yellow',
                             'Good':'Green'})
fig.update_traces(quartilemethod='exclusive')
fig.show()


# In[25]:


fig=px.box(data,
          x='Credit_Score',
          y='Interest_Rate',
          color='Credit_Score',
          title='Credit_score on Interest Rate',
          color_discrete_map={'Poor':'red',
                             'Standard':'yellow',
                             'Good':'green'})
fig.update_traces(quartilemethod='exclusive')
fig.show()


# In[29]:


fig=px.box(data,
           x='Credit_Score',
          y='Num_of_Loan',
          color='Credit_Score',
          title='Num_of-loans Credit scores',
          color_discrete_map={'Poor':'red',
                             'Standard':'Yellow',
                             'Good':'Green'})
fig.update_traces(quartilemethod='exclusive')
fig.show()


# In[31]:


fig=px.box(data,
          x='Credit_Score',
          y='Delay_from_due_date',
          color='Credit_Score',
          color_discrete_map={'Poor':'red',
                             'Good':'green',
                             'Standard':'yellow'})
fig.update_traces(quartilemethod='exclusive')
fig.show()


# In[32]:


fig = px.box(data, 
             x="Credit_Score", 
             y="Outstanding_Debt", 
             color="Credit_Score", 
             title="Credit Scores Based on Outstanding Debt",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[33]:


fig = px.box(data, 
             x="Credit_Score", 
             y="Credit_Utilization_Ratio", 
             color="Credit_Score",
             title="Credit Scores Based on Credit Utilization Ratio", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[34]:


fig = px.box(data, 
             x="Credit_Score", 
             y="Credit_History_Age", 
             color="Credit_Score", 
             title="Credit Scores Based on Credit History Age",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[35]:


fig = px.box(data, 
             x="Credit_Score", 
             y="Total_EMI_per_month", 
             color="Credit_Score", 
             title="Credit Scores Based on Total Number of EMIs per Month",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[36]:


fig = px.box(data, 
             x="Credit_Score", 
             y="Amount_invested_monthly", 
             color="Credit_Score", 
             title="Credit Scores Based on Amount Invested Monthly",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[37]:


fig = px.box(data, 
             x="Credit_Score", 
             y="Monthly_Balance", 
             color="Credit_Score", 
             title="Credit Scores Based on Monthly Balance Left",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[38]:


data['Credit_Mix']=data['Credit_Mix'].map({'Standard':1,
                                          'Good':2,
                                          'Bad':0})


# In[39]:


data['Credit_Mix']


# In[48]:


from sklearn.model_selection import train_test_split
x=np.array(data[["Annual_Income", "Monthly_Inhand_Salary", 
                   "Num_Bank_Accounts", "Num_Credit_Card", 
                   "Interest_Rate", "Num_of_Loan", 
                   "Delay_from_due_date", "Num_of_Delayed_Payment", 
                   "Credit_Mix", "Outstanding_Debt", 
                   "Credit_History_Age", "Monthly_Balance"]])
y=np.array(data[["Credit_Score"]])


# In[49]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.33,random_state=42)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(xtrain,ytrain)


# In[50]:


print("Credit Score Prediction : ")
a = float(input("Annual Income: "))
b = float(input("Monthly Inhand Salary: "))
c = float(input("Number of Bank Accounts: "))
d = float(input("Number of Credit cards: "))
e = float(input("Interest rate: "))
f = float(input("Number of Loans: "))
g = float(input("Average number of days delayed by the person: "))
h = float(input("Number of delayed payments: "))
i = input("Credit Mix (Bad: 0, Standard: 1, Good: 3) : ")
j = float(input("Outstanding Debt: "))
k = float(input("Credit History Age: "))
l = float(input("Monthly Balance: "))


features = np.array([[a, b, c, d, e, f, g, h, i, j, k, l]])
print("Predicted Credit Score = ", model.predict(features))

