#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px


# In[2]:


data=pd.read_csv("D:\DataScience Projects(amarkarwal)\Delivery-time\Delivery time\deliverytime.txt")


# In[3]:


print(data.head())


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[6]:


R=6371

def deg_to_rad(degrees):
    return degrees*(np.pi/180)

def distcalculate(lat1,lon1,lat2,lon2):
    d_lat=deg_to_rad(lat2-lat1)
    d_lon=deg_to_rad(lon2-lon1)
    a=np.sin(d_lat/2)**2+np.cos(deg_to_rad(lat1))*np.cos(deg_to_rad(lat2))*np.sin(d_lon/2)**2
    c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    return R*c
data['distance']=np.nan

for i in range(len(data)):
    data.loc[i,'distance']=distcalculate(data.loc[i,'Restaurant_latitude'],
                                        data.loc[i,'Restaurant_longitude'],
                                        data.loc[i,'Delivery_location_latitude'],
                                        data.loc[i,'Delivery_location_longitude'])
    
    


# In[7]:


print(data.head())


# In[8]:


figure=px.scatter(data_frame=data,x='distance',y='Time_taken(min)',size='Time_taken(min)',trendline='ols',title='Relationship b/w distance and time taken')
figure.show()


# In[9]:


fig=px.scatter(data_frame=data,x='Delivery_person_Age',
              y='Time_taken(min)',
              size='Time_taken(min)',
              color='distance',
              trendline='ols',
              title='Releationship b\w time taken and age')
fig.show()


# In[10]:


fig2=px.scatter(data_frame=data,x='Delivery_person_Ratings',
               y='Time_taken(min)',
               size='Time_taken(min)',
               color='distance',
               trendline='ols',
               title='Relationship between Time Taken and Ratings')
fig2.show()


# In[11]:


box=px.box(data,x='Type_of_vehicle',
          y='Time_taken(min)',
          color='Type_of_order')
box.show()


# In[12]:


from sklearn.model_selection import train_test_split
x=np.array(data[['Delivery_person_Age',
                'Delivery_person_Ratings',
                'distance']])
y=np.array(data[['Time_taken(min)']])
xtrain,xtest,ytrain,ytest=train_test_split(x,y,
                                          test_size=0.10,random_state=42)


# In[13]:


from keras.models import Sequential
from keras.layers import Dense,LSTM
model =Sequential()
model.add(LSTM(128,return_sequences=True,input_shape=(xtrain.shape[1],1)))
model.add(LSTM(64,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()


# In[14]:


model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(xtrain,ytrain,batch_size=1,epochs=9)


# In[15]:


print("food delivery time prediction")
a=int(input('Age of Delivery partner:'))
b=float(input('Ratings of Previous Deliveries:'))
c=int(input('Total Distance:'))

features=np.array([[a,b,c]])
print('Predicted Delivery Time in minutes =',model.predict(features))


# In[ ]:




