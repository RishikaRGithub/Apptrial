# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 11:49:14 2021

@author: Ranganath R
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.title('DreamHomes.com')
st.text('Find your dream homes here')
df=pd.read_csv('kc_house_data.csv')
st.image('homes.jpg')
st.video('https://www.youtube.com/watch?v=_dzr1pm3Ymw')
#LINK SLIDER TO PLOT in this code
price_set=st.slider("Price Range",min_value=int(df['price'].min()),max_value=int(df['price'].max()),step=50,value=int(df['price'].max()))
st.text("Price Selected is "+str(price_set))
fig=px.scatter_mapbox(df.loc[df['price']<price_set],lat='lat',lon='long',color='sqft_living',size='price')
fig.update_layout(mapbox_style='open-street-map')
st.plotly_chart(fig)

#Do HW min as 100000 and max as 149000

#ML Part-20th session-13/10
#1. Display title as Price predictor for predictive modelling in ML
st.header('Price Predictor')
#2nd is to choose the type of regression - Linear/Ridge or Lasso, sel_Var is a list of the regression option chosen
sel_box_var=st.selectbox("Select Method",['Linear','Ridge','Lasso'],index=0)
#3rd method-select more than 1 option variable using multi_var that is a list having chosen var.s
multi_var=st.multiselect("Select Additional Variables for Accuracy=",['sqft_living','sqft_lot','sqft_basement'])
#4th ML part starts-create a new dictionary df_new that has converted multi_var list into a data frame
df_new=[]
df_new=df[multi_var]
#5th 
if sel_box_var=='Linear':
    df_new['bedrooms']=df['bedrooms']
    df_new['bathrooms']=df['bathrooms']
    X=df_new
    Y=df['price']
    model=LinearRegression()
#6th
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
    reg=model.fit(X_train,Y_train)  
    Y_pred = reg.predict(X_test)
    st.text("Intercept="+str(reg.intercept_))
    st.text("Coefficient="+str(reg.coef_))
    st.text("R^2="+str(r2_score(Y_test,Y_pred)))
    st.text("mse="+str(mean_squared_error(Y_test, Y_pred)))
elif sel_box_var=='Lasso':
    df_new['bedrooms']=df['bedrooms']
    df_new['bathrooms']=df['bathrooms']
    X=df_new
    Y=df['price']
    model=Lasso()
#7th
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
    reg=model.fit(X_train,Y_train)  
    Y_pred = reg.predict(X_test)
    st.text("Intercept="+str(reg.intercept_))
    st.text("Coefficient="+str(reg.coef_))
    st.text("R^2="+str(r2_score(Y_test,Y_pred)))
    st.text("mse="+str(mean_squared_error(Y_test, Y_pred)))
else:
    df_new['bedrooms']=df['bedrooms']
    df_new['bathrooms']=df['bathrooms']
    X=df_new
    Y=df['price']
    model=Ridge()
#8th
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
    reg=model.fit(X_train,Y_train)  
    Y_pred = reg.predict(X_test)
    st.text("Intercept="+str(reg.intercept_))
    st.text("Coefficient="+str(reg.coef_))
    st.text("R^2="+str(r2_score(Y_test,Y_pred)))
    st.text("mse="+str(mean_squared_error(Y_test, Y_pred)))
#9th-Depreciation
#st.set_option('depreciation.showPyplotGlobalUse',False)
st.set_option('deprecation.showPyplotGlobalUse', False)
sns.regplot(Y_test,Y_pred)
st.pyplot()
count=0
pred_val=0
for i in df_new.keys():
    try:
        val=st.text_input('Enter no./val of',+i) #enter beta coefficient
        pred_val=pred_val+float(val)*reg.coef_[count] #keep on adding to past previous val
        count=count+1 #increment each case
    except: #if any warnings, or errors, dont print it
        pass
st.text('Predicted Prices are:'+str(pred_val+reg.intercept_))

#Session-21 STARTS from this 
st.header("Application Details") #a new title or heading is displayed
img = st.file_uploader("Upload Application") 
st.text("Details for the representative to contact you")
st.text("Enter your address")
address=st.text_area("Your address Here") #ask user to enter address
date = st.date_input("Enter a date") #ask user to enter a date
time = st.time_input("Enter the time") #ask user to enter a time
if st.checkbox("I confirm the date and time", value=False): #radio button to ask user if date and time is fine or not, by default is False to be unchecked and user must click it
    st.write('Thanks for confirming!')
st.number_input('Rate our site', min_value = 1.0, max_value = 10.0, step = 1.0) #ask user review and min of 1, max of 10, with jump of 1 each time
st.write('We owe you a lot for taking time out of your busy schedule to fill this form for us and we are indebted to u, have a great day!')