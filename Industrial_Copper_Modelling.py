#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import streamlit as st
import re
st.set_page_config(layout="wide")

st.write("""
<div style='text-align:center'>
    <h1 style='color:#5e17eb;'>Industrial Copper Modeling Application</h1>
</div>
""", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["PREDICT SELLING PRICE", "PREDICT STATUS"])

#Predicting the Selling Price
with tab1:
    # Define the possible values for the dropdown menus
   
    country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67.,
                           79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    product = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665',
               '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407',
               '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662',
               '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738',
               '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
    
    # Define the widgets for user input
    with st.form("my_form"):
        col1, col2 = st.columns(2)
        with col1:
                     
            st.write(
                f'<h5 style="color:#ee4647;">Enter 0 or 1 for Item_Type Columns</h5>',
                unsafe_allow_html=True)
            item_type_WI=st.text_input("Item_type_WI")
            item_type_W=st.text_input("Item_type_W")
            item_type_SLAWR=st.text_input("Item_type_SLAWR")
            item_type_S=st.text_input("Item_type_S")
            item_type_PL=st.text_input("Item_type_PL")
            item_type_Others=st.text_input("Item_type_Others")


        with col2:
            st.write(
                f'<h5 style="color:#ee4647;">NOTE: Min & Max given for reference, you can enter any value</h5>',
                unsafe_allow_html=True)
            country = st.selectbox("Country", sorted(country_options), key=3)
            application = st.selectbox("Application", sorted(application_options), key=4)
            product_ref = st.selectbox("Product Reference", product, key=5)
            quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
            thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            width = st.text_input("Enter width (Min:1, Max:2990)")
            customer = st.text_input("customer ID (Min:12458, Max:30408185)")

            submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
            
    if  submit_button:
        import pickle
        
        with open(r'C:\Users\SKAN\Desktop\Raajee\copper\model.pkl', 'rb') as file:
           loaded_model= pickle.load(file)   
        
        
        new_sample = [[float(country),float(application),float(product_ref),float(item_type_WI),float(item_type_W),float(item_type_SLAWR),float(item_type_S),
                       float(item_type_PL),float(item_type_Others),float(quantity_tons),float(thickness),float(width),float(customer)]]
        
        new_pred = loaded_model.predict(new_sample)[0]
        st.write('## :green[Predicted selling price:] $.',"{:.2f}".format(new_pred) )  

#Predicting the status        
with tab2:
     try:   
        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f'<h5 style="color:#ee4647;">Enter 0 or 1 for Item_Type Columns</h5>',
                unsafe_allow_html=True)
            citem_type_WI=st.text_input("Item_type_WI")
            citem_type_W=st.text_input("Item_type_W")
            citem_type_SLAWR=st.text_input("Item_type_SLAWR")
            citem_type_S=st.text_input("Item_type_S")
            citem_type_PL=st.text_input("Item_type_PL")
            citem_type_Others=st.text_input("Item_type_Others")
           
        with col2:
          
            ccountry = st.selectbox("Country", sorted(country_options), key=31)
            capplication = st.selectbox("Application", sorted(application_options), key=41)
            cproduct_ref = st.selectbox("Product Reference", product, key=51)
            cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
            cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            cwidth = st.text_input("Enter width (Min:1, Max:2990)")
            ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
               
            csubmit_button = st.button("PREDICT STATUS")          

            if csubmit_button:
                import pickle
                with open(r'C:\Users\SKAN\Desktop\Raajee\copper\cmodel.pkl', 'rb') as file:
                      cloaded_model = pickle.load(file)
        
            new_sample1 = [[int(citem_type_WI),int(citem_type_W),int(citem_type_SLAWR),int(citem_type_S),int(citem_type_PL),
                            int(citem_type_Others),float(ccountry),float(capplication),float(cproduct_ref),float(cquantity_tons),
                            float(cthickness),float(cwidth),float(ccustomer)]]
        
            new_pred1 = cloaded_model.predict(new_sample1)[0] 
            st.write(new_pred1)
            if new_pred1 == 1:
                st.write('## :green[The Status is Won] ')
            else:
                st.write('## :red[The status is Lost] ')
     except:
         st.write("")        
          
              

    
            

