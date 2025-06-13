# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 17:42:15 2025

@author: shreya
"""

# DATA PRE-PROCESSING PIPELINE   


# 1 - import lib.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2 - import dataset
dataset=pd.read_csv(r"D:\nit_prac\10th june\data.csv")

# 3 - split data into x and y

x=dataset.iloc[:,:-1].values  #independent var
y=dataset.iloc[:,3].values    #dependent var


# 3 - data cleaning  
# missing value imputation 
# transformation from cat to num
# convert entire dataset to number


from sklearn.impute import SimpleImputer

imputer=SimpleImputer()  # parameter tuning  -->default param="mean"


#hyper parameter tuning or fine tuning
#imputer=SimpleImputer(strategy="mode")  --->  InvalidParameterError mode not in this

#imputer=SimpleImputer(strategy="most_frequent")

imputer=imputer.fit(x[:,1:3])

x[:,1:3] = imputer.transform(x[:,1:3])

# ⭐ convert categorical data to numerical data

from sklearn.preprocessing import LabelEncoder

labelencoder_x=LabelEncoder()

labelencoder_x.fit_transform(x[:,0])
x[:,0]=labelencoder_x.fit_transform(x[:,0])

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

# 5 - train_test_split

from sklearn.model_selection import train_test_split

x_train, x_text, y_train, y_test = train_test_split(x,y,train_size=0.8,test_size=0.2)

# 👆 if we run the above code another time then data picked will be random.... everytime changing... no accuracy achieved

x_train, x_text, y_train, y_test = train_test_split(x,y,train_size=0.8,test_size=0.2, random_state=0)

# random_state will give same records in every run 


# FEATURE SCALING

from sklearn.preprocessing import Normalizer

sc_X = Normalizer() 

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)












