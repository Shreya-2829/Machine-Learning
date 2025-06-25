
#  IMPORTNING THE LIBRARY

import numpy as np 	#Array		

import matplotlib.pyplot as plt		

import pandas as pd		

#--------------------------------------------

# import the dataset & divided my dataset into independe & dependent

dataset = pd.read_csv(r"D:\nit_prac\23rd june - gd,sgd,bgd\Scaling (transformer)\5. Data preprocessing\Data.csv")

#  X is the independent variables (features) --->    .iloc[:, :-1] selects all rows and all columns except the last one.

X = dataset.iloc[:, :-1].values	

#  Selects the 4th column (index 3) as y — the dependent variable (target/output).

y = dataset.iloc[:,3].values  

#--------------------------------------------

from sklearn.impute import SimpleImputer # SPYDER 4 

#  SimpleImputer helps fill in missing values (like NaN) in your data. 
#  Create an imputer object (by default, it fills missing values with the mean of the column).

imputer = SimpleImputer() 

#  Looks at columns 1 and 2 of X (i.e., 2nd and 3rd columns), and calculates their mean (used for filling missing values).

imputer = imputer.fit(X[:,1:3]) 

#  Fills missing values in columns 1 and 2 using the computed mean values.

X[:, 1:3] = imputer.transform(X[:,1:3])

#-----------------------------------------------------------------------------


#  HOW TO ENCODE CATEGORICAL DATA & CREATE A DUMMY VARIABLE

#  LabelEncoder is used to convert text labels (categorical) into numbers (ML models need numbers, not text).


from sklearn.preprocessing import LabelEncoder

#   This encodes the first column of X (like countries or categories) into numeric values.
#  Example: ["France", "Germany", "France"] → [0, 1, 0]

labelencoder_X = LabelEncoder()

labelencoder_X.fit_transform(X[:,0]) 

X[:,0] = labelencoder_X.fit_transform(X[:,0]) 

#  Also encodes the target variable y (if it is categorical like "Yes"/"No").

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

#-----------------------------------------------------------------------

#SPLITING THE DATASET IN TRAINING SET & TESTING SET

from sklearn.model_selection import train_test_split

#You need to train the model on one part and test it on another part.

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2,random_state=0) 

# if you remove random_stat then your model not behave as accurate 

#-----------------------------------------------------------------------

#FEATURE SCALING

from sklearn.preprocessing import Normalizer 

#  Import Normalizer to scale all values in a row so their total length = 1 (unit vector).

n_X = Normalizer() 

# Create a normalizer object.

X_train = n_X.fit_transform(X_train)

#  Fits the scaler to X_train and transforms it (normalizes the data).
#  Applies the same scaling to X_test.
X_test = n_X.transform(X_test)

#---------------------------------------------------------------------


from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)











