# Logistic Regression on affairs data
#classify that a person had an affair or not 

#Data description :
#A data frame containing 601 observations on 9 variables.
#affairs : numeric. How often engaged in extramarital sexual intercourse during the past year?
#gender : factor indicating gender.
#age : numeric variable coding age in years: 17.5 = under 20, 22 = 20–24, 27 = 25–29, 32 = 30–34, 37 = 35–39, 42 = 40–44, 47 = 45–49, 52 = 50–54, 57 = 55 or over.
#yearsmarried : numeric variable coding number of years married: 0.125 = 3 months or less, 0.417 = 4–6 months, 0.75 = 6 months–1 year, 1.5 = 1–2 years, 4 = 3–5 years, 7 = 6–8 years, 10 = 9–11 years, 15 = 12 or more years.
#children : factor. Are there children in the marriage?
#religiousness : numeric variable coding religiousness: 1 = anti, 2 = not at all, 3 = slightly, 4 = somewhat, 5 = very.
#education : numeric variable coding level of education: 9 = grade school, 12 = high school graduate, 14 = some college, 16 = college graduate, 17 = some graduate work, 18 = master's degree, 20 = Ph.D., M.D., or other advanced degree.
#occupation : numeric variable coding occupation according to Hollingshead classification (reverse numbering).
#rating : numeric variable coding self rating of marriage: 1 = very unhappy, 2 = somewhat unhappy, 3 = average, 4 = happier than average, 5 = very happy.

#importing libraries
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

#Importing Dataset
dataset=pd.read_csv('affairs.csv')

# creating dummy columns for the categorical columns 
dataset.columns
dummies = pd.get_dummies(dataset[["gender","children"]])
# Dropping the columns for which we have created dummies
dataset.drop(["gender","children"],inplace=True,axis = 1)

# adding the columns to the dataset data frame 
dataset = pd.concat([dataset,dummies],axis=1)

dataset["affairs_cat"] = 0

dataset.loc[dataset.affairs_cat>=1,"affairs_cat"] = 1
dataset.affairs.value_counts()
dataset.affairs_cat.value_counts()
dataset.drop(["affairs"],axis=1,inplace=True)
dataset.columns

dataset.head(10)

# Getting the barplot for the target columns vs features
sb.countplot(x="affairs_cat",data=dataset,palette="hls")
pd.crosstab(dataset.affairs_cat,dataset.age).plot(kind="bar")
pd.crosstab(dataset.affairs_cat,dataset.yearsmarried).plot(kind="bar")
pd.crosstab(dataset.affairs_cat,dataset.education).plot(kind="bar")
pd.crosstab(dataset.affairs_cat,dataset.occupation).plot(kind="bar")

sb.countplot(x="age",data=dataset,palette="hls")
pd.crosstab(dataset.age,dataset.education).plot(kind="bar")
sb.countplot(x="occupation",data=dataset,palette="hls")
sb.countplot(x="yearsmarried",data=dataset,palette="hls")

# Data Distribution - Boxplot of continuous variables wrt to each category of categorical columns

sb.boxplot(x="affairs_cat",y="age",data=dataset,palette="hls")
sb.boxplot(x="affairs_cat",y="education",data=dataset,palette="hls")

# To get the count of null values in the data 

dataset.isnull().sum() #no na values


dataset.shape 

# spillting into X as input and Y as output variables
X=dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9]].values #all the columns except last one OR X=dataset.iloc[:, :10].values
Y=dataset.iloc[:, 10].values # all the columns in index 3

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)

# Fitting Logistic Regression to the Training Set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X, Y)
classifier.coef_ # coefficients of features 
classifier.predict_proba (X) # Probability values

#predicting 
Y_pred=classifier.predict(X)

dataset["y_pred"] = Y_pred
y_prob = pd.DataFrame(classifier.predict_proba(X))
new_df = pd.concat([dataset,y_prob],axis=1)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y, Y_pred)
print (confusion_matrix)
type(Y_pred)
accuracy = sum(Y==Y_pred)/dataset.shape[0]
pd.crosstab(Y_pred,Y)

