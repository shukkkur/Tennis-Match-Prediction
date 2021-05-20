__author__ = 'shukkkur'

'''
GitHub: https://github.com/shukkkur
Date: 17/05/2021

***

I build and trained the following linear model classifiers:
    Ridge
    SVC
    KNN
    Logistic Regression
    Decision Tree
According to my results, SVC performs the best. The final answers would be based on these algorithm.

The Final Results is stored in DataFrame names "df"
'''


# Neccessary Imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix


# To Display All Columns, when printing a DataFrame
pd.options.display.max_columns = None

##def namestr(obj, namespace=globals()):
##      '''Print the name of the variable'''
##      return ' '.join([name for name in namespace if namespace[name] is obj][0].split("_"))

# Training Data
Sachenko_Kristina = 'https://raw.githubusercontent.com/shukkkur/Tennis-Match-Prediction/main/TrainingData/SachenkoKristina.csv'
Schwaibiger_Anastasia = 'https://raw.githubusercontent.com/shukkkur/Tennis-Match-Prediction/main/TrainingData/SchwaibigerAnastasia.csv'
Shelekhova_Laura = 'https://raw.githubusercontent.com/shukkkur/Tennis-Match-Prediction/main/TrainingData/ShelekhovaLaura.csv'
Solakov_Leticia = 'https://raw.githubusercontent.com/shukkkur/Tennis-Match-Prediction/main/TrainingData/SolakovLeticia.csv'
Wolter_Franziska = 'https://raw.githubusercontent.com/shukkkur/Tennis-Match-Prediction/main/TrainingData/WolterFranziska.csv'
Wächter_Daria = 'https://raw.githubusercontent.com/shukkkur/Tennis-Match-Prediction/main/TrainingData/W%C3%A4chterDaria.csv'
Öztürk_Tuana = 'https://raw.githubusercontent.com/shukkkur/Tennis-Match-Prediction/main/TrainingData/%C3%96zt%C3%BCrkTuana.csv'

files = [Sachenko_Kristina, Schwaibiger_Anastasia,Shelekhova_Laura,Solakov_Leticia,Wolter_Franziska,Wächter_Daria, Öztürk_Tuana]

# Prediction Data
Grigorieva_Darja = 'https://raw.githubusercontent.com/shukkkur/Tennis-Match-Prediction/main/PredictionData/MoreData/GrigorievaDarja_original.csv'
Khomich_Michelle = 'https://raw.githubusercontent.com/shukkkur/Tennis-Match-Prediction/main/PredictionData/MoreData/KhomichMichelle_original.csv'
Nitzsche_Lavinia_Maria = 'https://raw.githubusercontent.com/shukkkur/Tennis-Match-Prediction/main/PredictionData/MoreData/NitzscheLaviniaMaria_original.csv'
Ribbert_Paula = 'https://raw.githubusercontent.com/shukkkur/Tennis-Match-Prediction/main/PredictionData/MoreData/RibbertPaula_original.csv'
Wächter_Daria = 'https://raw.githubusercontent.com/shukkkur/Tennis-Match-Prediction/main/PredictionData/MoreData/W%C3%A4chterDaria_original.csv'

predict = [Grigorieva_Darja,Khomich_Michelle,Nitzsche_Lavinia_Maria,Ribbert_Paula,Wächter_Daria]

### To train my models, it would be easier if all the csv's are in a single file.
### Concatenate the files
lst = []

for file in files:
    df = pd.read_csv(file, index_col=None, header=0)
    lst.append(df)

df = pd.concat(lst, axis=0, ignore_index=True)
#dfs.replace({'verloren':'lose', 'gewonnen':'win'}, inplace = True)
#print(df.player1_name.unique())


### Training And Testing Data
cols = ['lk1', 'lk2']
X = df[cols]
y = df['match_outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0, shuffle=True)

# Ridge Classifier - L2
ridge = RidgeClassifier(alpha = 0.5)
ridge.fit(X_train, y_train)

# SVC
svc = SVC(C = 10)
svc.fit(X_train, y_train)

# KNN - Neighrest Neighbor 
knn = KNeighborsClassifier(n_neighbors=18)
knn.fit(X_train, y_train)

# Logistic Regression Classifier
logreg = LogisticRegression(C = 0.01)
logreg.fit(X_train, y_train)

# Decision Tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)


if __name__ == "__main__":
##      print('Ridge Classifier Score: {:.2f}'.format(ridge.score(X_test, y_test)))
##      print('SVC Score: {:.2f}'.format(svc.score(X_test, y_test)))
##      print('KNN Score: {:.2f}'.format(knn.score(X_test, y_test)))
##      print('LogReg Score: {:.2f}'.format(logreg.score(X_test, y_test)))
##      print('Decision Tree Score: {:.2f}'.format(tree.score(X_test, y_test)))
##      print('\n\n')
      
      # SVC performs the best, threfore the final answers would be according to SVC model
##      test = df[['lk1', 'lk2']]
##      y_pred = svc.predict(test)
##      df.insert(7, 'predicted_outcome', y_pred)
##      print(df.head())
##      print('\n\n')
      
##      # Confusion Matrix To evaluate our results
##      matrix = confusion_matrix(df.match_outcome.values, y_pred)
##      print('Confusion Matrix\n', matrix, '\n')
##      print("True Positives (correctly predicted 'verloren')", matrix[1,1])
##      print("False Positives (incorrectly predicted 'verloren')", matrix[0,1])
##      print("True Negatives (correctly predicted 'gewonnen')", matrix[0,0])
##      print("False Negatives (incorrectly predicted 'gewonnen')", matrix[1,0])
##
##      print('\nDataFrame is stored in variable "df"')
      for file in predict:
            df = pd.read_csv(file)
            X = df[cols]
            print(df.iloc[0]['player1_name'])
            print(svc.predict(X), end='\n\n')
            
            
