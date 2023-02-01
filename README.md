# Customer-churn-prediction
This contains code for customer churn prediction
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN
import pickle
Data=pd.read_csv(r'C:\Users\t1u5h\Dropbox\Data analytics\Python libraries vs code\tel_churn.csv')
x = Data.drop('Churn', axis = 1)
y = Data['Churn']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
model_Data = DecisionTreeClassifier(criterion= 'gini',random_state= 100, max_depth= 6, min_samples_leaf=8)
model_Data.fit(x_train , y_train)
y_pred = model_Data.predict(x_test)
model_Data.score(x_test, y_pred)
print(classification_report(y_test, y_pred, labels = [0,1]))
sm = SMOTEENN()
x_resampled, y_resampled = sm.fit_resample(x,y)
xs_train, xs_test, ys_train, ys_test = train_test_split(x_resampled ,y_resampled, test_size = 0.2)
model_Data_SMOTEENN = DecisionTreeClassifier(criterion= 'gini',random_state= 100, max_depth= 6, min_samples_leaf=8)
model_Data_SMOTEENN.fit(xs_train , ys_train)
y_pred_SMOTEENN = model_Data_SMOTEENN.predict(xs_test)
y_pred_SMOTEENN
model_Data_SMOTEENN.score(xs_test, y_pred_SMOTEENN)
print(classification_report(ys_test, y_pred_SMOTEENN, labels = [0,1]))
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators = 100,criterion= 'gini',random_state= 100, max_depth= 6, min_samples_leaf=8)
model_rf.fit(x_train , y_train)
y_pred_rf = model_rf.predict(x_test)
print(classification_report(y_test, y_pred, labels = [0,1]))
sm = SMOTEENN()
x_resampled, y_resampled = sm.fit_resample(x,y)
xs_train, xs_test, ys_train, ys_test = train_test_split(x_resampled ,y_resampled, test_size = 0.2)
model_rf_SMOTEENN = RandomForestClassifier(n_estimators = 100,criterion= 'gini',random_state= 100, max_depth= 6, min_samples_leaf=8)
model_rf_SMOTEENN.fit(xs_train , ys_train)
y_pred_SMOTEENN_rf = model_rf_SMOTEENN.predict(xs_test)
print(classification_report(ys_test, y_pred_SMOTEENN_rf, labels = [0,1]))
filename = 'model.sav'
pickle.dump(model_Data_SMOTEENN, open(filename, 'wb'))


Prediction models:
  precision    recall  f1-score   support

           0       0.83      0.93      0.87      1040
           1       0.68      0.45      0.54       367

    accuracy                           0.80      1407
   macro avg       0.75      0.69      0.71      1407
weighted avg       0.79      0.80      0.79      1407

with SMOTEENN in decisiontreeclassifier:
precision    recall  f1-score   support

           0       0.95      0.89      0.92       536
           1       0.91      0.96      0.94       647

    accuracy                           0.93      1183
   macro avg       0.93      0.93      0.93      1183
weighted avg       0.93      0.93      0.93      1183

with SMOTEENN in randomforestclassifier:
 precision    recall  f1-score   support

           0       0.97      0.90      0.93       533
           1       0.92      0.97      0.95       640

    accuracy                           0.94      1173
   macro avg       0.95      0.94      0.94      1173
weighted avg       0.94      0.94      0.94      1173

