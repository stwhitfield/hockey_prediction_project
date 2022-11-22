import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics 
from sklearn.model_selection import GridSearchCV
import pickle
import joblib


# Read in data and assign X and y
data = pd.read_csv('../../data/train.csv', index_col=0)
X = data[data.columns.tolist()[:-1]]
y = data[['isGoal']]

   
#Train and valid split
X_train,X_val,y_train,y_val = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.values.ravel()

# Create the random grid
random_grid = {'splitter': ['best', 'random'],
               'max_features': ['auto', 'sqrt', 'log2'],
               'max_depth': [None, 10,  30,  50, 70,  90, 100, 120, 150],
               'min_samples_split': [ 2, 5, 10, 20, 30],
               'min_samples_leaf': [1, 2, 4, 10, 20, 30],
               'criterion': ['gini', 'entropy'] }
              
model = DecisionTreeClassifier()

gs = GridSearchCV(estimator = model, param_grid = random_grid,  cv = 5, verbose=2,  n_jobs = -1)
gs.fit(X_train, y_train)


print('gs.best_params_')
print(gs.best_params_)

import joblib
joblib.dump(gs.best_params_, 'dt_random_best_params.pkl')
joblib.dump(gs.best_estimator_, 'dt_random_best_estimator.pkl')
joblib.dump(gs, 'dt_random_best_model.pkl')
pickle.dump(gs, open('dt_random_best_modeli_pickle_dump.pkl', 'wb'))

y_pred = gs.predict(X_val)
    
pred_probs = gs.predict_proba(X_val)
probs_isgoal = pred_probs[:,1]

gs_best_score = gs.best_score_
accuracy = metrics.accuracy_score(y_val, y_pred)
f1_score = metrics.f1_score(y_val, y_pred)
precision = metrics.precision_score(y_val, y_pred)
recall = metrics.recall_score(y_val, y_pred)
cf_matrix = metrics.confusion_matrix(y_val,y_pred)
roc_auc = metrics.roc_auc_score(y_val,probs_isgoal)

print(f' accuracy: {accuracy}')
print(f' f1_score: {f1_score}')
print(f' precision: {precision}')
print(f' recall: {recall}')
print(f' roc_auc: {roc_auc}')
print('Confusion Matrix')
print(cf_matrix)
