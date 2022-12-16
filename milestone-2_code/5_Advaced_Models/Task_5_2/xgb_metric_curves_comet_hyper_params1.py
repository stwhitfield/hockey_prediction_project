#!/usr/bin/env python
# coding: utf-8

# Import comet_ml at the top of your file, before sklearn!
from comet_ml import Experiment
import os 
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
import xgboost as xgb
from plot_metrics import *


# Create an experiment with your api key
experiment = Experiment(
    api_key=os.environ.get('COMET_API_KEY'),  
    project_name="ift6758-project",
    workspace="ift6758-project",
    auto_output_logging="simple",
)


# set an experiment name for basemodel
experiment_name = "xgb_all_features_grid_search1"  #base name for log_model, log_image
experiment.set_name(experiment_name)
#add tags
experiment.add_tags(['grid_search1'])

   
# Read in data and assign X and y
data = pd.read_csv('../../data/train.csv', index_col=0)
X = data[data.columns.tolist()[:-1]]
y = data[['isGoal']]


def XGB(X, y):
      
    #Train and valid split
    X_train,X_val,y_train,y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.values.ravel()
    
    # XGB Classifier
    model = xgb.XGBClassifier()  
    
    params = {
    'n_estimators': [100, 200, 500],
    'max_depth' : [3,6,10],
    'learning_rate': [0.01,0.05, 0.1],
    'booster': ['gbtree', 'gblinear', 'dart'],
    'gamma': [0, 0.5, 1],
    'reg_alpha': [0, 0.5, 1],
    'reg_lambda': [0.5, 1, 5],
    }
    
    gs = RandomizedSearchCV(estimator=model, 
                    param_distributions=  params, 
                    scoring = 'roc_auc', 
                    refit=True,
                    cv = 5,
                    verbose=10, 
                    n_iter = 100,
                    #n_jobs = -1
                    )
    
    gs.fit(X_train, y_train)
    
    print('Best score:', gs.best_score_)
    print('Best score:', gs.best_params_)
    
    # Predict on validation set
    y_pred = gs.predict(X_val)
    
    #Probability estimates
    pred_probs = gs.predict_proba(X_val)
    
    #Model Evaultion Metrics
    gs_best_score = gs.best_score_
    accuracy = metrics.accuracy_score(y_val, y_pred)
    f1_score = metrics.f1_score(y_val, y_pred)
    precision = metrics.precision_score(y_val, y_pred)
    recall = metrics.recall_score(y_val, y_pred)
    cf_matrix = metrics.confusion_matrix(y_val,y_pred)
    

    # save the model to disk
    filename = experiment_name + '.pkl'
    pickle.dump(gs, open(filename, 'wb'))
    
    
    
    metrics_dict = { 'Grid_search_BestScore' :  gs_best_score,
                    'accuracy': accuracy,
                    "f1_score": f1_score,
                    "precision": precision,
                    "recall": recall,
                    "roc_auc": roc_auc,
                    
                    }

    #experiment.log_dataset_hash(X_train)
    experiment.log_parameters(params)
    experiment.log_metrics(metrics_dict)
    experiment.log_confusion_matrix(matrix=cf_matrix)
    experiment.log_model(experiment_name, filename)
       
    return pred_probs, accuracy,f1_score, precision, recall, roc_auc, cf_matrix, gs_best_score
    

if __name__ == '__main__':
    
    pred_probs, accuracy,f1_score, precision, recall, roc_auc, cf_matrix = XGB(X, y)
    print(gs_best_score, accuracy,f1_score, precision, recall, roc_auc )
    print(cf_matrix)

    
    


