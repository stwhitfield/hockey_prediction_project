#!/usr/bin/env python
# coding: utf-8

# Import comet_ml at the top of your file, before sklearn!
from comet_ml import Experiment
import os 
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from plot_metrics import *


# Create an experiment with your api key
experiment = Experiment(
    api_key=os.environ.get('COMET_API_KEY'),  
    project_name="ift6758-project",
    workspace="ift6758-project",
    auto_output_logging="simple",
)

# set an experiment name for basemodel
#experiment.set_name("test_log_reg_basemodel")
#add tags
#experiment.add_tags(['Distance', 'Default_Settings'])



# Read in data and assign X and y
data = pd.read_csv('fe_train_data.csv', index_col=0)
y = data[['isGoal']]
X = data.drop(columns = 'isGoal')
   

def XGB(X, y):
    
    feature_list = (['distanceFromNet'], ['angleFromNet'], ['distanceFromNet', 'angleFromNet']  )
    feature_name_list = ['distance', 'angle', 'distance_angle']
    
    #Select 0,1,2 for 'Distance from Net', 'Angle from Net', 'Distance and Angle from Net'features.
    #i = 1
    features = feature_list[i]
    feature_name = feature_name_list[i]
    
    # set an experiment name for basemodel
    experiment_name = "xgb_" + feature_name #base name for log_model, log_image
    experiment.set_name(experiment_name)
    #add tags
    experiment.add_tags([feature_name])
    

    #Train and valid split
    X_train,X_val,y_train,y_val = train_test_split(X[features], y, test_size=0.2, random_state=42)

    # Logistic regression model fitting
    clf = GradientBoostingClassifier()
    y_train = y_train.values.ravel()
    clf.fit(X_train, y_train)
    
    # Predict on validation set
    y_pred = clf.predict(X_val)
    
    #Probability estimates
    pred_probs = clf.predict_proba(X_val)
    
    #Model Evaultion Metrics
    accuracy = metrics.accuracy_score(y_val, y_pred)
    f1_score = metrics.f1_score(y_val, y_pred)
    precision = metrics.precision_score(y_val, y_pred)
    recall = metrics.recall_score(y_val, y_pred)
    cf_matrix = metrics.confusion_matrix(y_val,y_pred)
    
    #ROC AUC Curve
    probs_isgoal = pred_probs[:,1]
    roc_auc = metrics.roc_auc_score(y_val,probs_isgoal)
    plot_ROC(y_val, pred_probs)
    
    #Goal Rate Plot
    df_percentile =  calc_percentile(pred_probs, y_val)
    goal_rate_df = goal_rate(df_percentile)
    plot_goal_rates(goal_rate_df)
    
    #Cumulative Goal Rate Plot
    plot_cumulative_goal_rates(df_percentile)
    
    #Calibration Curve
    plot_calibration_curve_prediction(y_val, pred_probs)

    # save the model to disk
    filename = experiment_name + '.pkl'
    pickle.dump(clf, open(filename, 'wb'))
    
    #params = {}
    
    metrics_dict = { 'accuracy': accuracy,
                    "f1_score": f1_score,
                    "precision": precision,
                    "recall": recall,
                    "roc_auc": roc_auc}

    #experiment.log_dataset_hash(X_train)
    #experiment.log_parameters(params)
    experiment.log_metrics(metrics_dict)
    experiment.log_confusion_matrix(matrix=cf_matrix)
    experiment.log_image('roc_curve.png', name= experiment_name + '_roc_curve.png', overwrite=True)
    experiment.log_image('goal_rate_plot.png', name= experiment_name + '_goal_rate_plot.png', overwrite=True)
    experiment.log_image('cumulative_goal_rate.png', name= experiment_name + '_cumulative_goal_rate_plot.png', overwrite=True)
    experiment.log_image('calibration_curve.png', name= experiment_name + '_calibration_curve.png', overwrite=True)
    experiment.log_model(experiment_name, filename)
       
    return pred_probs, accuracy,f1_score, precision, recall, roc_auc, cf_matrix
    

if __name__ == '__main__':
    #Select 0,1,2 for 'Distance from Net', 'Angle from Net', 'Distance and Angle from Net'features.
    i = 2
    pred_probs, accuracy,f1_score, precision, recall, roc_auc, cf_matrix = XGB(X, y)
    print(accuracy,f1_score, precision, recall, roc_auc )
    print(cf_matrix)

    
    


