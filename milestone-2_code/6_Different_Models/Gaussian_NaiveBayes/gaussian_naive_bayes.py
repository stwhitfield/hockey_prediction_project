#!/usr/bin/env python
# coding: utf-8

# Import comet_ml at the top of your file, before sklearn!
from comet_ml import Experiment
import os 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from plot_metrics import *
import joblib
from sklearn.naive_bayes import GaussianNB
import pickle

# Create an experiment with your api key
experiment = Experiment(
    api_key=os.environ.get('COMET_API_KEY'),  
    project_name="ift6758-project",
    workspace="ift6758-project",
    auto_output_logging="simple",
)


# set an experiment name for basemodel
experiment_name = "gaussian_naive_bayes"  #base name for log_model, log_image
experiment.set_name(experiment_name)
#add tags
experiment.add_tags(['gaussian_naive_bayes'])

   
# Read in data and assign X and y
data = pd.read_csv('../../data/train.csv', index_col=0)
X = data[data.columns.tolist()[:-1]]
y = data[['isGoal']]


#Train and Validation Split
X_train,X_val,y_train,y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#Model Fitting
clf = GaussianNB()
y_train = y_train.values.ravel()
clf.fit(X_train, y_train)
    
# Predict on validation set
y_pred = clf.predict(X_val)

#Probability estimates
pred_probs = clf.predict_proba(X_val)
probs_isgoal = pred_probs[:,1]
    
#Model Evaultion Metrics
accuracy = metrics.accuracy_score(y_val, y_pred)
f1_score = metrics.f1_score(y_val, y_pred)
precision = metrics.precision_score(y_val, y_pred)
recall = metrics.recall_score(y_val, y_pred)
cf_matrix = metrics.confusion_matrix(y_val,y_pred)
roc_auc = metrics.roc_auc_score(y_val,probs_isgoal)

#ROC AUC Curve
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

metrics_dict = { 'accuracy': accuracy,
                    "f1_score": f1_score,
                    "precision": precision,
                    "recall": recall,
                    "roc_auc": roc_auc}

#experiment.log_dataset_hash(X_train)
#experiment.log_parameters(dt_best_params)
experiment.log_metrics(metrics_dict)
experiment.log_confusion_matrix(matrix=cf_matrix)
experiment.log_image('roc_curve.png', name= experiment_name + '_roc_curve.png', overwrite=True)
experiment.log_image('goal_rate_plot.png', name= experiment_name + '_goal_rate_plot.png', overwrite=True)
experiment.log_image('cumulative_goal_rate.png', name= experiment_name + '_cumulative_goal_rate_plot.png', overwrite=True)
experiment.log_image('calibration_curve.png', name= experiment_name + '_calibration_curve.png', overwrite=True)
experiment.log_model(experiment_name, filename)
    


    
    

