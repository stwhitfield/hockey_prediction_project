#!/usr/bin/env python
# coding: utf-8
from comet_ml import Experiment
import os 
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from plot_metrics import *
import pickle

# Create an experiment with your api key
experiment = Experiment(
    api_key=os.environ.get('COMET_API_KEY'),  
    project_name="ift6758-project",
    workspace="ift6758-project",
    auto_output_logging="simple",
)


# set an experiment name for basemodel
experiment_name = "xgb_featSel_rfe_best_model_params"  #base name for log_model, log_image
experiment.set_name(experiment_name)
#add tags
experiment.add_tags(['rfe_feature_selec', 'grid_best_params'])

# Read in data and assign X and y
data = pd.read_csv('../../data/train.csv', index_col=0)
X = data[data.columns.tolist()[:-1]]
y = data[['isGoal']]
X


#RFE Feature Selection
X = X.copy()

col_list = X.columns.to_list()
feature_names = np.array(X.columns.to_list())
print("Column Features",feature_names)

tot_col = len(X.columns)

min_max_scaler = MinMaxScaler()
X[col_list] = min_max_scaler.fit_transform(X[col_list])

model = xgb.XGBClassifier()

rfe = RFE(model, n_features_to_select=15)
fit = rfe.fit(X, y)

print("Num Features: ", fit.n_features_)

#Selected feature names
selected_features = feature_names[fit.support_]
print('Selected feature names: ', selected_features)

features_dropped = set(col_list).difference(selected_features)
print('features_dropped: ', features_dropped)

#New X after selecting features through RFE
X_new = X[selected_features]
print('Shape of new X', X_new.shape)


#Training on X on selected features using XGB saved model best parameters from grid search 
saved_model = "../Task_5_2/xgb_all_features_grid_search1.pkl"
xgb_saved_model = joblib.load(saved_model)
xgb_best_params = xgb_saved_model.best_params_
xgb_best_params


###############################################################################
def XGB_best_params(X,y, params):
    #Train and valid split
    X_train,X_val,y_train,y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGB Classifier
    clf = xgb.XGBClassifier(**params)

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
    experiment.log_parameters(xgb_best_params)
    experiment.log_metrics(metrics_dict)
    experiment.log_confusion_matrix(matrix=cf_matrix)
    experiment.log_image('roc_curve.png', name= experiment_name + '_roc_curve.png', overwrite=True)
    experiment.log_image('goal_rate_plot.png', name= experiment_name + '_goal_rate_plot.png', overwrite=True)
    experiment.log_image('cumulative_goal_rate.png', name= experiment_name + '_cumulative_goal_rate_plot.png', overwrite=True)
    experiment.log_image('calibration_curve.png', name= experiment_name + '_calibration_curve.png', overwrite=True)
    experiment.log_model(experiment_name, saved_model)
    
    return pred_probs, accuracy,f1_score, precision, recall, roc_auc, cf_matrix

###############################################################################

pred_probs, accuracy,f1_score, precision, recall, roc_auc, cf_matrix = XGB_best_params(X_new, y, xgb_best_params)
print(f' accuracy: {accuracy}')
print(f' f1_score: {f1_score}')
print(f' precision: {precision}')
print(f' recall: {recall}')
print(f' roc_auc: {roc_auc}')
print(f' Confusin Matrix')
print(cf_matrix)




# In[ ]:




