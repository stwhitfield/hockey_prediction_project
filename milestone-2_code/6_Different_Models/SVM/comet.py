#!/usr/bin/env python
# coding: utf-8
from comet_ml import Experiment
import os
from sklearn import metrics
from plot_metrics import *
from dotenv import load_dotenv

load_dotenv()

def push_best_model(X, y, model, experiment_name, model_name, tags=[]):
    # Predict on validation set
    y_pred = model.predict(X)
    #Probability estimates
    pred_probs = model.predict_proba(X)
    probs_isgoal = pred_probs[:,1]

    #Model Evaultion Metrics
    accuracy = metrics.accuracy_score(y, y_pred)
    f1_score = metrics.f1_score(y, y_pred)
    precision = metrics.precision_score(y, y_pred)
    recall = metrics.recall_score(y, y_pred)
    cf_matrix = metrics.confusion_matrix(y, y_pred)
    roc_auc = metrics.roc_auc_score(y, probs_isgoal)
    
    #ROC AUC Curve
    plot_ROC(y, pred_probs)
    y = pd.DataFrame(y, columns=["isGoal"])

    #Goal Rate Plot
    df_percentile = calc_percentile(pred_probs, y)
    goal_rate_df = goal_rate(df_percentile)
    plot_goal_rates(goal_rate_df)
 
    #Cumulative Goal Rate Plot
    plot_cumulative_goal_rates(df_percentile)
    
    #Calibration Curve
    plot_calibration_curve_prediction(y, pred_probs)   
    
    metrics_dict = { "accuracy": accuracy,
                        "f1_score": f1_score,
                        "precision": precision,
                        "recall": recall,
                        "roc_auc": roc_auc}

    experiment = Experiment(
        api_key=os.getenv('COMET_API_KEY'),  
        project_name="ift6758-project",
        workspace="ift6758-project",
        auto_output_logging="simple",
    )

    experiment.add_tags(tags)
    experiment.log_parameters(model.get_params())
    experiment.log_metrics(metrics_dict)
    experiment.log_confusion_matrix(matrix=cf_matrix)
    experiment.log_image('roc_curve.png', name= experiment_name + '_roc_curve.png', overwrite=True)
    experiment.log_image('goal_rate_plot.png', name= experiment_name + '_goal_rate_plot.png', overwrite=True)
    experiment.log_image('cumulative_goal_rate.png', name= experiment_name + '_cumulative_goal_rate_plot.png', overwrite=True)
    experiment.log_image('calibration_curve.png', name= experiment_name + '_calibration_curve.png', overwrite=True)
    experiment.log_model(experiment_name, model_name)

    return pred_probs, accuracy, f1_score, precision, recall, roc_auc, cf_matrix

