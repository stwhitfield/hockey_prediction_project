#!/usr/bin/env python
# coding: utf-8

# Author: Shawn Whitfield <br>
# Version: 1 <br>
# Date: 2022-10-07 <br> 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Read in data and assign X and y
data = pd.read_csv('fe_train_data.csv', index_col=0)
y = data['isGoal']
X = data.drop(columns = 'isGoal')


# Using training dataset, create a training and validation split


from sklearn.model_selection import train_test_split
# Standard 80:20 split train:validation
X_train,X_val,y_train,y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Using only the distance feature, train a Logistic Regression classifier with the completely default settings


# Select only the distance feature and format it for the classifier
X_train_d = X_train['distanceFromNet'].to_numpy().reshape(-1,1)
X_val_d = X_val['distanceFromNet'].to_numpy().reshape(-1,1)


# Train the classifier and get predictions
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=42)
clf.fit(X_train_d,y_train)
clf_preds = clf.predict(X_val_d)
score = clf.score(X_val_d,y_val)
print(f'accuracy score: {score}')



# Look at where it's all gone wrong
X_val_compar = X_val.copy()
X_val_compar['preds'] = clf_preds
X_val_compar['actual'] = y_val
wrong_preds = X_val_compar[X_val_compar['preds'] != y_val]
wrong_preds


print(wrong_preds.describe())


# The model systematically underestimates goals ie. it fails to predict goals. <br>
# All of the actual are 1 while all the predicted are 0.


# Get probabilities associated with each class, for each point in X_val_d
probs_d = clf.predict_proba(X_val_d)


def plot_ROC(y_val,probs,title = False, savename=False):
    """
    Plots an ROC curve for the given y (ground truth) and model probabilities, and calculates the AUC.
    """
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_val,probs)
    roc_auc = auc(fpr,tpr)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    #Include a random classifier baseline, i.e. each shot has a 50% chance of being a goal
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title}")
    plt.legend(loc="lower right")

    plt.tight_layout()
    if savename:
        plt.savefig(f'{savename}.png')
    plt.show()
    plt.clf()


plot_ROC(y_val, probs_d[:,1], 'ROC curve for distance')


#TODO:

# The goal rate (#goals / (#no_goals + #goals)) as a function of the shot probability 
#     model percentile, i.e. if a value is the 70th percentile, it is above 70% of the data.


# The cumulative proportion of goals (not shots) as a function of the shot probability model percentile.



# The reliability diagram (calibration curve). Scikit-learn provides functionality to create 
# a reliability diagram in a few lines of code; check out the CalibrationDisplay API 
# (specifically the .from_estimator() or .from_predictions() methods) for more information.
from sklearn.calibration import calibration_curve, CalibrationDisplay


