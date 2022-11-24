# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 22:01:49 2022

@author: rajes
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve, CalibrationDisplay
import matplotlib.ticker as ticker



def plot_ROC(y_val,pred_probs):
    """
    Plots an ROC curve for the given y (ground truth) and model probabilities, and calculates the AUC.
    """
    fpr, tpr, _ = roc_curve(y_val, pred_probs)
    roc_auc = auc(fpr,tpr)
    #plt.figure()
    plt.figure(figsize=(8,6))
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
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.title('ROC Curves', fontsize=16)
    plt.legend(loc="lower right")
    
    ax = plt.gca()
    ax.grid()
    ax.set_facecolor('0.95')
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.show()
    
def calc_percentile(pred_probs, y_val):
    
    #Create a df for shot probabilities
    df_probs = pd.DataFrame(pred_probs)
    df_probs = df_probs.rename(columns={0: "Not_Goal_prob", 1: "Goal_prob"})
    
    # Combining 'Goal Probability' and 'Is Goal' into one df. 
    df_probs = pd.concat([df_probs["Goal_prob"].reset_index(drop=True), y_val["isGoal"].reset_index(drop=True)],axis=1)
    
    # Computing and adding Percentile Column
    percentile_values=df_probs['Goal_prob'].rank(pct=True)
    df_probs['Percentile'] = percentile_values*100
    df_percentile = df_probs.copy()
    
    return df_percentile

def goal_rate(df_percentile):
   
    rate_list = []
    
    # Find total number of goals
    #total_goals = df_percentile['isGoal'].value_counts()[1]
   
    
    bin_width = 5

    i = 0
    i_list = []
    
    
    while i< (100-bin_width+1):  # 95 is the lower bound of last bin
        i_list.append(i)

        # i-th bin size
        bin_lower_bound = i
        bin_upper_bound = i + bin_width

        # finding rows have percentiles fall in this range
        bin_rows = df_percentile[(df_percentile['Percentile']>=bin_lower_bound) & (df_percentile['Percentile']<bin_upper_bound)]
        
        # Calculating the goal rate from total number of goals and shots in each bin_rows
        goals = bin_rows['isGoal'].value_counts()[1]      
        shots = len(bin_rows) #total shots in bin_rows
        rate = (goals/shots)*100 # goal rate in pecerntage

        rate_list.append(rate)

        i+=bin_width
    
    # Creating a new dataframe Combining goal rate list and percentile list 
    goal_rate_df = pd.DataFrame(list(zip(rate_list, i_list)),columns=['Rate', 'Percentile'])
    
    return goal_rate_df

def plot_goal_rates(goal_rate_df):
    plt.figure(figsize=(8,6))
    
    ax = plt.gca()
    ax.grid()
    
    ax.set_facecolor('0.95')
    x = goal_rate_df['Percentile']
    y = goal_rate_df['Rate']
    plt.plot(x,y)

    ax.set_ylim([0,100])
    ax.set_xlim([0,100])
    ax.invert_xaxis()
    major_ticks = np.arange(0, 110, 10)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    
    #ax.legend(['Model 1'])
    
    plt.xlabel('Shot probability model percentile', fontsize=16)
    plt.title('Goal Rate', fontsize=16)
    plt.ylabel('Goals / (Shots+Goals)%', fontsize=16)
    plt.savefig('goal_rate_plot.png')
    plt.show()
    
def plot_cumulative_goal_rates(df_percentile):
    
    plt.figure(figsize=(8,6))
    df_precentile_only_goal = df_percentile[df_percentile['isGoal'] == 1]
    
    ax = sns.ecdfplot(data=df_precentile_only_goal, x=100 - df_precentile_only_goal.Percentile)
    
    ax = plt.gca()
    ax.grid()
    ax.set_facecolor('0.95')
    plt.yticks(np.arange(0, 1.05, 0.1))
    plt.xticks(np.arange(0, 100 * 1.01, 10))
    xvals = ax.get_xticks()
    ax.set_xticklabels(100 - xvals.astype(np.int32), fontsize=16)
    yvals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(y) for y in yvals], fontsize=16)
    ax.set_xlabel('Shot probability model percentile', fontsize=16)
    ax.set_ylabel('Proportion', fontsize=16)
    ax.set_title(f"Cumulative % of Goals", fontsize=16)
    #plt.legend(loc='lower right')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.savefig('cumulative_goal_rate.png')
    ax.legend(['Neural Network'])
    plt.show()
    
def plot_calibration_curve_prediction(y_val, pred_probs):
    plt.figure(figsize=(8,6))
    
    ax = CalibrationDisplay.from_predictions(y_val['isGoal'],pred_probs, n_bins=50)
   
    ax = plt.gca()
    ax.grid()
    ax.set_facecolor('0.95')
    ax.set_title(f"Calibration Curve", fontsize=16)
    plt.ylabel('Fraction of positives', fontsize=16)
    plt.xlabel('Mean predicted probability', fontsize=16)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.savefig('calibration_curve.png')
    plt.show()
