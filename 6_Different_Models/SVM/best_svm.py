# from comet_ml import Experiment
import os 
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from scipy import stats
from sklearn import svm
from comet import push_best_model
from plot_metrics import *

def getData(dataset="train"):
  data = pd.read_csv('../../data/' + dataset + '.csv')
  data = data.dropna()
  x_train = data.iloc[:,0:20]
  x_train = x_train.drop(["gameId"], axis=1)
  cols = ["shotType", "lastEventType"]
  x_train[cols] = x_train[cols].astype('category')
  x_train["shotType"] = x_train['shotType'].cat.rename_categories({k: v for v, k in enumerate(np.array(pd.Categorical(x_train["shotType"]).categories))})
  x_train["lastEventType"] = x_train['lastEventType'].cat.rename_categories({k: v for v, k in enumerate(np.array(pd.Categorical(x_train["lastEventType"]).categories))})
  x_train[cols] = x_train[cols].astype('int')
  y_train = data.iloc[:,-1:].to_numpy().flatten()
  return (x_train, y_train)

# from best_svm import SVM, getData
# x_train, y_train = getData()
# ludo = SVM(x_train, y_train)

def SVM(X, y, experiment_name="svm"):
    #Train and valid split
    X_train,X_val,y_train,y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # SVM Classifier
    # model = svm.SVC()
    gs = svm.SVC(probability=True)

    # params = {
    #     'C': stats.expon(scale=100), 
    #     'gamma': stats.expon(scale=.1),
    #     'kernel': ['rbf'], 
    #     'class_weight':['balanced']
    # }

    # gs = RandomizedSearchCV(
    #                         estimator = model, 
    #                         param_distributions = params, 
    #                         scoring = 'recall', 
    #                         refit = True,
    #                         cv = 3,
    #                         verbose = 10, 
    #                         n_iter = 25,
    #                         # n_iter = 2,
    #                         n_jobs = -1)
    
    gs.fit(X_train, y_train)

    # save the model to disk
    filename = '../model/' + experiment_name + '_best_prob.pkl'
    pickle.dump(gs, open(filename, 'wb'))
    
    # print('Best score:', gs.best_score_)
    # print('Best score:', gs.best_params_)
    
    # Predict on validation set
    y_pred = gs.predict(X_val)
    
    #Probability estimates
    # pred_probs = gs.predict_proba(X_val)
    
    #Model Evaultion Metrics
    # gs_best_score = gs.best_score_sad
    accuracy = metrics.accuracy_score(y_val, y_pred)
    f1_score = metrics.f1_score(y_val, y_pred)
    precision = metrics.precision_score(y_val, y_pred)
    recall = metrics.recall_score(y_val, y_pred)
    roc_auc = metrics.roc_auc_score(y_val, y_pred)
    cf_matrix = metrics.confusion_matrix(y_val,y_pred)
    metrics_dict = { 
                    # 'Grid_search_BestScore' :  gs_best_score,
                    'accuracy': accuracy,
                    "f1_score": f1_score,
                    "precision": precision,
                    "recall": recall,
                    "roc_auc": roc_auc,
                    }
    print(metrics_dict)

    # experiment.log_parameters(params)
    # experiment.log_metrics(metrics_dict)
    # experiment.log_confusion_matrix(matrix=cf_matrix)
    # experiment.log_model(experiment_name, filename)
       
    # return pred_probs, accuracy, f1_score, precision, recall, roc_auc, cf_matrix, gs_best_score
    return accuracy, f1_score, precision, recall, roc_auc, cf_matrix


if __name__ == '__main__':
  x_train, y_train = getData()
  svc = SVM(x_train, y_train)
  # print("Pushing model...")
  # res = push_best_model(x_train, y_train, "best_svm_1", "../model/svm_best_1.pkl", ["svm", "grid_search_best_model"])
