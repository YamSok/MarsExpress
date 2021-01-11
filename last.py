import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from joblib import dump, load
from preprocessing import *
from models import *
from sklearn.cluster import KMeans
from tensorflow.keras import models
import xgboost as xgb

from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor


import warnings
warnings.filterwarnings('ignore')

def get_estimator(estimator):
    if estimator == 'booster':
        e = xgb.XGBModel(objective='reg:squarederror',
                                 max_depth=11,
                                 subsample=0.5,
                                 colsample_bytree=0.5,
                                 learning_rate=0.1,
                                 n_estimators=100,
                                 silent=1,
                                 seed=42)
                
    elif estimator == 'xtrees':
        e = ExtraTreesRegressor(n_estimators=100,
                                      random_state=0,
                                      min_samples_leaf=20,
                                      n_jobs=-1)
    elif estimator == 'rf':
        e=RandomForestRegressor(n_estimators=100,
                                        random_state=1,
                                        min_samples_leaf=10,
                                        n_jobs=-1)
    return e

def get_trained_estimator(estimator, X, y):
    e = get_estimator(estimator)
    e.fit(X,y)
    return e

def get_all_models(estimators, power_lines, X_list, train_all, feature_dict=None, feat_no=None):
    models = {}
    for w in power_lines:
        models[w] = []
        features = list(X_list[0].columns)
        if feature_dict is not None:
            features=feature_dict[w]
        for estimator in estimators:
            for X in X_list:
                e=get_trained_estimator(estimator,
                                         X[features[:feat_no]].astype(float),
                                         train_all.loc[X.index][w].astype(float))
                print(estimator, w)
                models[w].append(e)
    return models


X_train, X_test, y_train, y_test = generate_train_data("chrono", True)
X_train = add_delays(X_train, 4)
X_test = add_delays(X_test, 4)

train_all = import_data()

features = X_train.columns
power_lines = y_train.columns

try:
    # Check if pickle is already there
    importances = pickle.load( open( "importances.pickle", "rb" ) ) 
except IOError:
    # Get Extra Trees models, one per power line, just to see feature importances
    models=get_all_models(['xtrees'], power_lines, [X_train[features]], train_all)
    imp_per_w = {}
    
    # Get importances
    for i in power_lines:
        imp = models[i][0].feature_importances_
        imp_per_w[i] = []
        indices = np.argsort(imp)[::-1]
        for f in range(X_train[features].shape[1]):
            imp_per_w[i].append(list(X_train.columns)[indices[f]])
            
    # Sorted Feature importances are dumped to a pickle file
    pickle.dump( imp_per_w, open( "importances.pickle", "wb" ) )
    importances = imp_per_w

models=get_all_models(['xtrees','booster'], power_lines, [X_train], train_all, importances, 40)
predictions  = {}
submission = y_test.copy()
submission.index = y_test.index
for w in power_lines:
    predictions[w] = X_test[['ut']]
    for idx, est in enumerate(models[w]):
        m = "model_%d"%idx
        predictions[w][m]=est.predict(X_test[importances[w][:40]].astype(float))
    submission[w] = predictions[w][[x for x in predictions[w].columns if 'model' in x]].mean(axis=1)

n_estimators = 500
timestamp = d.datetime.now().strftime("%d-%m-%Y(%H:%M:%S)")
delay_str = "delay_" 
datareader_str = "datareader_" 
importance_str = f"{40}best_features_" 
params = f"{n_estimators}estimators_{40}variables_{datareader_str}{importance_str}{delay_str}_test"

submission.to_pickle(f"results/predictions_{param}.p")