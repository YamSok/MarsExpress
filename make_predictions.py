import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from joblib import dump, load
from preprocessing import *
from sklearn.cluster import KMeans
from tensorflow.keras import models
from preprocessing import add_delays

import warnings
warnings.filterwarnings('ignore')

import argparse


def get_model_file_name(model_type):
    query = f"models/*{model_type}*"
    query_list = glob.glob(f"models/*{model_type}*")
    if len(query_list) > 0:
        query_list_formated = [s.split("models/")[1] for s in query_list]
        query_list_formated_numbered  = [f'{i} - {query_list_formated[i]}' for i in range(len(query_list_formated))]
        for c in query_list_formated_numbered:
            print(c)
        cin = input("Selection : ")
        file_name = query_list[int(cin)]
    else : 
        file_name = query_list[0]
    
    return file_name

def generate_predictions(file_name, X_test, powerlines):
    print("> Loading", file_name)
    
    model = load(file_name)
    if "xgboost" in file_name:
        # 33 models
        predictions = {}
        for pl in powerlines:
            model_i = model[pl]
            prediction = model_i.predict(X_test)
            predictions[pl] = prediction
        predict = pd.DataFrame.from_dict(predictions)
    else: 
        predict = model.predict(X_test)
    print("> Loaded")

    return predict

def main(file_name):

    # file_name = get_model_file_name(model_type = model_type)
    # Parsing
    datareader = "datareader" in file_name
    delay = "delay" in file_name
    importance = "best" in file_name
    if importance:
        nb_features = int(file_name.split('_')[2].split("v")[0])
    params = "_".join(file_name.split("_")[:-1]).split("/")[1]

    print("> Making prediction")
    print("> Model :", params)

    #Data preparation
    X_train, X_test, y_train, y_test = generate_train_data("chrono", datareader)
    if delay:
        X_test = add_delays(X_test, 4)
    if importance:
        importance_tab = load("importance")
        X_test = X_test[importance_tab[:nb_features]]
    
    # Prediction
    power_ids = y_train.columns[y_train.columns.str.match("NPWD")]
    pred = generate_predictions(file_name, X_test, power_ids)
    predict = pd.DataFrame(pred, columns = power_ids )
    y_test = y_test.copy()
    predict.index = y_test.index

    # Output
    # timestamp = d.datetime.now().strftime("%d-%m-%Y(%H:%M:%S)")
    predict_file_name = f"results/predictions_{params}.p"
    predict.to_pickle(predict_file_name)
    print("> Predictions DataFrame saved :", predict_file_name)

# parser = argparse.ArgumentParser()

# parser.add_argument("-m", default="", help="Type of model (ex: random_forest, xtrees ...)", required=False)

# args = parser.parse_args()

file_names = glob.glob("models/*")
for file_name in file_names:
    main(file_name)
    print("~~~"*15)