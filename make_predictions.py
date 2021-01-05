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

def generate_predictions(file_name, X_test,model_type):
    print("Loading", file_name)
    query = f"models/*{model_type}*"
    if len(glob.glob(query + "/**")) != 0: ## means : Has subfolder == True
        model = models.load_model(file_name)
        print("Keras")
    else:
        model = load(file_name)
    print("> Loaded")
    predict = model.predict(X_test)
    return predict

def main(model_type):

    file_name = get_model_file_name(model_type = "")
    datareader = "datareader" in file_name
    delay = "delay" in file_name
    
    X_train, X_test, y_train, y_test = generate_train_data("chrono", datareader)
    if delay:
        X_test = add_delays(X_test, 4)

    pred = generate_predictions(file_name, X_test, model_type="")
    power_ids = y_train.columns[y_train.columns.str.match("NPWD")]

    predict = pd.DataFrame(pred, columns = power_ids )
    y_test = y_test.copy()
    predict.index = y_test.index

    timestamp = d.datetime.now().strftime("%d-%m-%Y(%H:%M:%S)")
    params = "_".join(file_name.split("_")[:-1]).split("/")[1]
    predict_file_name = f"results/predictions_{params}_{timestamp}.p"
    predict.to_pickle(predict_file_name)
    print("Saved predictions :", predict_file_name)

parser = argparse.ArgumentParser()

parser.add_argument("-m", help="Type of model (ex: random_forest, xtrees ...)", required=False)

args = parser.parse_args()

main(args.m)