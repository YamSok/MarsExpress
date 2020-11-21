import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import models, layers, initializers
import datetime as d

def import_file(file_name, file_type):
    data = pd.read_csv(file_name, sep=",", index_col=0) # index -> ut_ms
    data.index = pd.to_datetime(data.index, unit='ms') # ut_ms -> datetime

    if file_type == "dmop":
        data['commande'] = data.apply(lambda x : x['subsystem'][:4], axis=1)
        #data['len'] = data.apply(lambda x : len(x['subsystem'].split(".")), axis=1)
        data = pd.get_dummies(data[["commande"]]) #One Hot Encoding (naïf)
        data = data.resample('60min').pad()

    elif file_type == "ftl":
        data = pd.get_dummies(data[["type", "flagcomms"]]) #One Hot Encoding (naïf)
        data["flagcomms"] = data["flagcomms"].astype(int) # Passer les true/false en 1/0
        data = data.resample('60min').pad()
    else:
        data = data.resample('60min').mean()
    return data


## import de la première année d'une classe de variables quantitative (saaf, ltdata)
def import_all_files(folder, file_types, verbose):
    
    """
    Import all files for one martian year and apply specific preprocessing regarding variable types in the file.
    Return a list of DataFrames
    """
    
    df_list = []
    for file_type in file_types:
        print('Importing', file_type, "data ...") if verbose else 0
        query = f"{folder}*{file_type}*"
        file_name_list = glob.glob(query) 
        if len(file_name_list) > 1 :
            file_name_list_sorted = np.sort(file_name_list)
            df_3years = []
            for file_name in file_name_list_sorted:
                print('>', file_name) if verbose else 0
                data = import_file(file_name, file_type)
                df_3years.append(data)

            data = pd.concat(df_3years)
        else : 
            file_name = file_name_list[0]
            print('>', file_name) if verbose else 0
            data = import_file(file_name, file_type)
        df_list.append(data)
        print("Done.") if verbose else 0

    return df_list
    

def interpolate(df_list, verbose):
    """
    Concatenate a list of DataFrames, sort by date and interpolate (linearly or pad interpolation.
    """
    
    print("Concatenating ...") if verbose else 0
    combined = pd.concat(df_list, axis=1, sort=False)
    print(f"-> Concatenate done, {len(combined)} rows.\n Sort by date ...")if verbose else 0
    #combined = combined.sort_values("date").reset_index(drop=True)
    print("-> Sort done")if verbose else 0
    col = list(combined.columns)
    print("Interpolating ...")if verbose else 0
    for c in col:
        if c.split('_')[0] == "type" or c == "flagcomms" or c == "commande" : #categorical data
            combined[c] = combined[c].interpolate(method="ffill")
        else : #quantitative data
            combined[c] = combined[c].interpolate(method="linear")
    print("Data interpolated")if verbose else 0
    print("Traitement des NaN")if verbose else 0
    combined = combined.fillna(method ="bfill")
    combined.dropna(inplace=True)
    print("Done.")if verbose else 0

    return combined

def generate_data(train_folder, test_folder, verbose):
    """
    Generate 3 years DataFrames for training and testing set
    """
    
    file_types = ["saaf", "dmop", "ltdata", "ftl"]
    
    print("@@@@@@"*4) if verbose else 0
    print("Creating train dataset") if verbose else 0
    print("@@@@@@"*4) if verbose else 0
    df_list_train = import_all_files(train_folder, ["power"] + file_types, verbose)
    print("@@@@@@"*4) if verbose else 0
    print("Creating test dataset") if verbose else 0
    print("@@@@@@"*4) if verbose else 0
    df_list_test = import_all_files(test_folder, file_types, verbose)
    
    print("@@@@@@"*4) if verbose else 0
    print("Interpolating train dataset") if verbose else 0
    print("@@@@@@"*4) if verbose else 0
    train = interpolate(df_list_train, verbose) 
    print("@@@@@@"*4) if verbose else 0
    print("Interpolating test dataset") if verbose else 0
    print("@@@@@@"*4) if verbose else 0
    test = interpolate(df_list_test, verbose)
    
    to_del_test = np.setdiff1d(test.columns, train.columns[33:])
    to_del_train = np.setdiff1d(train.columns[33:],test.columns)
    test.drop(to_del_test, inplace=True, axis=1)
    train.drop(to_del_train, inplace=True, axis=1)
    timestamp = d.datetime.now().strftime("%d-%m-%Y(%H:%M:%S)")
    train.to_pickle(f"data/train_{timestamp}.p")
    test.to_pickle(f"data/test_{timestamp}.p")
    return train, test

def import_data():
    train_list = glob.glob(f"data/train*") 
    train_path = train_list[0]
    test_list = glob.glob(f"data/test*") 
    test_path = train_list[0]
    train = pd.read_pickle(train_path)
    test = pd.read_pickle(test_path)
    return train, test
    
def generate_train_data(method):
    train, test = import_data()

    power_ids = train.columns[train.columns.str.match("NPWD")]
    X = train.copy()
    X.drop(list(power_ids), inplace = True, axis = 1) # drop power
    print(np.shape(X))
    y = train.copy()[list(power_ids)]
    print(np.shape(y))
    if method == "chrono":
        train_start_date = "2008"
        train_end_date = "2012-05-27"
        test_start_date = "2012-05-28"
        test_end_date = "2014-04-14"
        mask_train = (train.index > train_start_date) & (train.index <= train_end_date)
        mask_test = (train.index > test_start_date) & (train.index <= test_end_date)
        X_train = X.loc[mask_train]
        y_train = y.loc[mask_train]
        X_test = X.loc[mask_test]
        y_test = y.loc[mask_test]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    TRAIN_FOLDER = "../data_MEX/train_set/"
    TEST_FOLDER = "../data_MEX/test_set/"
    generate_data(TRAIN_FOLDER, TEST_FOLDER, True)
