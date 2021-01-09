## Preprocessing
from sklearn.preprocessing import StandardScaler
from preprocessing import *

## Modeling
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb

from tensorflow import keras
from tensorflow.keras import models, layers, initializers

## Saving
from joblib import dump, load
import pickle


def save_model(trained_model, model_type, keras, params = ''):
    timestamp = d.datetime.now().strftime("%d-%m-%Y(%H:%M:%S)")
    file_name = f'models/{model_type}_{params}{timestamp}.model'
    if keras:
        trained_model.save(file_name)
    else :
        dump(trained_model, file_name)
    print("Saved:", file_name)

def build_ann(n_layers, width):
    """
    Return a compiled keras ANN with n_layers layers and width neurons by layer
    """
    model = models.Sequential()
    # hidden layers
    for i in range(n_layers):
        model.add(layers.Dense(width))

    #output layers
    model.add(layers.Dense(1, activation = "linear"))
    # Optimisation parameters
    metrics = ["mse"]
    model.compile(optimizer = 'adam', loss='mse', metrics = metrics)
    return model

def train_ann(X_train, y_train, model, epochs) :
    """
    Run the training procedure and return the trained model, 
    """
    history = model.fit(X_train, 
                        y_train, 
                        epochs = epochs,
                        validation_split=0.2, 
                        verbose = 2)
    return history, model


def ann(X_train, y_train, n_layers, width, epochs):

    model = build_ann(n_layers, width)
    scaler = StandardScaler() 
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    history, trained_model = train_ann(X_train_scaled, y_train, model, epochs)
    params = f"({n_layers}x{width}x{epochs}epochs)_"
    save_model(trained_model, "ann", True, params)
    plt.plot(history.history["loss"])
    plt.show()

def reglin(X_train, y_train):
    trained_model = LinearRegression().fit(X_train, y_train)
    save_model(trained_model, "reglin", False, "")

def random_forest(X_train, y_train, n_estimators, params):
    print("> Model type : Random forest")
    trained_model = RandomForestRegressor(n_estimators=n_estimators, random_state=0, min_samples_leaf=10, n_jobs=-1)
    trained_model.fit(X_train, y_train)
    save_model(trained_model, "random_forest", False, params)

def extra_trees(X_train, y_train, n_estimators, params):
    print("> Model type : Extra Trees")
    trained_model = ExtraTreesRegressor(n_estimators=n_estimators, random_state=0, min_samples_leaf=20, n_jobs=-1)
    trained_model.fit(X_train, y_train)
    save_model(trained_model, "xtrees", False, params)
    return trained_model

def xgboosting(X_train, y_train, n_estimators, params):
    print("> Model type : XGBoost")
    model = xgb.XGBModel(objective='reg:squarederror',
                                    max_depth=11,
                                    subsample=0.5,
                                    colsample_bytree=0.5,
                                    learning_rate=0.1,
                                    n_estimators=n_estimators,
                                    verbosity=0,
                                    seed=42)
    power_lines = y_train.columns
    trained_models = {}
    for pl in power_lines:
        print(">> Fitting", pl)
        trained_model = model.fit(X_train, y_train[pl])
        trained_models[pl] = trained_model
    save_model(trained_models, "xgboost", False, params)

def get_importance_features(X_train, y_train, n_estimators, params):
    model = extra_trees(X_train, y_train, n_estimators, params)
    imp = model.feature_importances_
    indices = np.argsort(imp)[::-1]
    importance = X_train.columns[indices]
    dump(importance, "importance")
    print(importance[:15])

def run_test(datareader, delay, importance, nb_features, n_estimators):
## Data init
    X_train, X_test, y_train, y_test = generate_train_data("chrono", datareader)
    if delay:
        X_train = add_delays(X_train, 4)
    if importance:
        importance_tab = load("importance")
        X_train = X_train[importance_tab[:nb_features]]
    n,p = X_train.shape

    ## Output
    delay_str = "delay_" if delay else ""
    datareader_str = "datareader_" if datareader else ""
    importance_str = f"{nb_features}best_features_" if importance else ""
    params = f"{n_estimators}estimators_{p}variables_{datareader_str}{importance_str}{delay_str}"

    print("Training model : "), 
    print("> Number of variables :", p)
    print("> Delay :", delay)
    print("> Importance :", importance)
    if importance:
        print("> Nb features :", nb_features)
    print("> n_estimators :", n_estimators)

    ## Model
    # get_importance_features(X_train, y_train, 500, params)
    for n in [50,100,200,500]:
        extra_trees(X_train, y_train, n, params)
        xgboosting(X_train, y_train, n, params)
        random_forest(X_train, y_train, n, params)

if __name__ == "__main__":
    ## Test 1 
    datareader = True
    delay = True
    importance = False
    nb_features = 40
    n_estimators = 5
    run_test(datareader, delay, importance, nb_features, n_estimators)

    ## Test 2
    importance = True
    run_test(datareader, delay, importance, nb_features, n_estimators)
