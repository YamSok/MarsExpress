## Preprocessing
from sklearn.preprocessing import StandardScaler
from preprocessing import *

## Modeling
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
from tensorflow.keras import models, layers, initializers

## Saving
from joblib import dump, load


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
    print("Training model : ")
    print("> Number of variables :", p)
    print("> Delay :", delay)
    print("> n_estimators :", n_estimators)
    print("> Model type : Random forest")
    trained_model = RandomForestRegressor(n_estimators=n_estimators, random_state=0, min_samples_leaf=10, n_jobs=-1)
    trained_model.fit(X_train, y_train)

    save_model(trained_model, "random_forest", False, params)


def extra_trees(X_train, y_train, n_estimators, params):
    print("Training model : "), 
    print("> Number of variables :", p)
    print("> Delay :", delay)
    print("> n_estimators :", n_estimators)
    print("> Model type : Extra Trees")
    trained_model = RandomForestRegressor(n_estimators=n_estimators, random_state=0, min_samples_leaf=20, n_jobs=-1)
    trained_model.fit(X_train, y_train)
    save_model(trained_model, "xtrees", False, params)

if __name__ == "__main__":
    ## Params
    datareader = True
    delay = True
    n_estimators = 50

    ## Data init
    X_train, X_test, y_train, y_test = generate_train_data("chrono", datareader)
    if delay:
        X_train = add_delays(X_train, 4)
    n,p = X_train.shape
    delay_str = "delay_" if delay else ""
    datareader_str = "datareader_" if datareader else ""
    params = f"{n_estimators}estimators_{p}variables_{datareader_str}{delay_str}"

    ## Model
    extra_trees(X_train, y_train, n_estimators, params)

