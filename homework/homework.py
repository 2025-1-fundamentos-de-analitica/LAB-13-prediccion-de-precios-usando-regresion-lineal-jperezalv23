#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import os
import json
import gzip
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error


# Cargar datos
df_entreno = pd.read_csv("./files/input/train_data.csv.zip", index_col=False, compression="zip")
df_test = pd.read_csv("./files/input/test_data.csv.zip", index_col=False, compression="zip")

df_entreno["Age"] = 2021 - df_entreno["Year"]
df_test["Age"] = 2021 - df_test["Year"]

df_entreno.drop(columns=["Year", "Car_Name"], inplace=True)
df_test.drop(columns=["Year", "Car_Name"], inplace=True)

X_train = df_entreno.drop(columns=["Present_Price"])
y_train = df_entreno["Present_Price"]

X_test = df_test.drop(columns=["Present_Price"])
y_test = df_test["Present_Price"]

columnas_categoricas = ["Fuel_Type", "Selling_type", "Transmission"]
columnas_numericas = [col for col in X_train.columns if col not in columnas_categoricas]

preprocesado = ColumnTransformer([
    ("cat", OneHotEncoder(), columnas_categoricas),
    ("scaler", MinMaxScaler(), columnas_numericas)
])

pipeline_modelo = Pipeline([
    ("preprocessor", preprocesado),
    ("feature_selection", SelectKBest(f_regression)),
    ("classifier", LinearRegression())
])

param_grid = {
    "feature_selection__k": range(1, 12),
    "classifier__fit_intercept": [True, False],
    "classifier__positive": [True, False],
}

buscador = GridSearchCV(
    estimator=pipeline_modelo,
    param_grid=param_grid,
    cv=10,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    refit=True,
    verbose=1,
)

buscador.fit(X_train, y_train)

# Evaluación
pred_entreno = buscador.predict(X_train)
pred_test = buscador.predict(X_test)

metricas_entreno = {
    "type": "metrics",
    "dataset": "train",
    "r2": r2_score(y_train, pred_entreno),
    "mse": mean_squared_error(y_train, pred_entreno),
    "mad": median_absolute_error(y_train, pred_entreno),
}

metricas_test = {
    "type": "metrics",
    "dataset": "test",
    "r2": r2_score(y_test, pred_test),
    "mse": mean_squared_error(y_test, pred_test),
    "mad": median_absolute_error(y_test, pred_test),
}

os.makedirs("files/output/", exist_ok=True)
with open("files/output/metrics.json", "w") as archivo_salida_metricas:
    archivo_salida_metricas.write(json.dumps(metricas_entreno) + "\n")
    archivo_salida_metricas.write(json.dumps(metricas_test) + "\n")

# Guardar modelo entrenado
os.makedirs("files/models/", exist_ok=True)
with gzip.open("files/models/model.pkl.gz", "wb") as archivo_salida:
    pickle.dump(buscador, archivo_salida)
