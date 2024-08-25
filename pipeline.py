import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    OutlierHandler:
    Esta clase maneja los outliers en las columnas del DataFrame.
    Utiliza el rango intercuartílico (IQR) para determinar los límites inferiores y superiores
    para luego reemplazar cualquier valor fuera de estos límites por el valor limite mas cercano.

    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.limits = {}
        for col in self.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.limits[col] = (lower_bound, upper_bound)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            lower_bound, upper_bound = self.limits[col]
            #Aplica los limites a los valores fuera de rango
            X[col] = X[col].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)
        return X
    
#Funcion para features ciclicos (horas) utilizando el seno y coseno

def add_cyclic_features(df, column, max_val):
    df = df.copy()
    df[column + '_sin'] = np.sin(2 * np.pi * df[column] / max_val)
    df[column + '_cos'] = np.cos(2 * np.pi * df[column] / max_val)
    return df

#Funcion para obtener relaciones entre algunos features
def add_interaction_features(df):
    df = df.copy()
    if 'hora_sin' in df.columns and 'hora_cos' in df.columns:
        #Relacion entre temperatura y hora
        df['temp_hora_interaction'] = df['temperatura'] * df['hora_sin'] * df['hora_cos']
        #Relacion entre temperatura y sensación termica
        df['temp_sensacion_interaction'] = df['temperatura'] * df['sensacion_termica']
    return df

#Funcion para extraer las columnas importantes de la fecha
def extract_date_features(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column])
    df['dia_mes'] = df[date_column].dt.day
    df['mes'] = df[date_column].dt.month
    df['dia_semana'] = df[date_column].dt.weekday
    df['trimestre'] = df[date_column].dt.quarter
    df['año'] = df[date_column].dt.year
    return df.drop(columns=[date_column])

#Funcion para la busqueda de hiperparámetros usando Randomized Search
def hyperparameter_search(pipeline, param_distributions, X_train, y_train):
    search = RandomizedSearchCV(pipeline, param_distributions, n_iter=50, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    best_params = search.best_params_
    
    return best_model, best_params

#Funcion para evaluar el modelo mediante distintas metricas
def evaluate_model(pipeline, X_train, y_train, X_test, y_test):
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse = (-scores.mean()) ** 0.5
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    return rmse, r2, y_pred

def build_pipeline(model):
    numerical_features = ['hora', 'temperatura', 'sensacion_termica', 'humedad', 'velocidad_viento']
    categorical_features = ['temporada', 'anio', 'mes', 'feriado', 'dia_semana', 'dia_trabajo', 'clima']

    return Pipeline([
        ('cyclic_features', FunctionTransformer(add_cyclic_features, kw_args={'column': 'hora', 'max_val': 24})),
        ('interaction_features', FunctionTransformer(add_interaction_features)),
        ('preprocessor', ColumnTransformer([
            ('outlier_handler', OutlierHandler(columns=['sensacion_termica', 'humedad', 'velocidad_viento']), 
             ['sensacion_termica', 'humedad', 'velocidad_viento']),
            ('imputer', SimpleImputer(strategy='median'), numerical_features),
            ('scaler', StandardScaler(), numerical_features),
            ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])),
        ('model', model)
    ])


def run_pipeline(data_path='./data/dataset_alquiler.csv'):
    scooters_df = pd.read_csv(data_path)

    #Eliminar duplicados
    scooters_df = scooters_df.drop_duplicates()

    scooters_df = extract_date_features(scooters_df, 'fecha')

    X = scooters_df.drop(columns=['indice', 'u_casuales', 'u_registrados', 'total_alquileres'])
    y = scooters_df['total_alquileres'].fillna(scooters_df['total_alquileres'].median())

    #Eliminar registros con valores nulos
    X = X.dropna()
    y = y.loc[X.index]

    #Split del dataset en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Crear un directorio para guardar los modelos en caso de que no exista
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)

    #Busqueda de mejores hiperparametros
    models = {
        'LinearRegression': (LinearRegression(), {'model__fit_intercept': [True, False]}),
        'KNN': (KNeighborsRegressor(), {'model__n_neighbors': randint(3, 11), 'model__weights': ['uniform', 'distance']}),
        'DecisionTree': (DecisionTreeRegressor(random_state=42), {'model__max_depth': randint(3, 20), 'model__min_samples_split': randint(2, 10)}),
        'RandomForest': (RandomForestRegressor(random_state=42), {'model__n_estimators': randint(100, 300), 'model__max_depth': randint(3, 20), 'model__min_samples_split': randint(2, 10)}),
        'XGBoost': (xgb.XGBRegressor(random_state=42), {'model__n_estimators': randint(100, 300), 'model__max_depth': randint(3, 10), 'model__learning_rate': uniform(0.01, 0.1)})
    }

    best_rmse = float('inf')
    best_model = None
    best_model_name = ""
    best_predictions = None

    for name, (model, param_distributions) in models.items():
        print(f"Evaluacion del modelo: {name}")
        pipeline = build_pipeline(model)
        best_model_current, best_params = hyperparameter_search(pipeline, param_distributions, X_train, y_train)
        print(f"Mejores hiperparametros para {name}: {best_params}")
        rmse, r2, y_pred = evaluate_model(best_model_current, X_train, y_train, X_test, y_test)
        print(f"RMSE para {name}: {rmse}")
        print(f"R^2 para {name}: {r2}")

        #Guardar cada modelo en el directorio creado
        model_filename = os.path.join(model_dir, f"{name}_model.joblib")
        joblib.dump(best_model_current, model_filename)
        print(f"Modelo {name} exportado como {model_filename}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = best_model_current
            best_model_name = name
            best_predictions = y_pred

    #Mostrar el mejor modelo
    if best_model is not None:
        print(f"El mejor modelo es {best_model_name} con RMSE: {best_rmse}")

        #Grafico de las predicciones del mejor modelo
        print("\nVerificando las predicciones del mejor modelo...")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, best_predictions, alpha=0.5)
        plt.xlabel("Valores Reales")
        plt.ylabel("Predicciones")
        plt.title(f"Verificación de las Predicciones - Modelo: {best_model_name}")
        plt.show()

        #Metricas del mejor modelo
        mae = mean_absolute_error(y_test, best_predictions)
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {best_rmse}")
        print(f"R^2 Score: {r2}")


if __name__ == "__main__":
    run_pipeline()
