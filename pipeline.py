import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

class OutlierHandler(BaseEstimator, TransformerMixin):
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
            X[col] = X[col].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)
        return X
    
def extract_date_features(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column])
    df['dia_mes'] = df[date_column].dt.day
    df['mes'] = df[date_column].dt.month
    df['dia_semana'] = df[date_column].dt.weekday
    df['trimestre'] = df[date_column].dt.quarter
    df['año'] = df[date_column].dt.year
    return df.drop(columns=[date_column])

# Función para la búsqueda de hiperparámetros
def hyperparameter_search(pipeline, param_grid, X_train, y_train):
    search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    best_params = search.best_params_
    
    return best_model, best_params

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
        ('preprocessor', ColumnTransformer([
            ('outlier_handler', OutlierHandler(columns=['sensacion_termica', 'humedad', 'velocidad_viento']), 
             ['sensacion_termica', 'humedad', 'velocidad_viento']),
            ('imputer', SimpleImputer(strategy='median'), numerical_features),
            ('scaler', StandardScaler(), numerical_features),
            ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])),
        ('model', model)
    ])

# Función principal para ejecutar el pipeline
def run_pipeline(data_path='data/dataset_alquiler.csv'):
    data = pd.read_csv(data_path)

    # Eliminar duplicados
    data = data.drop_duplicates()

    # Extraer características de la fecha
    data = extract_date_features(data, 'fecha')

    X = data.drop(columns=['indice', 'u_casuales', 'u_registrados', 'total_alquileres'])
    y = data['total_alquileres'].fillna(data['total_alquileres'].median())

    # Eliminar filas con valores nulos en X o y
    X = X.dropna()
    y = y.loc[X.index]

    # División del dataset en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Evaluar modelos y realizar la búsqueda de hiperparámetros
    models = {
        'LinearRegression': (LinearRegression(), {'model__fit_intercept': [True, False]}),
        'KNN': (KNeighborsRegressor(), {'model__n_neighbors': [3, 5, 7], 'model__weights': ['uniform', 'distance']}),
        'DecisionTree': (DecisionTreeRegressor(random_state=42), {'model__max_depth': [None, 10, 20], 'model__min_samples_split': [2, 5, 10]}),
        'RandomForest': (RandomForestRegressor(random_state=42), {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10], 'model__min_samples_split': [2, 5]}),
        'XGBoost': (xgb.XGBRegressor(random_state=42), {'model__n_estimators': [100, 200], 'model__max_depth': [3, 6], 'model__learning_rate': [0.01, 0.1]})
    }

    best_rmse = float('inf')
    best_model = None
    best_model_name = ""
    best_predictions = None

    for name, (model, param_grid) in models.items():
        print(f"Evaluando modelo: {name}")
        pipeline = build_pipeline(model)
        best_model_current, best_params = hyperparameter_search(pipeline, param_grid, X_train, y_train)
        print(f"Mejores hiperparámetros para {name}: {best_params}")
        rmse, r2, y_pred = evaluate_model(best_model_current, X_train, y_train, X_test, y_test)
        print(f"RMSE para {name} (mejorado): {rmse}")
        print(f"R^2 para {name} (mejorado): {r2}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = best_model_current
            best_model_name = name
            best_predictions = y_pred

    # Exportar el mejor modelo
    if best_model is not None:
        model_filename = f"best_model_{best_model_name}.joblib"
        joblib.dump(best_model, model_filename)
        print(f"Mejor modelo ({best_model_name}) exportado: {model_filename} con RMSE: {best_rmse}")

        # Verificación de las predicciones
        print("\nVerificando las predicciones del mejor modelo...")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, best_predictions, alpha=0.5)
        plt.xlabel("Valores Reales")
        plt.ylabel("Predicciones")
        plt.title(f"Verificación de las Predicciones - Modelo: {best_model_name}")
        plt.show()

        # Calcular y mostrar métricas adicionales
        mae = np.mean(np.abs(y_test - best_predictions))
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {best_rmse}")
        print(f"R^2 Score: {r2}")

if __name__ == "__main__":
    run_pipeline()
