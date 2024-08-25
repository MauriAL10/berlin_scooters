import sys
sys.path.append("..")

from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

#Cargar los modelos
models = {
    "LinearRegression": joblib.load("./models/LinearRegression_model.joblib"),
    "KNN": joblib.load("./models/KNN_model.joblib"),
    "DecisionTree": joblib.load("./models/DecisionTree_model.joblib"),
    'XGBoost': joblib.load('./models/XGBoost_model.joblib'),
    # 'RandomForest': joblib.load('./models/RandomForest_model.joblib'),
}

def add_cyclic_features(df, column, max_val):
    df[column + '_sin'] = np.sin(2 * np.pi * df[column] / max_val)
    df[column + '_cos'] = np.cos(2 * np.pi * df[column] / max_val)
    return df

def add_interaction_features(df):
    df['temp_hora_interaction'] = df['temperatura'] * df['hora_sin'] * df['hora_cos']
    df['temp_sensacion_interaction'] = df['temperatura'] * df['sensacion_termica']
    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            hora = float(request.form['hora'])
            if not 0 <= hora <= 23:
                raise ValueError("La hora debe estar entre 0 y 23.")

            temperatura = float(request.form['temperatura'])
            sensacion_termica = float(request.form['sensacion_termica'])
            humedad = float(request.form['humedad'])
            velocidad_viento = float(request.form['velocidad_viento'])
            temporada = int(request.form['temporada'])
            anio = int(request.form['anio'])
            mes = int(request.form['mes'])
            if not 1 <= mes <= 12:
                raise ValueError("El mes debe estar entre 1 y 12.")

            feriado = int(request.form['feriado'])
            if feriado not in [0, 1]:
                raise ValueError("Feriado debe ser 0 o 1.")

            dia_semana = int(request.form['dia_semana'])
            if not 0 <= dia_semana <= 6:
                raise ValueError("El día de la semana debe estar entre 0 y 6.")

            dia_trabajo = int(request.form['dia_trabajo'])
            if dia_trabajo not in [0, 1]:
                raise ValueError("Día de trabajo debe ser 0 o 1.")

            clima = int(request.form['clima'])
            if clima not in [1, 2, 3]:
                raise ValueError("Clima debe ser 1, 2, o 3.")

            modelo = request.form['modelo']

            #DataFrame con los datos
            input_data = pd.DataFrame([{
                'hora': hora,
                'temperatura': temperatura,
                'sensacion_termica': sensacion_termica,
                'humedad': humedad,
                'velocidad_viento': velocidad_viento,
                'temporada': temporada,
                'anio': anio,
                'mes': mes,
                'feriado': feriado,
                'dia_semana': dia_semana,
                'dia_trabajo': dia_trabajo,
                'clima': clima
            }])

            #Aplicar las funciones al DataFrame
            input_data = add_cyclic_features(input_data, 'hora', 24)
            input_data = add_interaction_features(input_data)

            #Prediccion con los modelos (por defecto XGBoost)
            model = models.get(modelo, models['XGBoost'])
            prediction = model.predict(input_data)
            prediction_value = float(prediction[0])

            return render_template('index.html', prediction=prediction_value)

        except ValueError as ve:
            return render_template('index.html', error=str(ve))
        except Exception as e:
            return render_template('index.html', error="Ocurrió un error inesperado: " + str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
