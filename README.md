# Predicción de Alquileres de Monopatines en Berlín

## Descripción del Proyecto
Este proyecto tiene como objetivo predecir el número de alquileres de monopatines en una hora determinada en Berlín, Alemania, utilizando un modelo de aprendizaje automático. La predicción se realiza en función de diversas características como la temperatura, la humedad, la velocidad del viento, la sensación térmica, y factores temporales como la hora del día, el día de la semana, el mes, la estación y si es un día laborable o festivo.

## Contenido del proyecto

EDA (Exploratory Data Analysis): Se realizó un análisis exploratorio de los datos para entender mejor las variables y su posible impacto en el número de alquileres. Se identificaron anomalías en las variables: sensacion_termica, humedad, y velocidad_viento. También se pudo observar que la hora pico de alquileres está entre las 15:00 y las 20:00.

### Pipeline
El pipeline de este trabajo está diseñado para manejar las tareas de forma secuencial, asegurando la limpieza de datos, transformación de features y el entrenamiento del modelo.
Componentes del pipeline
1. Manejo de outliers: se utiliza el rango intercuartílico para identificar y reemplazar los valores fuera de los límites definidos. Aquellos valores que estén fuera del rango se reemplazan por los límites más cercanos.
2. Transformaciones cíclicas: esta función convierte la variable de la hora del día, utilizando seno y coseno, ya que las horas son cíclicas, porque 0 y 23 en otro contexto pueden ser valores muy alejados, pero en este caso son cercanos.
3. Relación o interacción entre variables: esta función crea nuevos features para capturar relaciones entre los datos.
4. Extracción de features de la fecha: extrae features importantes de la columna fecha, como el día, mes, año.
5. Búsqueda de hiperparámetros: se utiliza el Randomized Search probando distintas configuraciones de los modelos: Linear Regression, K-Nearest Neighbors, Decision Tree, Random Forest y XGBoost.
6. Evaluación del modelo: se utilizaron métricas como RMSE (Root Mean Squared Error) y R^2. Se realiza cross validation, entenando el modelo en el train set y comparando las predicciones en el test set.
7. Construcción del pipeline: "Pipeline" de Scikit-Learn se usa para encadenar los pasos del preprocesamiento y el modelo final.

### Interfaz Gráfica de Usuario (GUI) con Flask
1. Ingreso de datos: el usuario puede ingresar todos los parámetros necesarios para realizar una predicción, como la fecha y hora, condiciones climáticas, una vez ingresados, valores como la temperatura y humedad, se normalizan para coincidir con los datos utilizados en el entrenamiento del modelo. Los rangos para la normalización son aproximados, por ejemplo, la temperatura va desde -5° hasta 40°.
2. Selección del modelo: los usuarios pueden elegir entre distintos modelos, mencionados en la sección del pipeline. El modelo por defecto es XGBoost, ya que fue el que demostró un mejor rendimiento.
3. Una vez que los datos son ingresados, el usuario puede hacer clic en el botón "Predecir" para generar una predicción del número de monopatines que se alquilarían bajo esas condiciones.

## Requisitos
Para ejecutar el proyecto localmente, se deben tener los siguientes paquetes instalados:
- Python 3.7 o superior
- Flask
- Joblib
- Pandas
- Numpy
- Scikit-learn
- XGBoost
Se pueden instalar ejecutando
```
pip install -r requirements.txt
```
## Instrucciones de ejecución
1. Clonar el repositorio
```
git clone https://github.com/MauriAL10/berlin_scooters
```
2. Ejecutar el pipeline: en la raíz del proyecto ejecutar el siguiente comando para construir el pipeline y exportar los modelos
```
python main.py
```
Nota: Una vez se exporten todos los modelos aparecerá una imagen que muestra la calidad de predicciones del mejor modelo, se debe cerrar la imagen para terminar la ejecución.
3. Ejecutar la aplicación: para poder observar la GUI, ejecutar el siguiente comando:
```
python app/app.py
```
Abrir el navegador y dirigirse a http://127.0.0.1:5000/ para acceder a la aplicación.
4. Ingresar los datos relevantes en la interfaz del usuario y seleccionar el modelo de predicción deseado. Finalmente se debe hacer clic en "Predecir".
Ejemplos de algunas predicciones:
- Invierno: 15 de enero de 2012, 10:00 AM, -2°C, -5°C, 85% de humedad, 15 km/h de viento, día nublado.
- Primavera: 20 de abril de 2012, 6:00 PM, 16°C, 16°C, 60% de humedad, 10 km/h de viento, día soleado.
- Verano: 10 de agosto de 2012, 2:00 PM, 28°C, 30°C, 70% de humedad, 5 km/h de viento, día soleado.
- Otoño: 25 de octubre de 2012, 8:00 AM, 10°C, 8°C, 75% de humedad, 12 km/h de viento, día nublado.
