import joblib
import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

# Cargar el modelo guardado
modelo = joblib.load('C:/Users/kbm19/Desktop/Franco Cursos/Curso Data Science/Proyecto_alquiler_usa/modelo_random_forest.pkl')

# Ruta del archivo que contiene los datos de entrenamiento para conocer las columnas exactas
# Si tienes un archivo CSV o un DataFrame con las características de entrenamiento
# data_train = pd.read_csv("ruta/a/tu/dataset_de_entrenamiento.csv")  # Carga el archivo original de entrenamiento
# X_train = pd.get_dummies(data_train[['habitaciones', 'baños', 'metros_cuadrados', 'estado', 'latitud', 'longitud']], drop_first=True)
# Las columnas de características con las que el modelo fue entrenado
# columnas_dummies = X_train.columns

# Cargar las columnas desde el modelo (esto te da el número exacto de características)
columnas_dummies = modelo.feature_names_in_

@csrf_exempt
def prediccion(request):
    if request.method == 'POST':
        try:
            # Obtener los datos del cuerpo de la solicitud
            datos = json.loads(request.body)
            print(f"Datos recibidos: {datos}")  # Imprimir los datos para depuración

            # Obtener las variables enviadas en la solicitud
            habitaciones = datos.get('habitaciones')
            baños = datos.get('baños')
            metros_cuadrados = datos.get('metros_cuadrados')
            estado = datos.get('estado')
            latitud = datos.get('latitud')
            longitud = datos.get('longitud')

            # Asegurarse de que todas las variables estén presentes
            if None in [habitaciones, baños, metros_cuadrados, estado, latitud, longitud]:
                raise ValueError("Faltan datos. Asegúrese de enviar todas las variables necesarias.")

            # Aquí, transformamos las variables categóricas en dummies (One-Hot Encoding)
            data = pd.DataFrame({
                'habitaciones': [habitaciones],
                'baños': [baños],
                'metros_cuadrados': [metros_cuadrados],
                'estado': [estado],
                'latitud': [latitud],
                'longitud': [longitud]
            })

            # Convertir las variables categóricas a dummies
            data_dummies = pd.get_dummies(data, drop_first=True)

            # Asegurarse de que las columnas generadas coincidan con las del entrenamiento
            data_dummies = data_dummies.reindex(columns=columnas_dummies, fill_value=0)

            print(f"Variables dummy para la predicción: {data_dummies}")  # Para ver las columnas antes de hacer la predicción

            # Convertir a array de numpy para hacer la predicción
            variables = np.array(data_dummies)

            # Hacer la predicción
            resultado = modelo.predict(variables)
            print(f"Resultado de la predicción: {resultado}")  # Ver la predicción

            # Retornar la predicción en formato JSON
            return JsonResponse({'prediccion': resultado[0]})

        except Exception as e:
            # Devolver un error más detallado
            print(f"Error al procesar la solicitud: {str(e)}")  # Imprimir el error
            return JsonResponse({'error': str(e)}, status=400)
