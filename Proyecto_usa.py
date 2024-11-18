# --- Análisis de Dataset: Apartamentos en alquiler en Estados Unidos ---

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

# --- Análisis Exploratorio de Datos --- 

# Creo el DataFrame
ruta = r"C:\Users\kbm19\Desktop\Franco Cursos\Curso Data Science\apartment_for_rent_classified.csv"
data = pd.read_csv(ruta)

# Primera vista del dataset
def analizar_dataframe(df):
    """
    Analiza un DataFrame mostrando:
    - Las primeras filas
    - Valores nulos
    - Descripción estadística de las variables numéricas
    - Información general del DataFrame
    
    Args:
    df (pd.DataFrame): El DataFrame a analizar.
    """
    print("---- Primeras filas del DataFrame ----")
    print(df.head(), "\n")  # Muestra las primeras filas

    print("---- Valores nulos en el DataFrame ----")
    print(df.isnull().sum(), "\n")  # Muestra la cantidad de valores nulos por columna

    print("---- Descripción estadística de las variables numéricas ----")
    print(df.describe(), "\n")  # Muestra estadísticas descriptivas

    print("---- Información general del DataFrame ----")
    print(df.info())  # Muestra información general del DataFrame

# Llamar a la función con el DataFrame 
analizar_dataframe(data)

# --- Limpieza de Datos --- 

# Eliminar las columnas 'amenities' y 'address'
data.drop(columns=['amenities', 'address', "time", "title", "body", "has_photo", "price_display", "source"], inplace=True)

# Convertir las columnas numéricas a un formato adecuado
numeric_columns = ['bathrooms', 'bedrooms', 'price', 'square_feet', 'latitude', 'longitude']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Rellenar las columnas 'cityname' y 'state' con el valor anterior
data['cityname'] = data['cityname'].fillna(method='ffill')
data['state'] = data['state'].fillna(method='ffill')

# Llenar valores faltantes en las otras columnas

# 1. Columnas numéricas: Rellenar con la mediana
data['bathrooms'] = data['bathrooms'].fillna(data['bathrooms'].median()).round().astype(int)
data['bedrooms'] = data['bedrooms'].fillna(data['bedrooms'].median())
data['price'] = data['price'].fillna(data['price'].median())
data['square_feet'] = data['square_feet'].fillna(data['square_feet'].median())
data['latitude'] = data['latitude'].fillna(data['latitude'].median())
data['longitude'] = data['longitude'].fillna(data['longitude'].median())

# 2. Columnas categóricas: Rellenar con el valor más frecuente (moda)
data['currency'] = data['currency'].fillna(data['currency'].mode()[0])
data['fee'] = data['fee'].fillna(data['fee'].mode()[0])
data['pets_allowed'] = data['pets_allowed'].fillna(data['pets_allowed'].mode()[0])
data['price_type'] = data['price_type'].fillna(data['price_type'].mode()[0])

# Confirmar que no queden valores faltantes
print(data.isnull().sum())

# Eliminar filas donde 'cityname' es un número
filas_incorrectas_ciudad = data[~data['cityname'].apply(lambda x: isinstance(x, str))]
num_filas_eliminadas_ciudad = len(filas_incorrectas_ciudad)

# Eliminar esas filas del DataFrame
data = data[data['cityname'].apply(lambda x: isinstance(x, str))]

# Mostrar el número de filas eliminadas
print(f"Número de filas eliminadas debido a valores numéricos en 'cityname': {num_filas_eliminadas_ciudad}")

# Cambiar el nombre de las columnas 
data.rename(columns={
    "category": "categoria",
    'cityname': 'ciudad',
    'state': 'estado',
    'bathrooms': 'baños',
    'bedrooms': 'habitaciones',
    'price': 'precio',
    'square_feet': 'metros_cuadrados',
    'latitude': 'latitud',
    'longitude': 'longitud',
    'currency': 'moneda',
    'fee': 'cuota',
    'pets_allowed': 'mascotas_permitidas',
    'price_type': 'tipo_precio',
}, inplace=True)

# Verificar los nuevos nombres de las columnas
print(data.columns)

# Verifico valores en cada columna para corroborar la consistencia de los datos 

# Crear una lista con todos los nombres de las columnas del DataFrame
columnas_para_verificar = data.columns.tolist()

# Mostrar la lista de columnas
print("Lista de todas las columnas:")
print(columnas_para_verificar)

# Recorro la lista para verificar
for columna in columnas_para_verificar:
    print(f"Valores únicos en la columna '{columna}':")
    print(data[columna].unique())
    print("\n")


def limpiar_columna(data, columna, valores_validos):
    """
    Limpia una columna del DataFrame verificando si los valores son válidos según una lista proporcionada.
    Elimina las filas que contienen valores no válidos.

    :param data: DataFrame a verificar y limpiar
    :param columna: Nombre de la columna a limpiar
    :param valores_validos: Lista de valores válidos
    """
    # Convertir la lista de valores válidos a un conjunto para comparación
    valores_validos_set = set(valores_validos)

    # Reemplazar los valores no válidos por NaN
    data[columna] = data[columna].where(data[columna].isin(valores_validos_set), other=np.nan)

    # Eliminar filas que contienen NaN en la columna especificada
    data.dropna(subset=[columna], inplace=True)

    # Imprimir resultados
    valores_unicos = data[columna].unique()
    print(f"Valores únicos en la columna '{columna}' después de limpiar: {valores_unicos}\n")

# Definir valores válidos para algunas de las columnas del DataFrame
valores_validos_moneda = ['USD']  
valores_validos_cuota = ['Yes', 'No'] 
valores_validos_mascotas_permitidas = ['Cats', 'Cats,Dogs', 'Dogs', 'Cats,Dogs,None']
valores_validos_tipo_precio = ['Monthly', 'Weekly', 'Monthly|Weekly']
valores_validos_estado = ['CA', 'VA', 'NC', 'NM', 'CO', 'WV', 'GA', 'MA', 'DC', 'AZ', 'IA', 'WA', 
                          'TX', 'IL', 'MS', 'OR', 'FL', 'MO', 'PA', 'WI', 'OK', 'UT', 'RI', 
                          'NJ', 'IN', 'MD', 'OH', 'TN', 'ND', 'NE', 'AR', 'MI', 'MN', 'HI', 
                          'ID', 'SC', 'KS', 'AL', 'SD', 'NY', 'KY', 'LA', 'AK', 'CT', 'NV', 
                          'WY', 'VT', 'NH', 'MT', 'DE', 'ME']

# Limpiar las columnas del DataFrame usando la función
limpiar_columna(data, 'moneda', valores_validos_moneda)
limpiar_columna(data, 'cuota', valores_validos_cuota)
limpiar_columna(data, 'mascotas_permitidas', valores_validos_mascotas_permitidas)
limpiar_columna(data, 'tipo_precio', valores_validos_tipo_precio)
limpiar_columna(data, 'estado', valores_validos_estado)

# Confirmar que no queden valores no válidos
print("Número de valores nulos en cada columna después de limpiar:")
print(data.isnull().sum())


# --- Análisis Final del Dataset Limpio ---

def analizar_dataframe(df):
    """
    Analiza un DataFrame mostrando:
    - Las primeras filas
    - Valores nulos
    - Descripción estadística de las variables numéricas
    - Información general del DataFrame
    
    Args:
    df (pd.DataFrame): El DataFrame a analizar.
    """
    print("---- Primeras filas del DataFrame ----")
    print(df.head(), "\n")  # Muestra las primeras filas

    print("---- Valores nulos en el DataFrame ----")
    print(df.isnull().sum(), "\n")  # Muestra la cantidad de valores nulos por columna

    print("---- Descripción estadística de las variables numéricas ----")
    print(df.drop(columns=["latitud", "longitud"]).describe(), "\n")  # Muestra estadísticas descriptivas

    print("---- Información general del DataFrame ----")
    print(df.info())  # Muestra información general del DataFrame

# Llamar a la función con el DataFrame 
analizar_dataframe(data)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paso 1: Preparar los datos para la regresión
features = ['habitaciones', 'baños', 'metros_cuadrados', 'estado', 'latitud', 'longitud']
target = 'precio'

# Crear variables dummy para las características categóricas
X = pd.get_dummies(data[features], drop_first=True)
y = data[target]

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 2: Crear y entrenar el modelo
rf_model = RandomForestRegressor(n_estimators=50, random_state=42)  # Reducir el número de árboles
rf_model.fit(X_train, y_train)

# Paso 3: Hacer predicciones
y_pred_rf = rf_model.predict(X_test)

# Paso 4: Evaluar el rendimiento del modelo
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print("Error cuadrático medio (MSE) con Random Forest:", mse_rf)
print("R² (coeficiente de determinación) con Random Forest:", r2_rf)
print("Error absoluto medio (MAE) con Random Forest:", mae_rf)

# Importancia de las características
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

feature_importance_df = pd.DataFrame({
    'feature': X.columns[indices],
    'importance': importances[indices]
})

# Paso 5: Agregar nuevos datos para ver cómo predice
num_nuevos_datos = 10

# Generar datos aleatorios
habitaciones = np.random.randint(1, 5, num_nuevos_datos)  # Habitaciones entre 1 y 4
baños = np.random.randint(1, 3, num_nuevos_datos)          # Baños entre 1 y 2
metros_cuadrados = np.random.randint(50, 150, num_nuevos_datos)  # Metros cuadrados entre 50 y 150
estados = np.random.choice(['CA', 'TX', 'NY', 'FL', 'IL'], num_nuevos_datos)  # Estados válidos
latitud = np.random.uniform(25.0, 50.0, num_nuevos_datos)  # Latitud en EE. UU.
longitud = np.random.uniform(-125.0, -66.0, num_nuevos_datos)  # Longitud en EE. UU.

# Crear el DataFrame de nuevos datos
nuevos_datos = pd.DataFrame({
    'habitaciones': habitaciones,
    'baños': baños,
    'metros_cuadrados': metros_cuadrados,
    'estado': estados,
    'latitud': latitud,
    'longitud': longitud
})

# Convertir a variables dummy
nuevos_datos_dummies = pd.get_dummies(nuevos_datos, drop_first=True)

# Asegurarte de que las columnas coincidan con el modelo
nuevos_datos_dummies = nuevos_datos_dummies.reindex(columns=X.columns, fill_value=0)

# Hacer predicciones
precios_predichos = rf_model.predict(nuevos_datos_dummies)
nuevos_datos['precio_predicho'] = precios_predichos

# Ver los precios predichos
print(nuevos_datos[['habitaciones', 'baños', 'metros_cuadrados', 'estado', 'latitud', 'longitud', 'precio_predicho']])

import joblib

# Guardar el modelo
joblib.dump(rf_model, 'modelo_random_forest.pkl')
print("Modelo guardado exitosamente como 'modelo_random_forest.pkl'")
