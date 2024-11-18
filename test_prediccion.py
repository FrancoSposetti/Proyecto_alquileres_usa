import requests
import json

# URL de tu servidor Django en el puerto 8000
url = "http://127.0.0.1:8000/api/prediccion/"

# Datos que se enviarán al servidor
data = {
    "habitaciones": 3,
    "baños": 2,
    "metros_cuadrados": 100,
    "estado": "CA",  # Asegúrate de que el valor del estado sea uno de los que tu modelo espera
    "latitud": 34.05,
    "longitud": -118.25
}

# Realizar la solicitud POST
response = requests.post(url, json=data)

# Verificar la respuesta
if response.status_code == 200:
    print("Predicción:", response.json())
else:
    print("Error:", response.status_code, response.text)
