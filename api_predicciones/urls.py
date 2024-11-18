# api_predicciones/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.prediccion, name='prediccion'),  # Asegúrate de que la vista esté asociada a la ruta correcta
]
