
# --- Gráficos univariados ---
# 1) Histograma del precio de los apartamentos
plt.figure(figsize=(10, 6))
data['precio'].plot(kind='hist', bins=30, color='blue', edgecolor='black')
plt.title('Distribución del Precio de los Apartamentos')
plt.xlabel('Precio (en USD)')
plt.ylabel('Frecuencia')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 2) Histograma del número de habitaciones
plt.figure(figsize=(10, 6))
data['habitaciones'].plot(kind='hist', bins=10, color='green', edgecolor='black')
plt.title('Distribución del Número de Habitaciones')
plt.xlabel('Número de Habitaciones')
plt.ylabel('Frecuencia')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# --- Gráficos bivariados ---
# 1) Relación entre el precio y la cantidad de habitaciones
plt.figure(figsize=(10, 6))
sns.scatterplot(x='habitaciones', y='precio', data=data, alpha=0.7, color='purple')
plt.title('Relación entre el Precio y la Cantidad de Habitaciones')
plt.xlabel('Número de Habitaciones')
plt.ylabel('Precio (en USD)')
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.show()

# 2) Relación entre el precio y los metros cuadrados
plt.figure(figsize=(10, 6))
sns.scatterplot(x='metros_cuadrados', y='precio', data=data, alpha=0.7, color='coral')
plt.title('Relación entre el Precio y los Metros Cuadrados')
plt.xlabel('Metros Cuadrados')
plt.ylabel('Precio (en USD)')
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.show()

# --- Gráficos categóricos ---
# 1) Boxplot del precio por estado
plt.figure(figsize=(12, 6))
sns.boxplot(x='estado', y='precio', data=data)
plt.xticks(rotation=45)
plt.title('Distribución del Precio por Estado')
plt.xlabel('Estado')
plt.ylabel('Precio (en USD)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 2) Boxplot del precio según si se permiten mascotas o no
plt.figure(figsize=(10, 6))
sns.boxplot(x='mascotas_permitidas', y='precio', data=data)
plt.title('Distribución del Precio según Permiso de Mascotas')
plt.xlabel('Mascotas Permitidas')
plt.ylabel('Precio (en USD)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#3) Gráfico de violín para la distribución del precio por ciudad
plt.figure(figsize=(14, 6))
sns.violinplot(x='ciudad', y='precio', data=data, density_norm='width')
plt.xticks(rotation=90)
plt.title('Distribución del Precio por Ciudad (Violin Plot)')
plt.xlabel('Ciudad')
plt.ylabel('Precio (en USD)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# --- Matriz de correlación ---
# Calcular la matriz de correlación para variables numéricas
correlation_matrix = data[['precio', 'habitaciones', 'baños', 'metros_cuadrados']].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación entre Variables Numéricas')
plt.show()
