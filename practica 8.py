
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('Energy_consumption_dataset.csv')

df['Month'] = pd.to_numeric(df['Month'], errors='coerce')

df_mes = df.groupby('Month')['EnergyConsumption'].mean().reset_index()

X = df_mes[['Month']].values  
y = df_mes['EnergyConsumption'].values 

modelo = LinearRegression()
modelo.fit(X, y)

y_pred = modelo.predict(X)

X_new = np.array([[13], [14], [15], [16], [17], [18]])
y_new_pred = modelo.predict(X_new)

plt.figure(figsize=(10,6))
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X, y_pred, color='red', label='Predicción modelo')
plt.scatter(X_new, y_new_pred, color='green', label='Predicciones futuras')
plt.xlabel('Mes')
plt.ylabel('Consumo energético promedio')
plt.title('Forecasting consumo energético por mes con regresión lineal')
plt.legend()
plt.show()












