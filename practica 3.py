
import pandas as pd
import matplotlib.pyplot as plt

# Leer datos
df = pd.read_csv("Energy_consumption_dataset.csv")

# Histograma de consumo de energía
plt.figure()
plt.hist(df["EnergyConsumption"], bins=30, color='skyblue')
plt.title("Histograma de Consumo de Energía")
plt.xlabel("Consumo")
plt.ylabel("Frecuencia de los registros")
plt.savefig("histograma_consumo.png")
plt.close()

# Línea: consumo promedio por hora
avg_by_hour = df.groupby("Hour")["EnergyConsumption"].mean()
plt.figure()
avg_by_hour.plot(kind='line', marker='o', color='orange')
plt.title("Consumo Promedio por Hora")
plt.xlabel("Hora")
plt.ylabel("Consumo Promedio")
plt.grid()
plt.savefig("linea_hora_consumo.png")
plt.close()

# Gráfico de barras: consumo promedio por día de la semana
avg_by_day = df.groupby("DayOfWeek")["EnergyConsumption"].mean()
plt.figure(figsize=(8, 6))
avg_by_day.plot(kind='bar', color='teal')
plt.title("Consumo Promedio por Día de la Semana")
plt.xlabel("Día de la Semana (0 = Lunes)")
plt.ylabel("Consumo Promedio de Energía")
plt.tight_layout()
plt.savefig("barras_consumo_dia.png")
plt.close()


# Pie chart: Energía renovable vs no renovable
df["UsaRenovable"] = df["RenewableEnergy"] > 0
renewable_counts = df["UsaRenovable"].value_counts()
plt.figure()
renewable_counts.plot(kind='pie', labels=["No Renovable", "Renovable"], autopct='%1.1f%%', colors=['red', 'green'])
plt.title("Distribución de Uso de Energía Renovable")
plt.ylabel("") 
plt.savefig("pie_renovable.png")
plt.close()

# Scatter plot: temperatura vs consumo de energía
plt.figure()
plt.scatter(df["Temperature"], df["EnergyConsumption"], alpha=0.3)
plt.title("Temperatura vs Consumo de Energía")
plt.xlabel("Temperatura")
plt.ylabel("Consumo")
plt.savefig("scatter_temp_consumo.png")
plt.close()

print("Se generaron las imagenes de las graficas")