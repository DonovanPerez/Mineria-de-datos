
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Energy_consumption_dataset.csv")

print(" Estadísticas descriptivas generales:")
print(df.describe())

print("\n Consumo promedio de energía por día de la semana:")
group_day = df.groupby("DayOfWeek")["EnergyConsumption"].agg(["mean", "sum", "count"])
print(group_day)

print("\n Consumo promedio por hora del día:")
group_hour = df.groupby("Hour")["EnergyConsumption"].agg(["mean", "sum", "count"])
print(group_hour)

print("\n Consumo promedio por mes:")
group_month = df.groupby("Month")["EnergyConsumption"].agg(["mean", "sum", "count"])
print(group_month)

#boxplot del consumo energético por día de la semana
plt.figure(figsize=(10, 6))
df.boxplot(column="EnergyConsumption", by="DayOfWeek")
plt.title("Boxplot del consumo energético por día de la semana")
plt.suptitle("")
plt.xlabel("Día de la semana")
plt.ylabel("Consumo de Energía")
plt.tight_layout()
plt.savefig("boxplot_energy_by_dayofweek.png")
plt.show()
