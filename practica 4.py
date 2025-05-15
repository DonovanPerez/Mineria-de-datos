
import pandas as pd
from scipy.stats import f_oneway

df = pd.read_csv("Energy_consumption_dataset.csv")

grupos = [grupo['EnergyConsumption'].values for _, grupo in df.groupby('DayOfWeek')]

f_stat, p_value = f_oneway(*grupos)

print(f"F-Statistic: {f_stat:.3f}")
print(f"P-Value: {p_value:.3f}")


if p_value < 0.05:
    print("Hay diferencias significativas entre los días de la semana en el consumo de energía.")
else:
    print("No hay diferencias significativas entre los días de la semana en el consumo de energía.")
