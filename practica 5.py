
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("Energy_consumption_dataset.csv")

df["Occupancy"] = df["Occupancy"].map({"Off": 0, "On": 1})
df["HVACUsage"] = df["HVACUsage"].map({"Off": 0, "On": 1})
df["LightingUsage"] = df["LightingUsage"].map({"Off": 0, "On": 1})

df = df.dropna(subset=["Temperature", "EnergyConsumption"])

X = df[["Temperature"]].values
y = df["EnergyConsumption"].values

modelo = LinearRegression()
modelo.fit(X, y)

y_pred = modelo.predict(X)
r2 = r2_score(y, y_pred)
print(f"R² score: {r2:.4f}")
print("Pendiente (coeficiente):", modelo.coef_[0])

plt.figure(figsize=(8, 6))
sns.regplot(x="Temperature", y="EnergyConsumption", data=df, line_kws={"color": "red"})
plt.title("Consumo de Energía en función de la Temperatura")
plt.xlabel("Temperatura")
plt.ylabel("Consumo de Energía")
plt.grid(True)
plt.show()

