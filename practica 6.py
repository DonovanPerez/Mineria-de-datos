
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv("Energy_consumption_dataset.csv")

df["HVACUsage"] = df["HVACUsage"].astype(str).str.strip().str.lower()
df["HVACUsage"] = df["HVACUsage"].map({"off": 0, "on": 1})

df["LightingUsage"] = df["LightingUsage"].astype(str).str.strip().str.lower()
df["LightingUsage"] = df["LightingUsage"].map({"off": 0, "on": 1})

df = df.dropna(subset=["HVACUsage", "LightingUsage", "Temperature", "Humidity", "SquareFootage"])

X = df[["Temperature", "Humidity", "SquareFootage", "LightingUsage"]]
y = df["HVACUsage"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo_knn = KNeighborsClassifier(n_neighbors=5)
modelo_knn.fit(X_train, y_train)

y_pred = modelo_knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))

X_plot = X_test[["Temperature", "Humidity"]]

y_pred = modelo_knn.predict(X_test)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_plot["Temperature"], X_plot["Humidity"], c=y_pred, cmap="coolwarm", edgecolor="k")
plt.xlabel("Temperatura")
plt.ylabel("Humedad")
plt.title("Clasificación KNN: HVACUsage")
plt.colorbar(scatter, label="HVACUsage predicho")
plt.grid(True)
plt.show()
