
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Energy_consumption_dataset.csv")

X = df[["Temperature", "Humidity"]].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df.loc[X.index, "Cluster"] = clusters

plt.figure(figsize=(10,6))
plt.scatter(X["Temperature"], X["Humidity"], c=clusters, cmap="viridis", edgecolor='k')
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.title("KMeans Clustering (3 clusters)")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()

print("Cantidad de puntos por cluster:")
print(df["Cluster"].value_counts())
