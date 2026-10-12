import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
df = pd.read_csv("/content/Mall_Customers_2500.csv")
print("FIRST 5 ROWS")
print(df.head())
print("\nDATASET INFORMATION")
df.info()
print("\nDATASET SIZE")
print(df.shape)
X = df[[
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]]
print("\nSELECTED FEATURES")
print(X.head())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nDATA AFTER SCALING")
print(X_scaled[:5])
inertia = []
for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(
    range(2, 11),
    inertia,
    marker="o"
)
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")
plt.show()
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)
kmeans_score = silhouette_score(
    X_scaled,
    df["KMeans_Cluster"]
)
print("\nK-MEANS SILHOUETTE SCORE")
print(kmeans_score)
linked = linkage(
    X_scaled,
    method="ward"
)
plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()
hierarchical = AgglomerativeClustering(
    n_clusters=5
)
df["Hierarchical_Cluster"] = hierarchical.fit_predict(
    X_scaled
)
hierarchical_score = silhouette_score(
    X_scaled,
    df["Hierarchical_Cluster"]
)
print("\nHIERARCHICAL SILHOUETTE SCORE")
print(hierarchical_score)
dbscan = DBSCAN(
    eps=0.8,
    min_samples=5
)
df["DBSCAN_Cluster"] = dbscan.fit_predict(
    X_scaled
)
print("\nDBSCAN CLUSTERS")
print(df["DBSCAN_Cluster"].unique())
dbscan_data = df[
    df["DBSCAN_Cluster"] != -1
]
if len(dbscan_data["DBSCAN_Cluster"].unique()) > 1:

    dbscan_score = silhouette_score(
        X_scaled[df["DBSCAN_Cluster"] != -1],
        dbscan_data["DBSCAN_Cluster"]
    )
    print("\nDBSCAN SILHOUETTE SCORE")
    print(dbscan_score)
else:
    dbscan_score = None
    print("\nDBSCAN DID NOT CREATE ENOUGH CLUSTERS")
print("\nFINAL CLUSTERED DATA")
print(
    df[
        [
            "CustomerID",
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)",
            "KMeans_Cluster",
            "Hierarchical_Cluster",
            "DBSCAN_Cluster"
        ]
    ].head(10)
)
print("\nK-MEANS CLUSTER COUNT")
print(
    df["KMeans_Cluster"].value_counts()
)
print("\nHIERARCHICAL CLUSTER COUNT")
print(
    df["Hierarchical_Cluster"].value_counts()
)
print("\nDBSCAN CLUSTER COUNT")
print(
    df["DBSCAN_Cluster"].value_counts()
)
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["KMeans_Cluster"]
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means Customer Clusters")
plt.show()
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Hierarchical_Cluster"]
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Hierarchical Customer Clusters")
plt.show()
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["DBSCAN_Cluster"]
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("DBSCAN Customer Clusters")
plt.show()
print("\n======================================")
print("       CLUSTERING COMPARISON")
print("======================================")
print(
    "K-Means Silhouette Score      :",
    kmeans_score
    )
print(
    "Hierarchical Silhouette Score :",
    hierarchical_score
)
if dbscan_score is not None:

    print(
        "DBSCAN Silhouette Score       :",
        dbscan_score
    )
scores = {
    "K-Means": kmeans_score,
    "Hierarchical": hierarchical_score
}
if dbscan_score is not None:
    scores["DBSCAN"] = dbscan_score
best_method = max(
    scores,
    key=scores.get
)
best_score = scores[best_method]
print("\n======================================")
print("          BEST METHOD")
print("======================================")
print("Best Clustering Method :", best_method)
print("Best Silhouette Score  :", best_score)
print("\n======================================")
print("          INTERPRETATION")
print("======================================")
print("K-Means groups customers into similar clusters.")
print("Hierarchical clustering shows customer groups")
print("using a dendrogram.")
print("DBSCAN groups customers based on density and")
print("can identify noise or outlier customers.")
print("A higher Silhouette Score indicates better")
print("separated and well-defined clusters.")
print("\nTherefore, the method with the highest")
print("Silhouette Score is considered the best method.")
