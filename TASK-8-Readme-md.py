
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv("/content/student_performance_pca.csv")
print("First five rows:")
print(df.head())
X = df.drop("Pass", axis=1)
y = df["Pass"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model1 = LogisticRegression()
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)
acc1 = accuracy_score(y_test, pred1)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print("\nFeatures before PCA:", X_train.shape[1])
print("Features after PCA:", X_train_pca.shape[1])
variance = pca.explained_variance_ratio_
print("PC1 variance:", round(variance[0] * 100, 2), "%")
print("PC2 variance:", round(variance[1] * 100, 2), "%")
print("Total variance:", round(sum(variance) * 100, 2), "%")
model2 = LogisticRegression()
model2.fit(X_train_pca, y_train)
pred2 = model2.predict(X_test_pca)
acc2 = accuracy_score(y_test, pred2)
print("\nAccuracy before PCA:", round(acc1 * 100, 2), "%")
print("Accuracy after PCA:", round(acc2 * 100, 2), "%")
plt.bar(["PC1", "PC2"], variance * 100)
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance (%)")
plt.title("PCA Variance Analysis")
plt.show()
