import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
uploaded = files.upload()
filename = list(uploaded.keys())[0]
df = pd.read_csv('/content/archive (9).zip')
print("Dataset loaded successfully!")
print(df.head())
print("\nTransaction Count:")
print(df['Class'].value_counts())
df['Class'].value_counts().plot(kind='bar')
plt.title("Normal vs Fraud Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.show()
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(
    y_test, predictions,
    target_names=['Normal', 'Fraud'],
    zero_division=0
))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
importance = permutation_importance(
    model, X_test, y_test,
    scoring='f1',
    n_repeats=3,
    random_state=42
)
features = pd.DataFrame({
    'Feature': df.drop('Class', axis=1).columns,
    'Importance': importance.importances_mean
})
features = features.sort_values(
    by='Importance', ascending=False
)
print("\nTop 10 Important Features:")
print(features.head(10))
features.head(10).sort_values(
    by='Importance'
).plot(
    x='Feature', y='Importance',
    kind='barh', legend=False
)
plt.title("Important Features for Fraud Detection")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
