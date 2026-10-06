import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
df = pd.read_csv("/content/archive (5).zip")
print("FIRST 5 ROWS")
print(df.head())
print("\nDATASET INFORMATION")
print(df.info())
print("\nDATASET SIZE")
print(df.shape)
print("\nMISSING VALUES")
print(df.isnull().sum())
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
logistic = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=5)
tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)
logistic.fit(X_train, y_train)
knn.fit(X_train, y_train)
tree.fit(X_train, y_train)
logistic_pred = logistic.predict(X_test)
knn_pred = knn.predict(X_test)
tree_pred = tree.predict(X_test)
logistic_prob = logistic.predict_proba(X_test)[:, 1]
knn_prob = knn.predict_proba(X_test)[:, 1]
tree_prob = tree.predict_proba(X_test)[:, 1]
def show_results(name, actual, predicted, probability):
    print("\n==============================")
    print(name)
    print("==============================")
    print("Accuracy:",
          accuracy_score(actual, predicted))
    print("Precision:",
          precision_score(actual, predicted))
    print("Recall:",
          recall_score(actual, predicted))
    print("F1 Score:",
          f1_score(actual, predicted))
    print("ROC-AUC:",
         roc_auc_score(actual, probability))
    print("\nConfusion Matrix:")
    print(confusion_matrix(actual, predicted))
show_results(
    "LOGISTIC REGRESSION",
    y_test,
    logistic_pred,
    logistic_prob
)
show_results(
    "KNN",
    y_test,
    knn_pred,
    knn_prob
)
show_results(
    "DECISION TREE",
    y_test,
    tree_pred,
    tree_prob
)
fpr1, tpr1, _ = roc_curve(
    y_test,
    logistic_prob
)
fpr2, tpr2, _ = roc_curve(
    y_test,
    knn_prob
)
fpr3, tpr3, _ = roc_curve(
    y_test,
    tree_prob
)
plt.figure(figsize=(7, 5))
plt.plot(
    fpr1,
    tpr1,
    label="Logistic Regression"
)
plt.plot(
    fpr2,
    tpr2,
    label="KNN"
)
plt.plot(
    fpr3,
    tpr3,
    label="Decision Tree"
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "KNN",
        "Decision Tree"
    ],
    "Accuracy": [
        accuracy_score(y_test, logistic_pred),
        accuracy_score(y_test, knn_pred),
        accuracy_score(y_test, tree_pred)
    ],
    "Precision": [
        precision_score(y_test, logistic_pred),
        precision_score(y_test, knn_pred),
        precision_score(y_test, tree_pred)
    ],
   "Recall": [
        recall_score(y_test, logistic_pred),
        recall_score(y_test, knn_pred),
        recall_score(y_test, tree_pred)
    ],
   "F1 Score": [
        f1_score(y_test, logistic_pred),
        f1_score(y_test, knn_pred),
        f1_score(y_test, tree_pred)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, logistic_prob),
        roc_auc_score(y_test, knn_prob),
        roc_auc_score(y_test, tree_prob)
    ]
})
print("\n==============================")
print("MODEL COMPARISON")
print("==============================")
print(results)
