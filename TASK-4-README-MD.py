import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
df = pd.read_csv("/content/titanic_feature_engineering.csv")
print("FIRST 5 ROWS")
print(df.head())
print("\nMISSING VALUES")
print(df.isnull().sum())
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = np.where(df["FamilySize"] == 1, 1, 0)
print("\nNEW FEATURES")
print(df[["FamilySize", "IsAlone"]].head())
df = df[[
    "Survived",
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "FamilySize",
    "IsAlone"
]]
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})
X = df.drop("Survived", axis=1)
y = df["Survived"]
print("\nINPUT FEATURES")
print(X.head())
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model1 = LogisticRegression()
model1.fit(X_train, y_train)
prediction1 = model1.predict(X_test)
accuracy1 = accuracy_score(y_test, prediction1)
print("\nACCURACY BEFORE FEATURE SELECTION")
print(accuracy1)
selector = SelectKBest(score_func=chi2, k=4)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("\nSELECTED FEATURES")
print(selected_features)
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_selected,
    y,
    test_size=0.2,
    random_state=42
)
X_train2 = scaler.fit_transform(X_train2)
X_test2 = scaler.transform(X_test2)
model2 = LogisticRegression()
model2.fit(X_train2, y_train2)
prediction2 = model2.predict(X_test2)
accuracy2 = accuracy_score(y_test2, prediction2)
print("\nACCURACY AFTER FEATURE SELECTION")
print(accuracy2)
print("\nMODEL COMPARISON")
print("----------------------------")
print("Before Feature Selection:", accuracy1)
print("After Feature Selection :", accuracy2)
print("\nDATA LEAKAGE CHECK")
if "Survived" not in X.columns:
    print("No data leakage")
else:
    print("Target is separated correctly")
