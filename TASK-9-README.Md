
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
data = pd.read_csv("/content/student_performance.csv")
print(data.head())
X = data.drop("Final_Marks", axis=1)
y = data["Final_Marks"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model1 = DummyRegressor(strategy="mean")
model1.fit(X_train, y_train)
prediction1 = model1.predict(X_test)
print("Baseline MAE:", mean_absolute_error(y_test, prediction1))
model2 = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestRegressor(random_state=42)
)
scores = cross_val_score(model2, X_train, y_train, cv=5, scoring="r2")
print("Cross-validation scores:", scores)
print("Average score:", scores.mean())
parameters = {
    "randomforestregressor__n_estimators": [50, 100],
    "randomforestregressor__max_depth": [3, 5, None]
}
search = GridSearchCV(model2, parameters, cv=5, scoring="r2")
search.fit(X_train, y_train)
best_model = search.best_estimator_
print("Best settings:", search.best_params_)
train_prediction = best_model.predict(X_train)
test_prediction = best_model.predict(X_test)
print("Training R2:", r2_score(y_train, train_prediction))
print("Testing R2:", r2_score(y_test, test_prediction))
print("Final MAE:", mean_absolute_error(y_test, test_prediction))
train_score = r2_score(y_train, train_prediction)
test_score = r2_score(y_test, test_prediction)
if train_score < 0.5 and test_score < 0.5:
    print("The model may be underfitting")
elif train_score - test_score > 0.2:
    print("The model may be overfitting")
else:
    print("The model has no strong sign of overfitting")
print("Program completed")
