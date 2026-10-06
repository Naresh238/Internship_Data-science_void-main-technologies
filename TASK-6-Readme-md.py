import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("/content/archive (6).zip")
print("FIRST 5 ROWS")
print(df.head())
print("\nDATASET INFORMATION")
print(df.info())
print("\nMISSING VALUES")
print(df.isnull().sum())
df["total_bedrooms"] = df["total_bedrooms"].fillna(
    df["total_bedrooms"].median()
)
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]
X = pd.get_dummies(X, columns=["ocean_proximity"], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
def check_model(name, model, x_train, x_test):
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, prediction)
    print("\n", name)
    print("MAE  :", round(mae, 2))
    print("MSE  :", round(mse, 2))
    print("RMSE :", round(rmse, 2))
    print("R2   :", round(r2, 4))
linear_model = LinearRegression()
check_model(
    "Linear Regression",
    linear_model,
    X_train_scaled,
    X_test_scaled
)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
poly_model = LinearRegression()
check_model(
    "Polynomial Regression",
    poly_model,
    X_train_poly,
    X_test_poly
)
tree_model = DecisionTreeRegressor(
    max_depth=10,
    random_state=42
)
check_model(
    "Decision Tree Regression",
    tree_model,
    X_train,
    X_test
)
forest_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
check_model(
    "Random Forest Regression",
    forest_model,
    X_train,
    X_test
)
ridge_model = Ridge(alpha=1.0)
check_model(
    "Ridge Regression",
    ridge_model,
    X_train_scaled,
    X_test_scaled
)
lasso_model = Lasso(alpha=0.1, max_iter=10000)
check_model(
    "Lasso Regression",
    lasso_model,
    X_train_scaled,
    X_test_scaled
)
print("\nREGRESSION COMPLETED SUCCESSFULLY")
