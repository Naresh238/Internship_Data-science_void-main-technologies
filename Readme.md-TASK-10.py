import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
df = pd.read_csv('/content/archive (8).zip')
print(df.head())
df = df.drop_duplicates()
df = df.dropna(subset=['Exam_Score'])
X = df.drop('Exam_Score', axis=1)
y = df['Exam_Score']
for col in ['ID', 'Student_ID']:
    if col in X.columns:
        X = X.drop(col, axis=1)
num_cols = X.select_dtypes(include='number').columns
cat_cols = X.select_dtypes(exclude='number').columns
num_steps = SimpleImputer(strategy='median')
cat_steps = Pipeline([
    ('fill', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])
prepare_data = ColumnTransformer([
    ('numbers', num_steps, num_cols),
    ('text', cat_steps, cat_cols)
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(
        n_estimators=50, random_state=42
    ),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(
        random_state=42
    )
}
models['Voting Regressor'] = VotingRegressor([
    ('rf', RandomForestRegressor(
        n_estimators=50, random_state=42
    )),
    ('ada', AdaBoostRegressor(random_state=42)),
    ('gb', GradientBoostingRegressor(random_state=42))
])
scores = []
for name, model in models.items():
    program = Pipeline([
        ('prepare', prepare_data),
        ('model', model)
    ])
    program.fit(X_train, y_train)
    predicted = program.predict(X_test)
    error = mean_absolute_error(y_test, predicted)
    accuracy = r2_score(y_test, predicted)
    scores.append([name, round(error, 2), round(accuracy, 3)])
result = pd.DataFrame(
    scores, columns=['Model', 'MAE', 'R2 Score']
)
print("\nModel Comparison")
print(result)
best = result.loc[result['R2 Score'].idxmax()]
print("\nBest Model:", best['Model'])
plt.bar(result['Model'], result['R2 Score'])
plt.xlabel('Models')
plt.ylabel('R2 Score')
plt.title('Model Performance Comparison')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
