import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("/content/student_performance_preprocessing.csv")
print("original dataset")
print(df.head())
print("\nDataset information")
print(df.info())
print("\nDatasize")
print(df.shape)
print("\nmissing values")
print(df.isnull().sum())
df["Age"]=df["Age"].fillna(df["Age"].median())
df["Attendance"]=df["Attendance"].fillna(df["Attendance"].median())
df["Python Marks"]=df["Python Marks"].fillna(df["Python Marks"].median())
df["ML Marks"]=df["ML Marks"].fillna(df["ML Marks"].median())
df["Projects Completed"]=df["Projects Completed"].fillna(df["Projects Completed"].median()
)
df["Department"]=df["Department"].fillna(df["Department"].mode()[0])
print("\nmissing values after treatment")
print(df.isnull().sum())
print("\nduplicates before removal:")
print(df.duplicated().sum())
df=df.drop_duplicates()
print("duplicates after removal:")
print(df.duplicated().sum())
df["Join Date"]=pd.to_datetime(df["Join Date"])
print("\njoin date datatype:")
print(df["Join Date"].dtype)
df[["Age","Attendance","Python Marks","ML Marks","Projects Completed"]].boxplot()
plt.title("outlier analysis")
plt.show()
columns=["Age","Attendance","Python Marks","ML Marks","Projects Completed"]
for Column in columns:
    Q1=df[Column].quantile(0.25)
    Q3=df[Column].quantile(0.75)
    IQR=Q3-Q1
    lower=Q1-1.5*IQR
    upper=Q3+1.5*IQR
    df[Column]=df[Column].clip(lower,upper)
print("\noutliers treated")

df=pd.get_dummies(df,columns=["Department"],dtype=int)
print("\nafter encoding")
print(df.head())
scale_columns=["Age","Attendance","Python Marks","ML Marks","Projects Completed"]
scaler=StandardScaler()
df[scale_columns]=scaler.fit_transform(df[scale_columns])
print("\nfinal preprocessed dataset")
print(df.head())
print("\nfinal missing values")
print(df.isnull().sum())
print("\nfinal duplicate count")
print(df.duplicated().sum())



