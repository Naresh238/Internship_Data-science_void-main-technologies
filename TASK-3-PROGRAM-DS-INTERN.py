import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
df=pd.read_csv("/content/archive (25).zip")
print("FIRST FIVE ROWS")
print(df.head())
print("\nDataset Information")
df.info()
print("\nDataset size")
print(df.shape)
print("\nMissing Values")
print(df.isnull().sum())
print("\nDescriptive Stastics")
print(df.describe())
print("\nMEAN")
print(df.mean(numeric_only=True))
print("\nMedian")
print(df.median(numeric_only=True))
print("\nStandard Deviation")
print(df.std(numeric_only=True))
plt.hist(df["math score"])
plt.title("Math Score Distribution")
plt.xlabel("Math Score")
plt.ylabel("Number Of Students")
plt.show()
plt.hist(df["reading score"])
plt.title("Reading Score Distribution")
plt.xlabel("Reading score")
plt.ylabel("Number of Students")
plt.show()
plt.hist(df["writing score"])
plt.title("Writing Score Distribution")
plt.xlabel("writing score")
plt.ylabel("Number of students")
plt.show()
plt.boxplot([
    df["math score"],
    df["reading score"],
    df["writing score"]
    ])
plt.title("score Distribution")
plt.xticks([1,2,3],["Math","Reading","Writing"])
plt.ylabel("Score")
plt.show()
plt.scatter(df["math score"],df["reading score"])
plt.title("Math Score vs Reading Score")
plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.show()
plt.scatter(df["math score"],df["writing score"])
plt.title("Math Score vs Writing Score")
plt.xlabel("Math score")
plt.ylabel("Writing Score")
plt.show()
print("\nCorrelation Matrix")
correlation = df[
    ["math score", "reading score", "writing score"]
].corr()
print(correlation)
print("\nCovariance Matrix")
covariance = df[
    ["math score", "reading score", "writing score"]
].cov()
print(covariance)
df["average score"] = (
    df["math score"]
    + df["reading score"]
    + df["writing score"]
) / 3
print("\nAverage Score")
print(df[[
    "math score",
    "reading score",
    "writing score",
    "average score"
]].head())
print("\nAverage Score Distribution")
print(df["average score"].describe())
completed = df[
    df["test preparation course"] == "completed"
]["math score"]

not_completed = df[
    df["test preparation course"] == "none"
]["math score"]
t_value, p_value = ttest_ind(
    completed,
    not_completed
)
print("\nHypothesis Testing")
print("T-value:", t_value)
print("P-value:", p_value)
if p_value < 0.05:
    print("There is a significant difference in math scores.")
else:
    print("There is no significant difference in math scores.")
print("\nFINAL FINDINGS")
print("1. Descriptive statistics show the basic properties of the scores.")
print("2. Histograms show the distribution of Math, Reading and Writing scores.")
print("3. Scatter plots show relationships between the scores.")
print("4. Correlation shows the strength of relationships between scores.")
print("5. Covariance shows the direction of relationships between scores.")
print("6. Average score is used for multivariate analysis.")
print("7. The t-test checks whether test preparation has a significant")
print("   effect on Math scores.")







