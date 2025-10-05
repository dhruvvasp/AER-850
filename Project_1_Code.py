#Step 1 - Reading Data and Converting into Dataframe

import pandas as pd

data = pd.read_csv("Project 1 Data.csv")

#Print first 50 rows of the dataframe
print(data.head(50)) 

#Print the shape of the dataframe
print(data.shape) 

#Print unique counts excluding NaN
print("\nUnique counts by column:") 
print(data.nunique())

#Step 2 - Visualizing Data by Creating Plots

import matplotlib.pyplot as plt

#Histogram of Data:
#Provides a quick overview of the distribution of values in each column
histogram = data.hist(
bins=30,                
figsize=(15, 10),       
grid=True,             
edgecolor='black',     
linewidth=1.2
)

plt.savefig("Project 1 Histogram.png") # Save the histogram as a PNG file
plt.show()
print(histogram) # Print the histogram  

#Scatter Plots for Pairwise Relationships:
#Useful for identifying correlations or patterns between pairs of variables
plt.figure(figsize=(15, 5))

#Plotting scatter plots for X vs Z and Y vs Z
plt.subplot(1, 2, 1) 
plt.scatter(data['X'], data['Z'], alpha=0.7)
plt.xlabel('X'); plt.ylabel('Z'); plt.title('X vs Z') 

plt.subplot(1, 2, 2)
plt.scatter(data['Y'], data['Z'], alpha=0.7)
plt.xlabel('Y'); plt.ylabel('Z'); plt.title('Y vs Z')

#Plotting layout adjustments
plt.tight_layout()
plt.savefig("Project 1 Scatter Plots.png", dpi=150) # Save the scatter plots as a PNG file
plt.show()

#Box Plots for Distribution by Category
#Useful for comparing distributions of X, Y, Z across different Steps
steps = sorted(data['Step'].unique())

x_by_step = [data.loc[data['Step'] == s, 'X'] for s in steps]
y_by_step = [data.loc[data['Step'] == s, 'Y'] for s in steps]
z_by_step = [data.loc[data['Step'] == s, 'Z'] for s in steps]

plt.figure(figsize=(15, 5))

#Plotting box plots for X, Y, Z by Step
plt.subplot(1, 3, 1)
plt.boxplot(x_by_step, tick_labels=steps)
plt.xlabel('Step'); plt.ylabel('X'); plt.title('X by Step')
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
plt.boxplot(y_by_step, tick_labels=steps)
plt.xlabel('Step'); plt.ylabel('Y'); plt.title('Y by Step')
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)
plt.boxplot(z_by_step, tick_labels=steps)
plt.xlabel('Step'); plt.ylabel('Z'); plt.title('Z by Step')
plt.xticks(rotation=45)

#Plotting layout adjustments
plt.tight_layout()
plt.savefig("Project 1 Box Plots.png", dpi=150) # Save the box plots as a PNG file
plt.show() 

#Step 3 - Correlation Analysis through Correlation Matrix and Plots

import seaborn as sns
corr_matrix = data.corr()

#Plot the Pearson Correlation matrix
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')     
plt.savefig("Project 1 Correlation Matrix.png", dpi=150) # Save the correlation matrix as a PNG file
plt.show()
print(corr_matrix) # Print the correlation matrix


#Step 4 - Classification Model Development/Engineering

# Data Splitting
X = data[["X", "Y", "Z"]]
y = data["Step"]

from sklearn.model_selection import train_test_split

# Stratified split (80/20)
#Train and test sets will have similar distribution of the target variable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

#Classification Model Training and Evaluation

#1. Logistic Regression Model with Hyperparameter Tuning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score

#CV strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Pipeline
regressionpipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs", random_state=42))
])

#Parameter grid
regressionparams = {
    "clf__C": [0.1, 1, 10, 100],
    "clf__class_weight": [None, "balanced"],
}

#Grid search
regressiongrid = GridSearchCV(
    regressionpipeline,
    regressionparams,
    cv=cv,
    n_jobs=-1,
    scoring="f1_macro",   
    refit=True,
    verbose=2
)

regressiongrid.fit(X_train, y_train)

regressiongrid.fit(X_train, y_train)

print("Best CV score - Logistic Regression:", regressiongrid.best_score_)
print("Best params - Logistic Regression:", regressiongrid.best_params_)

# Evaluate on test set - LR
best_lr = regressiongrid.best_estimator_
regressionprediction = best_lr.predict(X_test)
print("Test accuracy - Logistic Regression:", accuracy_score(y_test, regressionprediction))

#2. Support Vector Machine Classifier with Hyperparameter Tuning
from sklearn.svm import SVC

#Pipeline
svmpipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", decision_function_shape="ovr", random_state=42))
])

#Parameter grid
svmparams = {
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": ["scale", 0.1, 0.01, 0.001],
    "clf__class_weight": [None, "balanced"],
}

#Grid search
svmgrid = GridSearchCV(
    svmpipeline,
    svmparams,
    cv=cv,
    n_jobs=-1,
    scoring="f1_macro",   
    refit=True,
    verbose=2
)

svmgrid.fit(X_train, y_train)

print("Best CV score - SVM:", svmgrid.best_score_)
print("Best params - SVM:", svmgrid.best_params_)

# Evaluate on test set - SVM
best_svm = svmgrid.best_estimator_
svmprediction = best_svm.predict(X_test)
print("Test accuracy - SVM:", accuracy_score(y_test, svmprediction))

#3. Decision Tree Classifier with Hyperparameter Tuning
from sklearn.tree import DecisionTreeClassifier

dtclassifier = DecisionTreeClassifier(random_state=42)

#Parameter grid
dtparameters = {
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"],
}

#Grid search
dtgrid = GridSearchCV(
        dtclassifier,
        dtparameters, 
        cv=cv, 
        n_jobs=-1,
        scoring="f1_macro", 
        refit=True, 
        verbose=0
        )


dtgrid.fit(X_train, y_train)

print("Best CV score - DT:", dtgrid.best_score_)
print("Best params - DT:", dtgrid.best_params_)

# Evaluate on test set - DT
best_dt = dtgrid.best_estimator_
dtprediction = best_dt.predict(X_test)
print("Test accuracy - DT:", accuracy_score(y_test, dtprediction))

#4. Random Forest Classifier with Randomized Tuning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

randomforestmodel = RandomForestClassifier(random_state=42)

#Parameter grid
param_rf = {
    "n_estimators": [100, 200, 300, 400, 600],
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "criterion": ["gini", "entropy"],
}

#Randomized search
randomsearch = RandomizedSearchCV(
    randomforestmodel, param_rf, n_iter=30, cv=cv, scoring="f1_macro",
    random_state=42, n_jobs=-1, verbose=0, refit=True
)

randomsearch.fit(X_train, y_train)

print("Best CV score - RF:", randomsearch.best_score_)
print("Best params - RF:", randomsearch.best_params_)

# Evaluate on test set - RF
best_rf = randomsearch.best_estimator_
rfprediction = best_rf.predict(X_test)
print("Test accuracy - RF:", accuracy_score(y_test, rfprediction))










