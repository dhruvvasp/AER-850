#################################
# AER 850 - Project 1 Code
#################################

#Step 1 - Reading Data and Converting into Dataframe
import pandas as pd

data = pd.read_csv("Project 1 Data.csv")


#Print first 50 rows of the dataframe
print(data.head(10)) 

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

#Data Splitting
X = data[["X", "Y", "Z"]].values
y = data["Step"].values

from sklearn.model_selection import train_test_split

#Stratified split (80/20)
#Train and test sets will have similar distribution of the target variable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape} samples")
print(f"Testing set size: {X_test.shape} samples")



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
    ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42))
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

print("Best CV score - Logistic Regression:", regressiongrid.best_score_)
print("Best params - Logistic Regression:", regressiongrid.best_params_)

#Evaluate on test set - LR
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
    "clf__C": [0.1, 1, 10, 100, 1000],
    "clf__gamma": ["scale", 0.1, 0.01, 0.001],
    "clf__kernel": ["rbf"],
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

#Evaluate on test set - SVM
best_svm = svmgrid.best_estimator_
svmprediction = best_svm.predict(X_test)
print("Test accuracy - SVM:", accuracy_score(y_test, svmprediction))

#3. Decision Tree Classifier with Hyperparameter Tuning
from sklearn.tree import DecisionTreeClassifier

dtclassifier = DecisionTreeClassifier(random_state=42)

#Parameter grid
dtparameters = {
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_split": [10, 20],
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

#Evaluate on test set - DT
best_dt = dtgrid.best_estimator_
dtprediction = best_dt.predict(X_test)
print("Test accuracy - DT:", accuracy_score(y_test, dtprediction))

#4. Random Forest Classifier with Randomized Tuning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

randomforestmodel = RandomForestClassifier(random_state=42)

#Parameter grid
param_rf = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", None],
    "criterion": ["gini"],
}

#Randomized search
randomsearch = RandomizedSearchCV(
    randomforestmodel, param_rf, n_iter=30, cv=cv, scoring="f1_macro",
    random_state=42, n_jobs=-1, verbose=0, refit=True
)

randomsearch.fit(X_train, y_train)

print("Best CV score - RF:", randomsearch.best_score_)
print("Best params - RF:", randomsearch.best_params_)

#Evaluate on test set - RF
best_rf = randomsearch.best_estimator_
rfprediction = best_rf.predict(X_test)
print("Test accuracy - RF:", accuracy_score(y_test, rfprediction))

#Step 5 - Summary of Model Performance
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay
)

import numpy as np

#Collect the tuned best estimators (from Step 4)
models = {
    "Logistic Regression": best_lr,
    "SVM (RBF)":           best_svm,
    "Decision Tree":       best_dt,
    "Random Forest":       best_rf,
}

#Evaluate on the held-out test set
rows = []
pred_map = {}

for name, est in models.items():
    y_pred = est.predict(X_test)
    pred_map[name] = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    rows.append({
        "Model": name,
        "Accuracy": acc,
        "Precision_macro": prec,
        "Recall_macro": rec,
        "F1_macro": f1
    })

summary_df = pd.DataFrame(rows).sort_values("F1_macro", ascending=False)

print("\nTest-set Performance Summary")
print(summary_df.round(3).to_string(index=False))

#Pick winner by macro-F1 and plot confusion matrix
best_name = summary_df.iloc[0]["Model"]
print(f"\nSelected model (by macro-F1): {best_name}")

labels = np.sort(np.unique(y_test))
cm = confusion_matrix(y_test, pred_map[best_name], labels=labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, cmap="Blues", colorbar=True)
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig("Project 1 Confusion Matrix for Best Model.png", dpi=150)
plt.show()

#Step 6 - Stacked Model Performance Analysis

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.base import clone

#Collecting the tuned best estimators (from Step 4)
models = {
    "Logistic Regression": best_lr,
    "SVM (RBF)":           best_svm,
    "Decision Tree":       best_dt,
    "Random Forest":       best_rf,
}

#Picking top-2 by macro-F1 from Step 5 summary
metric_col = "F1_macro" if "F1_macro" in summary_df.columns else "CV_F1_macro"
ranked = summary_df[["Model", metric_col]].values.tolist()
ranked.sort(key=lambda x: x[1], reverse=True)

if len(ranked) < 2:
    raise ValueError("Need at least two models to stack. Check summary_df contents.")

top1, top2 = ranked[0][0], ranked[1][0]
print(f"\n[Stacking] Base models: {top1} + {top2}")

#Building the stack — clone tuned estimators so StackingClassifier can fit cleanly
base1 = clone(models[top1])
base2 = clone(models[top2])

stack = StackingClassifier(
    estimators=[("m1", base1), ("m2", base2)],
    final_estimator=LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42),
    passthrough=False,
    n_jobs=-1
)

#Fitting on train and evaluate on test
stack.fit(X_train, y_train)
y_pred_stack = stack.predict(X_test)

acc  = accuracy_score(y_test, y_pred_stack)
prec = precision_score(y_test, y_pred_stack, average="macro", zero_division=0)
rec  = recall_score(y_test, y_pred_stack, average="macro", zero_division=0)
f1m  = f1_score(y_test, y_pred_stack, average="macro", zero_division=0)

print(f"\nStacked ({top1} + {top2}): "
      f"Test Acc: {acc:.3f} | Prec(m): {prec:.3f} | Rec(m): {rec:.3f} | F1(m): {f1m:.3f}")

#Compare vs best single model by the same metric used to rank
best_single_name = ranked[0][0]
best_single_f1   = ranked[0][1]
delta_f1 = f1m - best_single_f1
print(f"F1_macro vs best single ({best_single_name}): {delta_f1:+.3f}")

#Confusion matrix for the stack
labels = np.unique(y_test)
cm = confusion_matrix(y_test, y_pred_stack, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, cmap="Blues", colorbar=True)
plt.title(f"Confusion Matrix — Stacked ({top1} + {top2})")
plt.tight_layout()
plt.savefig("Project 1 Stacked Confusion Matrix.png", dpi=150)
plt.show()

#Step 7 - Model Packaging and Prediction

from joblib import dump, load

#Packaging the best single model chosen in Step 5
#Model chosen is Logistic Regression
final_model = best_lr   # can also be best_svm / best_dt / best_rf
print(f"\nPackaging model: {final_model}")


dump(final_model, "Project1_Selected_Model.joblib")
print("Saved: Project1_Selected_Model.joblib")

#Load (sanity check)
model = load("Project1_Selected_Model.joblib")

#Predict for given coordinates [X, Y, Z]
to_predict = np.array([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0.000, 3.0625, 1.93],
    [9.400, 3.0000, 1.80],
    [9.400, 3.0000, 1.30],
])

pred_steps = model.predict(to_predict)

#Display and save predictions
out = pd.DataFrame(to_predict, columns=["X","Y","Z"])
out["Predicted Step"] = pred_steps
print(out.to_string(index=False))
out.to_csv("Project1_Selected_Model_Predictions.csv", index=False)
print("Saved: Project1_Selected_Model_Predictions.csv")