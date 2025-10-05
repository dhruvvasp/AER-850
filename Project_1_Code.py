#Step 1 - Reading Data and Converting into Dataframe

import pandas as pd

data = pd.read_csv("Project 1 Data.csv")

# Print first 50 rows of the dataframe
print(data.head(50)) 

# Print the shape of the dataframe
print(data.shape) 

# Print unique counts excluding NaN
print("\nUnique counts by column:") 
print(data.nunique())

#Step 2 - Visualizing Data by Creating Plots

import matplotlib.pyplot as plt

# Histogram of Data:
# Provides a quick overview of the distribution of values in each column
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

# Scatter Plots for Pairwise Relationships:
# Useful for identifying correlations or patterns between pairs of variables
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

# Box Plots for Distribution by Category
# Useful for comparing distributions of X, Y, Z across different Steps
steps = sorted(data['Step'].unique())

x_by_step = [data.loc[data['Step'] == s, 'X'] for s in steps]
y_by_step = [data.loc[data['Step'] == s, 'Y'] for s in steps]
z_by_step = [data.loc[data['Step'] == s, 'Z'] for s in steps]

plt.figure(figsize=(15, 5))

# Plotting box plots for X, Y, Z by Step
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
# Train and test sets will have similar distribution of the target variable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Model Training and Evaluation
# Using a simple Logistic Regression model for classification

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score

print("\nLogistic Regression Classifier Results:")

# Pipeline setup
clf1 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

# Hyperparameter grid for tuning
# Using a simple grid for demonstration; can be expanded based on needs
param_grid = {
    "clf__solver": ["lbfgs"],      
    "clf__penalty": ["l2"],        
    "clf__C": [0.01, 0.1, 1, 10],  
    "clf__class_weight": [None, "balanced"],  
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=clf1,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=2,
    refit=True
)

# Fit grid on training data
grid.fit(X_train, y_train)

print("Best CV accuracy:", grid.best_score_)
print("Best Hyperparameters:", grid.best_params_)

# Use the best model to evaluate
bestpipeline = grid.best_estimator_
lRpredictions = bestpipeline.predict(X_test)

print("Logistic Regression Training Accuracy:", bestpipeline.score(X_train, y_train))
print("Logistic Regression Test Accuracy:",  bestpipeline.score(X_test, y_test))

# (Optional) also print via accuracy_score
print("Test Accuracy (accuracy_score):", accuracy_score(y_test, lRpredictions))