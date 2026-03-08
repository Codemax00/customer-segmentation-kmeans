import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load Data
df = pd.read_csv('Mall_Customers_Clustered.csv')

# --- 1. Feature Engineering ---
print("--- Feature Engineering ---")
# Create new meaningful variables
df['Income_to_Age_Ratio'] = df['Annual Income (k$)'] / df['Age']
df['Spending_to_Income_Ratio'] = df['Spending Score (1-100)'] / df['Annual Income (k$)']

# Encode categorical variable Gender to numeric
df['Gender_Male'] = df['Gender'].map({'Male': 1, 'Female': 0})

X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Income_to_Age_Ratio', 'Spending_to_Income_Ratio', 'Gender_Male']]
y = df['Cluster']

# --- 2. Data Splitting ---
print("\n--- Data Splitting ---")
# Split total dataset: 15% Test, 85% remaining for Train & Validation
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Split remaining 85%: ~70% of total for Train and ~15% of total for Validation
# test_size here should be 15/85 = 0.17647
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.17647, random_state=42, stratify=y_temp)

print(f"Total dataset size: {len(X)}")
print(f"Train subset size: {len(X_train)} ({len(X_train)/len(X):.0%})")
print(f"Validation subset size: {len(X_val)} ({len(X_val)/len(X):.0%})")
print(f"Test subset size: {len(X_test)} ({len(X_test)/len(X):.0%})")

# Scale/normalize numerical data (fit only on training data to prevent data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# --- 3. Model Selection and Training ---
print("\n--- Model Training & Optimization ---")
# We start with Logistic Regression and use Validation set for manual Grid Search Hyperparameter tuning
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['lbfgs'], 'max_iter': [1000]}
grid = list(ParameterGrid(param_grid))

best_score = 0
best_params = None
best_model = None

print("Grid Search for Hyperparameter Tuning on Validation Set:")
for params in grid:
    model = LogisticRegression(**params, random_state=42, multi_class='multinomial')
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on Validation set to check for overfitting
    val_preds = model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_preds)
    
    train_preds = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_preds)
    
    print(f"Params: C={params['C']:<5} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    # Keep track of best model parameters
    if val_acc > best_score:
        best_score = val_acc
        best_params = params
        best_model = model

print(f"\nBest parameters found: {best_params} with Validation Accuracy: {best_score:.4f}")

# Optional: retrain final model on combined Train+Validation sets for final evaluation
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

X_train_val_scaled = scaler.fit_transform(X_train_val)
X_test_scaled_final = scaler.transform(X_test)

final_model = LogisticRegression(**best_params, random_state=42, multi_class='multinomial')
final_model.fit(X_train_val_scaled, y_train_val)

# Perform Cross-Validation on the combined subset to ensure model generalizes well
cv_scores = cross_val_score(final_model, X_train_val_scaled, y_train_val, cv=5)
print(f"\n5-Fold Cross-Validation Accuracy on Train+Val Set: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# --- 4. Final Evaluation ---
print("\n--- Final Evaluation on Test Set ---")
y_test_pred = final_model.predict(X_test_scaled_final)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Test Set Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues')
plt.title('Confusion Matrix on Test Set')
plt.savefig('confusion_matrix.png')
plt.close()

print("Model pipeline successfully completed! Confusion matrix saved to confusion_matrix.png")
