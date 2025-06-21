import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load CSVs
df1 = pd.read_csv('Data\\Dietician1.csv')
df2 = pd.read_csv('Data\\Dietician2.csv')
df3 = pd.read_csv('Data\\Dietician3.csv')
df4 = pd.read_csv('Data\\Dietician4.csv')
df5 = pd.read_csv('Data\\Dietician5.csv')
df6 = pd.read_csv('Data\\Dietician6.csv')

# Rename labels
df1 = df1.rename(columns={'label1': 'label'})
df2 = df2.rename(columns={'label2': 'label'})

# Apply classification
def classify_label(label):
    if label in ['Good', 'Healthy']:
        return 'Healthy'
    elif label in ['Moderate', 'Fair']:
        return 'Moderate'
    elif label in ['Unhealthy', 'Hazardous', 'Dangerous', 'Harmful', 'Poor']:
        return 'Unhealthy'
    return 'Moderate'

df5['label'] = df5['label'].apply(classify_label)
df6['label'] = df6['label'].apply(classify_label)

# Merge Dietician 1 & 2
merged_df_1_2 = df1.copy()
numeric_cols = ['salt_100g', 'sugars_100g', 'saturated-fat_100g', 'fiber_100g', 'proteins_100g', 'additives_n']
for col in numeric_cols:
    merged_df_1_2[col] = (df1[col] + df2[col]) / 2
merged_df_1_2['label'] = np.where(df1['label'] == df2['label'], df1['label'], 'Moderate')

# Merge Dietician 3 & 4
merged_df_3_4 = df3.copy()
for col in numeric_cols:
    merged_df_3_4[col] = (df3[col] + df4[col]) / 2
merged_df_3_4['label'] = np.where(df3['label'] == df4['label'], df3['label'], 'Moderate')

# Select top 200
merged_df_1_2_200 = merged_df_1_2.head(200)
merged_df_3_4_200 = merged_df_3_4.head(200)

# Concatenate all
final_merged_df = pd.concat([merged_df_1_2_200, merged_df_3_4_200, df5, df6], ignore_index=True)
final_merged_df.to_csv("final_merged_resolved_data.csv", index=False)

# Encode labels
le = LabelEncoder()
final_merged_df['encoded_label'] = le.fit_transform(final_merged_df['label'])

# Features and target
X = final_merged_df[numeric_cols]
y = final_merged_df['encoded_label']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Models
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=5000),
    'SVM': SVC()
}

# Hyperparameters
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

param_grid_lr = {
    'C': [0.1, 1],
    'solver': ['liblinear'],
    'max_iter': [2000]
}

param_grid_svm = {
    'C': [0.1, 1],
    'kernel': ['linear'],
    'gamma': ['scale']
}

kf = StratifiedKFold(n_splits=70, shuffle=True, random_state=42)

# Grid search
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    grid_search_rf = GridSearchCV(models['RandomForest'], param_grid_rf, cv=kf, n_jobs=-1)
    grid_search_rf.fit(X_scaled, y)
    best_rf_model = grid_search_rf.best_estimator_

    grid_search_lr = GridSearchCV(models['LogisticRegression'], param_grid_lr, cv=kf, n_jobs=-1)
    grid_search_lr.fit(X_scaled, y)
    best_lr_model = grid_search_lr.best_estimator_

    grid_search_svm = GridSearchCV(models['SVM'], param_grid_svm, cv=kf, n_jobs=-1)
    grid_search_svm.fit(X_scaled, y)
    best_svm_model = grid_search_svm.best_estimator_

# Final model evaluation
models = {'RandomForest': best_rf_model, 'LogisticRegression': best_lr_model, 'SVM': best_svm_model}
metrics_data = []

for name, model in models.items():
    accuracies, precisions, recalls, f1s, kappas = [], [], [], [], []

    for train_index, test_index in kf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='weighted'))
        recalls.append(recall_score(y_test, y_pred, average='weighted'))
        f1s.append(f1_score(y_test, y_pred, average='weighted'))
        kappas.append(cohen_kappa_score(y_test, y_pred))

    metrics_data.append({
        'Model': name,
        'Accuracy': np.mean(accuracies),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'F1': np.mean(f1s),
        'Kappa': np.mean(kappas)
    })

    print(f"\n{name} Results:")
    print(f"Accuracy: {np.mean(accuracies):.4f}")
    print(f"Precision: {np.mean(precisions):.4f}")
    print(f"Recall: {np.mean(recalls):.4f}")
    print(f"F1 Score: {np.mean(f1s):.4f}")

# Save final model
best_rf_model.fit(X_scaled, y)
joblib.dump(best_rf_model, "random_forest_model.pkl")
print("\nRandom Forest model saved as 'random_forest_model.pkl'.")

# Confusion matrix and kappa
y_pred_rf = best_rf_model.predict(X_scaled)
cm = confusion_matrix(y, y_pred_rf)
kappa_model = cohen_kappa_score(y, y_pred_rf)
print("\nConfusion Matrix:\n", cm)
print("Cohen's Kappa (model vs. actual):", kappa_model)

# ==================================
# GRAPH 1: Cohen's Kappa Plot
# ==================================
kappa_df = pd.DataFrame(metrics_data)
plt.figure(figsize=(6, 4))
sns.barplot(data=kappa_df, x='Model', y='Kappa', palette='viridis')
plt.title("Cohen’s Kappa Score per Model")
plt.ylabel("Kappa Score")
plt.tight_layout()
plt.show()

# ==================================
# GRAPH 2: Confusion Matrix Heatmap
# ==================================
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix: Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ==================================
# GRAPH 3: Feature Importance Bar Chart
# ==================================
importances = best_rf_model.feature_importances_
feature_df = pd.DataFrame({'Feature': numeric_cols, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 4))
sns.barplot(data=feature_df, x='Importance', y='Feature', palette='magma')
plt.title("Feature Importances from Random Forest")
plt.tight_layout()
plt.show()

# ==================================
# GRAPH 4: Performance Metrics Bar Chart
# ==================================
metrics_df = pd.DataFrame(metrics_data)
metrics_melted = metrics_df.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1'])

plt.figure(figsize=(10, 5))
sns.barplot(data=metrics_melted, x='Model', y='value', hue='variable')
plt.title("Model Performance Metrics")
plt.ylabel("Score")
plt.tight_layout()
plt.show()

# ==================================
# GRAPH 5: Sample Predictions Table
# ==================================
sample_df = final_merged_df.sample(5).copy()
sample_X = scaler.transform(sample_df[numeric_cols])
sample_df['ML Prediction'] = le.inverse_transform(best_rf_model.predict(sample_X))

print("\nSample Predictions Comparison:\n")
print(sample_df[['salt_100g', 'sugars_100g', 'saturated-fat_100g', 'fiber_100g', 'proteins_100g', 'additives_n', 'label', 'ML Prediction']])
