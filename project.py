import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

df = pd.read_csv('college_student_placement_dataset.csv')

df.columns = df.columns.str.replace('_$', '', regex=True)

for col in df.columns:
    if df[col].dropna().isin(['Yes', 'No']).all():
        df[col] = df[col].map({'Yes': 1, 'No': 0})

print("Columns in DataFrame:", df.columns.tolist())
print("--- Initial Data Overview ---")
print(df.head())
print("\n--- Data Info ---")
df.info()

print("\n--- Starting Exploratory Data Analysis (EDA) ---")

sns.set_style("whitegrid")

plt.figure(figsize=(6, 5))
sns.countplot(x='Placement', data=df, palette='viridis')
plt.title('Distribution of Placements (1: Placed, 0: Not Placed)', fontsize=14)
plt.show()

plt.figure(figsize=(12, 10))
correlation_matrix = df.drop('College_ID', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Features', fontsize=16)
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Placement', y='CGPA', data=df, palette='magma')
plt.title('CGPA Distribution by Placement Status', fontsize=14)
plt.show()

plt.figure(figsize=(7, 6))
sns.countplot(x='Internship_Experience', hue='Placement', data=df, palette='pastel')
plt.title('Placement Status by Internship Experience', fontsize=14)
plt.legend(title='Placed', labels=['Not Placed', 'Placed'])
plt.show()

print("\n--- Preparing Data for Modeling ---")

X = df.drop(['Placement', 'College_ID'], axis=1)
y = df['Placement']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

print("\n--- Training and Evaluating Models ---")

print("\n--- Logistic Regression ---")
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("AUC Score:", roc_auc_score(y_test, y_pred_log_reg))
print("Classification Report:\n", classification_report(y_test, y_pred_log_reg))

print("\n--- Random Forest Classifier ---")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("AUC Score:", roc_auc_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 7))
sns.barplot(x='importance', y='feature', data=feature_importances, palette='rocket')
plt.title('Feature Importance from Random Forest', fontsize=16)
plt.show()

print("\n--- Project Complete ---")
print("The Random Forest model shows which features are most predictive of placements.")
